import logging
import os
from functools import wraps
from typing import Any, Dict

import torch
import torch.fx as fx
import torch.multiprocessing as mp
import torch.nn as nn

from graph_prof_solution import GraphProfiler
from graph_tracer import SEPFunction, compile
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs

# This is the dummy model that is for use in starter code. But we will
# experiment with Resnet and Bert models from Torch Benchmark suite.


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


# Anymodel that is used will be wrapped with this model. We do this to call a
# dummy function 'SEPFunction', which is the separator function, that will call
# an identity operator at the end of the forward pass. This identity operator
# will get recorded in the computational graph and will inform you where the
# backward pass ends.


class WrappedDummyModel(nn.Module):
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return SEPFunction.apply(self.mod(x))


# This is the train_step function that takes in a model, optimizer and an input
# mini batch and calls the forward pass, loss function and the optimizer step. A
# computational graph corresponding to a train_step will be captured by the
# compiler.


def train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor
):
    out: torch.Tensor = model(batch)
    out.sum().backward()
    optim.step()
    optim.zero_grad()


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    name_to_node = {}
    for node in gm.graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    old_node_users = old_node.users
    for node in reversed(graph.nodes):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def activation_checkpointing(gm: fx.GraphModule, graph_profiler: GraphProfiler, max_peak_memory: int) -> fx.GraphModule:
    # NOTE: You need to create the function for your project and call it inside
    # the graph_transformation function after performing graph profiling.

    # In this example we are going to recompute one of the relu activations for the
    # backward pass instead of saving it. We know from our custom function
    # that we have 2 intermeidate nodes: ['relu', 'relu_1']

    # So the intermediate node to recompute is: ['relu'] and
    # intermediate nodes to checkpoint (retain) are: ['relu_1']

    # Nodes required to recompute 'relu' are ['w1_1', 'x_1']
    # First back use is at node 't'

    # NOTE: For your project, you will use GraphProfiler to identify the
    # intermediate nodes, their first back access, last forward access and
    # then MuTWO's algorithm to select the intermediate 'nodes_to_recompute' and
    # checkpoint (retain). The 'nodes_required_to_recompute' any of the
    # intermediate nodes MUST be a subset of the placeholder nodes and the
    # intermediate nodes that are checkpointed.

    # TODO 1: Where to get max_peak_memory from ? profiler ?
    # TODO 2: What values of memory_limit to experiment ?
    # TODO 3: Is candidate set the set of all graph nodes


    nodes_required_to_recompute, nodes_to_recompute = graph_profiler.algorithm_b_recomputation_policy(
        candidate_set=graph_profiler.intermediate_nodes,
        memory_limit=11000000,
        max_peak_memory=max_peak_memory,
    )

    name_to_node = get_name_to_node_map(gm)
    # first_back_access = name_to_node["t"]
    # node_to_recompute = [name_to_node["relu"]]
    node_to_recompute_names = [node.name for node in nodes_to_recompute]
    # nodes_required_to_recompute = [name_to_node["w1_1"], name_to_node["x_1"]]

    # # NOTE: we cannot directly use 'mm' to recompute 'relu' since 'mm' is not an
    # # intermediate node that is retained (checkpointed).

    # Obtain a sub-graph that recomputes the required nodes
    for recomp_node in nodes_to_recompute:
      recompute_subgraph = _extract_graph_with_inputs_outputs(
          joint_graph=gm.graph,
          inputs=graph_profiler.node_info[recomp_node].recomp_srcs,
          outputs=[recomp_node],
      )
      recompute_subgraph.print_tabular()

      first_back_access = graph_profiler.node_info[recomp_node].first_back_access

      # Insert the nodes of the new sub-graph in the old graph before the first
      # backward access of the node to be recomputed.
      with gm.graph.inserting_before(first_back_access):
          for n in recompute_subgraph.nodes:
              if n.op == "placeholder" or n.op == "output":
                  continue
              # Copy the nodes of the new sub-graph to old graph and transform its
              # inputs to match the old-graph inputs. The arg_transform function
              # will pass the input arguments of the new node and will expect a
              # mapping to the nodes of the old graph.
              new_node = gm.graph.node_copy(
                  n, arg_transform=lambda arg: name_to_node[arg.name]
              )

              if n.name in node_to_recompute_names:
                  old_node = name_to_node[n.name]
                  # Replace all the uses of the old node with new recomputation node
                  replace_subsequent_uses_of(
                      gm.graph, old_node=old_node, new_node=new_node
                  )
              # Add the new node to our name to node mapping
              name_to_node[n.name] = new_node

    gm.graph.lint()
    gm.recompile()

    print("graph tabular 2 start")
    print(gm.graph.print_tabular())
    print("graph tabular 2 end")

    return gm


# Below is a user defined function that accepts a graph module and arguments of
# used to run the graph. You can essentially do any operation, graph
# modification, profiling etc. inside this function. Subsequent to modifications
# or graph analysis, the function expects you to return the modified graph back.
# In the given example, we just print the graph, and then initilize the graph
# profiler. The graph profiler extends the class fx.Interpreter, that allows you
# to run the graph node by node, more explanation in graph_prof.py.


def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
    graph_profiler = GraphProfiler(gm)
    warm_up_iters, profile_iters = 2, 3
    with torch.no_grad():
        for _ in range(warm_up_iters):
            graph_profiler.run(*args)
        graph_profiler.reset_stats()
        for _ in range(profile_iters):
            graph_profiler.run(*args)
    graph_profiler.aggregate_stats()
    graph_profiler.print_stats()
    gm = activation_checkpointing(gm, graph_profiler)
    return gm


# We first initialize the model, pass it to the wrapper model, then create a
# random input mini-batch and initilize the optimizer. We then call the compile
# function that takes in two arguments, a train_step function and a
# graph_transformation function. The train_step function is the one that will be
# traced by the compiler and a computational graph for the same will be created.
# This computational graph is then passed to the graph_transformation function
# to do any graph profiling, modifications and optimizations. This modified
# graph is stored and will be returned as the compiled function. In essence we
# do the following inside the compile function:

# def compile (train_step, graph_transformation):
#     @wraps(train_step)
#     def inner(*args, **kwargs):
#         if not_compiled:
#             original_graph, input_args = graph_tracer(train_step)
#             modified_graph = graph_transformation(original_graph, input_args)
#         output = modified_graph(*args, **kwargs)
#         return output
#     return inner


def experiment():
    logging.getLogger().setLevel(logging.DEBUG)
    torch.manual_seed(20)
    batch_size = 1000
    layers = 10
    dim = 100
    num_iters = 5
    dummy_model = DummyModel(dim=dim, layers=layers)
    model = WrappedDummyModel(dummy_model).cuda()
    batch = torch.randn(batch_size, dim).cuda()
    optim = torch.optim.Adam(
        model.parameters(), lr=0.01, foreach=False, fused=True, capturable=True
    )

    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param)
    optim.step()
    optim.zero_grad()

    compiled_fn = compile(train_step, graph_transformation)
    compiled_fn(model, optim, batch)


if __name__ == "__main__":
    experiment()
