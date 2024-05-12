import importlib
from typing import Any, Dict, List
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fx as fx
from torchbenchmark.models import hf_Bert, resnet18, resnet50
from torchbenchmark.util.model import BenchmarkModel
from graph_prof_solution import GraphProfiler
from graph_tracer import SEPFunction, compile
import statistics as stats

model_names: List[str] = [
    "torchbenchmark.models.hf_Bert.Model",
    "torchbenchmark.models.resnet18.Model",
    "torchbenchmark.models.resnet50.Model",
]

actual_model_names: List[str] = [
    "hf_Bert",
    "resnet18",
    "resnet50",
]

model_batch_sizes: Dict[str, int] = {
    "torchbenchmark.models.hf_Bert.Model": [24, 28, 32, 36, 40],
    "torchbenchmark.models.resnet18.Model": [1024, 1088, 1152, 1280, 1344],
    "torchbenchmark.models.resnet50.Model": [256, 280, 320, 352, 384, 400, 416, 432, 448, 464],
}

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

def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm

def activation_checkpointing(gm: fx.GraphModule, graph_profiler: GraphProfiler, memory_limit: int, max_peak_memory: int) -> fx.GraphModule:

    # Using graph profiler, identify which nodes to recompute, and which nodes to store
    nodes_required_to_recompute, nodes_to_recompute = graph_profiler.algorithm_b_recomputation_policy(
        candidate_set=graph_profiler.intermediate_nodes,
        memory_limit=memory_limit,
        max_peak_memory=max_peak_memory,
    )

    name_to_node = get_name_to_node_map(gm)
    node_to_recompute_names = [node.name for node in nodes_to_recompute]

    for recomp_node in nodes_to_recompute:
      # For each node to be recomputed, Obtain a sub-graph that recomputes the required nodes
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
              # print("copied node!")
              new_node = gm.graph.node_copy(
                  n, arg_transform=lambda arg: name_to_node[arg.name]
              )

              if n.name in node_to_recompute_names:
                  old_node = name_to_node[n.name]
                  # Replace all the uses of the old node with new recomputation node
                  # print("removing subsequent uses!")
                  replace_subsequent_uses_of(
                      gm.graph, old_node=old_node, new_node=new_node
                  )
              # Add the new node to our name to node mapping
              name_to_node[n.name] = new_node

    gm = remove_detach_nodes(gm)
    gm.graph.lint()
    gm.recompile()
    return gm


class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        pos = model_name.rfind(".")
        module = importlib.import_module(model_name[:pos])
        model_class = getattr(module, model_name[(pos + 1) :])

        model: BenchmarkModel = model_class(
            "train", "cuda", batch_size=batch_size, extra_args=extra_args
        )
        self.model: nn.Module = model.model
        self.model_type = type(model)
        self.model_name = model_name

        self.batch_size = batch_size
        self.example_inputs = model.example_inputs

        if self.model_type == hf_Bert.Model:

            def bert_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = model(**example_inputs).loss
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = bert_train_step
            self.optimizer: optim.Optimizer = model.optimizer

        elif self.model_type in (resnet18.Model, resnet50.Model):
            self.loss_fn = model.loss_fn
            self.example_inputs = model.example_inputs[0]

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                output = model(example_inputs)
                target = torch.rand_like(output)
                loss = self.loss_fn(output, target)
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer: optim.Optimizer = model.opt
            self.train_step = resnet_train_step

    def init_opt_states(self):
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        warm_up_iters, profile_iters = 2, 3
        graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()

            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()
            graph_profiler.print_stats(save_path=f'{self.model_name}_batch_size_{self.batch_size}_profiler_stats_all.parquet')

        # Additional memory from subgraphs
        additional_sub_graph_memory = graph_profiler.get_peak_memory() - torch.cuda.get_device_properties(0).total_memory
        additional_sub_graph_memory = max(additional_sub_graph_memory, 0.0)
        if self.model_name == 'torchbenchmark.models.hf_Bert.Model':
            additional_sub_graph_memory = (graph_profiler.get_peak_memory() - 22561741312) * 0.80
        if self.model_name == 'torchbenchmark.models.resnet18.Model':
            additional_sub_graph_memory = (graph_profiler.get_peak_memory() - 22561741312) * 0.80
        if self.model_name == 'torchbenchmark.models.resnet50.Model':
            additional_sub_graph_memory = 0.0
        memory_limit = 0.9 * torch.cuda.get_device_properties(0).total_memory - additional_sub_graph_memory

        print(f'peak_memory: {graph_profiler.get_peak_memory()} gpu max: {torch.cuda.get_device_properties(0).total_memory} memory_limit: {memory_limit} additional_sub_graph_memory: {additional_sub_graph_memory}')
        gm = activation_checkpointing(
            gm,
            graph_profiler,
            memory_limit=memory_limit,
            max_peak_memory=graph_profiler.get_peak_memory(),
        )
        return gm

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")



if __name__ == "__main__":

    model_name = str(sys.argv[1])
    batch_size = int(sys.argv[2])
    print(f'model_name: {model_name} batch_size: {batch_size}')
    exp = Experiment(model_name, batch_size)
    exp.init_opt_states()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
    torch.cuda.synchronize()
    print(f"model_name: {model_name} batch_size: {batch_size} profiling complete!")

    num_iters = 3
    run_times = []
    peak_memory = []
    torch.cuda.empty_cache()
    for _ in range(num_iters):
        torch.cuda.reset_peak_memory_stats()
        print(f"model: {model_name} batch_size: {batch_size} iter: {_}")
        start_event = torch.cuda.Event(enable_timing = True)
        end_event = torch.cuda.Event(enable_timing = True)
        start_event.record()
        compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
        end_event.record()
        torch.cuda.synchronize()
        run_times.append(start_event.elapsed_time(end_event))
        peak_memory.append(torch.cuda.max_memory_allocated())
    run_time = stats.mean(run_times)
    print(f'model_name: {model_name} batch_size: {batch_size} peak_memory: {peak_memory[0]} run_time: {run_time}')