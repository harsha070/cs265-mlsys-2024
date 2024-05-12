from enum import Enum
from statistics import mean
import math
from dataclasses import dataclass, field
from typing import Dict
import torch
import torch.fx as fx
from typing import Dict, Any, List, cast, Set, Tuple
import tabulate
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
import pandas as pd

# Minimum memory allocated by PyTorch for a tensor, change according to your device type
_PYTORCH_MIN_ALLOCATE = 512


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


class MemStats:
    def __init__(
        self,
        param_and_opt_state_mem: int,
        grad_mem: int,
        act_mem: int,
        other_mem: int,
    ) -> None:
        self.param_and_opt_state_memory = param_and_opt_state_mem
        self.grad_memory = grad_mem
        self.activation_memory = act_mem
        self.other_memory = other_mem


@dataclass
class NodeInfo:
    rank: int = 0
    node_type: NodeType = None
    run_time: float = 1.0
    swap_time: float = 0.0
    peak_total_mem: int = 0
    mem_stats = None
    memory_size: int = 0
    cpu_ref: torch.Tensor = None
    last_forward_access: fx.Node = None
    first_back_access: fx.Node = None
    last_forward_uses: List[fx.Node] = field(default_factory=list)
    first_back_uses: List[fx.Node] = field(default_factory=list)
    # You can add more attributes to this class for applying the recomputation algorithm
    recomp_srcs: Set[fx.Node] = field(default_factory=set)
    recomp_time: float = 0.0
    total_recomp_time: float = 0.0
    recompute_ratio: float = 0.0
    active_memory: int = 0.0


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(
        self, module: fx.GraphModule, garbage_collect_values: bool = True
    ):
        super().__init__(module, garbage_collect_values)

        self.module = module
        self.node_info: Dict[fx.Node, NodeInfo] = {}
        self.intermediate_nodes: List[fx.Node] = []
        self.node_runtimes: Dict[fx.Node, List[float]] = {}
        self.node_swap_times: Dict[fx.Node, List[float]] = {}
        self.forward_end: fx.Node
        self.backward_start: fx.Node
        self.swapped_memory: int = 0
        self.param_and_opt_state_memory: int

        rank = 0
        for node in self.module.graph.nodes:
            n_info = NodeInfo()
            n_info.rank = rank
            # Initially set the node types of all nodes to other
            n_info.node_type = NodeType.OTHER
            rank += 1
            self.node_info[node] = n_info
            # Find the forward end and backward start dummy nodes
            if (
                node.name == "sep"
                and node.target == torch.ops.separator.sep.default
            ):
                self.forward_end = node
            elif (
                node.name == "sep_backward"
                and node.target == torch.ops.separator.sep_backward.default
            ):
                self.backward_start = node

            # Use the optimizer to get the parameter and gradient nodes

            if node.target == torch.ops.aten._fused_adam.default:
                param_adam_args = node.args[0]
                grad_adam_args = node.args[1]

                assert len(param_adam_args) == len(
                    grad_adam_args
                ), "Unequal number of params and gradients"

                for param in param_adam_args:
                    assert isinstance(
                        param, fx.Node
                    ), "Expected param to be an fx.Node instance"
                    assert (
                        param.op == OP.PLACEHOLDER
                    ), "Expected all params nodes to be of type PLACEHOLDER"
                    self.node_info[param].node_type = NodeType.PARAM

                for grad in grad_adam_args:
                    assert isinstance(
                        grad, fx.Node
                    ), "Expected grad to be an fx.Node instance"
                    self.node_info[grad].node_type = NodeType.GRAD

        for node in self.module.graph.nodes:
            if (
                node.op != OP.PLACEHOLDER
                and self.node_info[node].rank
                < self.node_info[self.forward_end].rank
            ):
                input_nodes: List[fx.Node] = node.all_input_nodes
                input_nodes_op: List[bool] = [
                    self.node_info[n].node_type == NodeType.PARAM
                    for n in input_nodes
                ]
                if all(input_nodes_op):
                    self.node_info[node].node_type = NodeType.PARAM
                    continue
                users = node.users
                # from the users we get the last forward use
                # and the first backward use using ranks
                last_forward = None
                first_backward = None
                for user in users:
                    u_info = self.node_info[user]
                    if u_info.rank < self.node_info[self.forward_end].rank:
                        if last_forward is None:
                            last_forward = user
                        elif self.node_info[last_forward].rank < u_info.rank:
                            last_forward = user
                    if u_info.rank > self.node_info[self.backward_start].rank:
                        if first_backward is None:
                            first_backward = user
                        elif self.node_info[first_backward].rank > u_info.rank:
                            first_backward = user
                if last_forward is not None and first_backward is not None:
                    n_info = self.node_info[node]
                    self.intermediate_nodes.append(node)
                    n_info.node_type = NodeType.ACT
                    self.node_info[last_forward].last_forward_uses.append(node)
                    self.node_info[first_backward].first_back_uses.append(node)
                    n_info.first_back_access = first_backward
                    n_info.last_forward_access = last_forward
                    # print(
                    #     f"Intermediate Node: {node.name}, Last forward use: {last_forward.name}, First backward use: {first_backward.name}"
                    # )

    def _swap_out_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be offloaded
        # 2) Retrieve their CPU reference (if none allocate a CPU tensor in
        #    pinned memory)
        # 3) Copy the tensor to the CPU, add the CPU tensor to the Interpreter
        #    environment
        # 4) Delete the GPU tensor
        nodes_to_offload = self.node_info[node].last_forward_uses
        for o_node in nodes_to_offload:
            o_info = self.node_info[o_node]
            cpu_ref = o_info.cpu_ref
            tensor = self.env[o_node]
            assert isinstance(tensor, torch.Tensor)
            if cpu_ref is None:
                cpu_ref = torch.zeros(
                    tensor.size(), dtype=tensor.dtype, layout=tensor.layout
                ).pin_memory()
            assert cpu_ref.is_pinned, f"CPU ref is not pinned for {o_node.name}"

            swap_start_event = torch.cuda.Event(enable_timing=True)
            swap_end_event = torch.cuda.Event(enable_timing=True)

            swap_start_event.record()
            cpu_ref = cpu_ref.copy_(tensor, False)
            swap_end_event.record()

            torch.cuda.synchronize()
            o_info.cpu_ref = cpu_ref
            self.env[o_node] = cpu_ref
            del tensor
            tensor = None
            cpu_ref = None
            self.swapped_memory += o_info.memory_size
            swap_time = swap_start_event.elapsed_time(swap_end_event)
            self.node_swap_times.setdefault(o_node, []).append(swap_time)

    def _swap_in_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be prefetched
        # 2) Retrieve their CPU reference (assert if it resides in pinned
        #    memory)
        # 3) Copy the tensor to GPU memory and add it to the Interpreter
        #    environment
        # 4) Update the state of intermediate tensor in NodeInfo
        nodes_to_fetch = self.node_info[node].first_back_uses
        for p_node in nodes_to_fetch:
            p_info = self.node_info[p_node]
            cpu_ref = cast(torch.Tensor, p_info.cpu_ref)

            swap_start_event = torch.cuda.Event(enable_timing=True)
            swap_end_event = torch.cuda.Event(enable_timing=True)

            swap_start_event.record()
            tensor = cpu_ref.to(
                device=torch.cuda.current_device(),
                memory_format=torch.preserve_format,
                non_blocking=False,
            )
            swap_end_event.record()
            self.env[p_node] = tensor.contiguous()
            tensor = None
            torch.cuda.synchronize()
            self.swapped_memory -= p_info.memory_size
            assert (
                self.swapped_memory >= 0
            ), f"Swapped memory is less than zero {self.swapped_memory}"
            swap_time = swap_start_event.elapsed_time(swap_end_event)
            self.node_swap_times.setdefault(p_node, []).append(swap_time)

    def get_total_memory_breakdown(self) -> int:
        grad_mem = 0
        act_mem = 0
        other_mem = 0
        param_and_opt_state_mem = self.param_and_opt_state_memory
        for node in self.env.keys():
            if node.op == OP.PLACEHOLDER:
                continue
            node_type = self.node_info[node].node_type
            memory_size = self.node_info[node].memory_size
            if node_type == NodeType.GRAD:
                grad_mem += memory_size
            elif node_type == NodeType.ACT:
                act_mem += memory_size
            else:
                other_mem += memory_size
        mem_stats = MemStats(
            param_and_opt_state_mem, grad_mem, act_mem, other_mem
        )
        return mem_stats

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> torch.Any:
        self.param_and_opt_state_memory = torch.cuda.memory_allocated()
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    def run_node(self, node: fx.Node) -> Any:
        if node.op == OP.PLACEHOLDER:
            return super().run_node(node)

        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.
        if self.node_info[node].rank > self.node_info[self.backward_start].rank:
            self._swap_in_node(node)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = super().run_node(node)
        end_event.record()
        torch.cuda.synchronize()
        self.env[node] = result
        active_memory = torch.cuda.memory_allocated()

        run_time = start_event.elapsed_time(end_event)
        n_info = self.node_info[node]
        self.node_runtimes.setdefault(node, []).append(run_time)
        n_info.mem_stats = self.get_total_memory_breakdown()
        n_info.peak_total_mem = active_memory + self.swapped_memory
        n_info.active_memory = active_memory

        # Alternate way to measure the memory of the resultant tensor

        if isinstance(result, torch.Tensor):
            size = result.untyped_storage().size()
            element_size = result.untyped_storage().element_size()
            tensor_memory = (
                math.ceil((size * element_size) / _PYTORCH_MIN_ALLOCATE)
                * _PYTORCH_MIN_ALLOCATE
            )
            n_info.memory_size = tensor_memory

        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.
        if self.node_info[node].rank < self.node_info[self.forward_end].rank:
            self._swap_out_node(node)

        return result

    def aggregate_stats(self):
        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            self.node_info[node].run_time = mean(self.node_runtimes[node])

            if node in self.intermediate_nodes:
                self.node_info[node].swap_time = mean(
                    self.node_swap_times[node]
                )

    def reset_stats(self):
        self.node_runtimes.clear()
        self.node_swap_times.clear()

    def print_stats(self, save_path=None):
        headers: List[str] = [
            "Node",
            "Target",
            "Size (B)",
            "Avg runtime (ms)",
            "Peak Memory (B)",
            "Swap Time (ms)",
            "Rank",
            "Param Opt State Memory",
            "Gradient Memory",
            "Activation Memory",
            "Other Memory",
            "Active Memory",
        ]
        node_summaries: List[List[Any]] = []
        node_summaries_data = []

        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            n_info = self.node_info[node]
            val_list = [
                node.name,
                node._pretty_print_target(node.target),
                n_info.memory_size,
                n_info.run_time,
                n_info.peak_total_mem,
            ]
            if node in self.intermediate_nodes:
                val_list.append(str(n_info.swap_time))
            else:
                val_list.append("")
            node_summaries.append(val_list)
            val_list.extend([
                n_info.rank,
                n_info.mem_stats.param_and_opt_state_memory,
                n_info.mem_stats.grad_memory,
                n_info.mem_stats.activation_memory,
                n_info.mem_stats.other_memory,
                n_info.active_memory,
            ])
            node_summaries_data.append(dict(zip(headers, val_list)))
        if save_path is not None:
            node_summaries_data = pd.DataFrame(node_summaries_data)
            node_summaries_data.to_parquet(save_path)
        else:
            print(tabulate.tabulate(node_summaries, headers=headers))

    def get_peak_memory(self) -> int:
        peak_memory = 0.0
        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            n_info = self.node_info[node]
            peak_memory = max(n_info.peak_total_mem, peak_memory)
        return peak_memory

    def algorithm_b_recomputation_policy(self, candidate_set: Set[fx.Node], memory_limit: int, max_peak_memory: int) -> Tuple[Set[fx.Node], Set[fx.Node]]:
        recomps: Set[fx.Node] = set()
        memory_consumption = max_peak_memory
        self.algorithm_d_initialize(candidate_set)
        while len(candidate_set) > 0:
            r_cand = self.algorithm_e_max_candidate(candidate_set=candidate_set)
            recomps.add(r_cand)
            cand = r_cand
            candidate_set.remove(cand)
            recomp_count = self.algorithm_h_update_existing_recomputations(recomps=recomps, candidate=cand)
            self.algorithm_i_update_candidates(t=cand, recomp_count=recomp_count, candidates=candidate_set, recomps=recomps)
            memory_consumption -= self.node_info[cand].memory_size
            # print(f"recomputing node: {cand}. memory consumption down to: {memory_consumption}")
            if (memory_consumption - memory_limit) <= 0:
              break
        return candidate_set, recomps

    def algorithm_d_initialize(self, candidate_set: Set[fx.Node]) -> fx.Node:

        def get_srcs(cand):
            srcs = []
            for node in cand.all_input_nodes:
              if node.op == "placeholder":
                srcs.append(node)
              elif node in candidate_set:
                srcs.append(node)
              else:
                srcs.extend(get_srcs(node))
            return srcs

        for candidate in candidate_set:
            node_info = self.node_info[candidate]
            node_info.recomp_srcs = set(get_srcs(candidate))
            node_info.recomp_time = node_info.run_time
            node_info.total_recomp_time = node_info.recomp_time
            node_info.recompute_ratio = node_info.memory_size / node_info.total_recomp_time

    def algorithm_e_max_candidate(self, candidate_set: Set[fx.Node]) -> fx.Node:
        max_candidate: fx.Node = None
        for candidate in candidate_set:
            if max_candidate is None:
                max_candidate = candidate
            else:
                if self.node_info[max_candidate].recompute_ratio < self.node_info[candidate].recompute_ratio:
                    max_candidate = candidate
        return max_candidate

    def algorithm_g_update_recompute_ratio(self, candidate_set: Set[fx.Node]) -> None:
        for candidate in candidate_set:
            node_info = self.node_info[candidate]
            node_info.recompute_ratio = node_info.memory_size / node_info.total_recomp_time

    def algorithm_h_update_existing_recomputations(self, recomps: Set[fx.Node], candidate: fx.Node) -> int:
        recomp_count = 1
        for recomp in recomps:
            recomp_node_info = self.node_info[recomp]
            if candidate in recomp_node_info.recomp_srcs:
                recomp_node_info.recomp_srcs.remove(candidate)
                recomp_node_info.recomp_srcs.update(self.node_info[candidate].recomp_srcs)
                recomp_node_info.recomp_time += self.node_info[candidate].recomp_time
                recomp_count += 1
        return recomp_count

    def algorithm_i_update_candidates(self, t: fx.Node, recomp_count: int, candidates: Set[fx.Node], recomps: Set[fx.Node]) -> None:
        for candidate in candidates:
            node_info = self.node_info[candidate]
            if t in node_info.recomp_srcs:
                node_info.recomp_srcs.remove(t)
                node_info.recomp_srcs.update(self.node_info[t].recomp_srcs)
                node_info.recomp_time += self.node_info[t].recomp_time
                node_info.total_recomp_time = node_info.recomp_time
                for rp in recomps:
                    if candidate in self.node_info[rp].recomp_srcs:
                        node_info.total_recomp_time += node_info.recomp_time
            if candidate in self.node_info[t].recomp_srcs:
                self.node_info[candidate].total_recomp_time = recomp_count * self.node_info[candidate].recomp_time
            self.algorithm_g_update_recompute_ratio({candidate})

if __name__ == "__main__":
    print("Executing this file")
