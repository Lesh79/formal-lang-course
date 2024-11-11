from dataclasses import dataclass
from typing import Dict, Set, Tuple, Self

import networkx as nx
from pyformlang import rsa
from pyformlang.finite_automaton import Symbol, DeterministicFiniteAutomaton


@dataclass(frozen=True)
class RsmState:
    variable: Symbol
    substate: str


class GSSNode:
    state: RsmState
    node: int
    edges: Dict[RsmState, Set[Self]]
    pop_set: Set[int]

    def __init__(self, state: RsmState, node: int):
        self.state = state
        self.node = node
        self.edges = {}
        self.pop_set = set()

    def pop(self, current_node: int) -> Set[Self]:
        result = set()
        if current_node not in self.pop_set:
            for edge_state, gss_nodes in self.edges.items():
                for gs_node in gss_nodes:
                    result.add(SPPFNode(gs_node, edge_state, current_node))
            self.pop_set.add(current_node)
        return result

    def add_edge(self, return_state: RsmState, pointer: Self) -> Set[Self]:
        result = set()
        state_edges = self.edges.get(return_state, set())
        if pointer not in state_edges:
            state_edges.add(pointer)
            for node in self.pop_set:
                result.add(SPPFNode(pointer, return_state, node))
        self.edges[return_state] = state_edges
        return result


@dataclass(frozen=True)
class SPPFNode:
    gss_node: GSSNode
    state: RsmState
    node: int


class GSStack:
    body: Dict[Tuple[RsmState, int], GSSNode]

    def __init__(self):
        self.body = {}

    def get_node(self, rsm_state: RsmState, node: int):
        node_instance = self.body.get((rsm_state, node))
        if node_instance is None:
            node_instance = GSSNode(rsm_state, node)
            self.body[(rsm_state, node)] = node_instance
        return node_instance


@dataclass
class RsmStateData:
    terminal_edges: Dict[Symbol, RsmState]
    variable_edges: Dict[Symbol, Tuple[RsmState, RsmState]]
    is_final: bool


class GllCFPQSolver:
    def is_terminal(self, symbol: str) -> bool:
        return Symbol(symbol) not in self.rsm_data

    def initialize_graph_data(self, graph: nx.DiGraph):
        for node in graph.nodes():
            self.nodes_edges[node] = {}

        for from_node, to_node, label in graph.edges(data="label"):
            if label is not None:
                edges = self.nodes_edges[from_node]
                nodes_set = edges.get(label, set())
                nodes_set.add(to_node)
                edges[label] = nodes_set

    def initialize_rsm_data(self, rsm: rsa.RecursiveAutomaton):
        for var in rsm.boxes:
            self.rsm_data[var] = {}

        for var, box in rsm.boxes.items():
            fa: DeterministicFiniteAutomaton = box.dfa
            gbox = fa.to_networkx()
            state_dict = self.rsm_data[var]

            for sub_state in gbox.nodes:
                is_final = sub_state in fa.final_states
                state_dict[sub_state] = RsmStateData({}, {}, is_final)

            for from_state, to_state, symbol in gbox.edges(data="label"):
                if symbol is not None:
                    edges_data = state_dict[from_state]
                    if self.is_terminal(symbol):
                        edges_data.terminal_edges[symbol] = RsmState(var, to_state)
                    else:
                        box_fa: DeterministicFiniteAutomaton = rsm.boxes[
                            Symbol(symbol)
                        ].dfa
                        start_state = box_fa.start_state.value
                        edges_data.variable_edges[symbol] = (
                            RsmState(Symbol(symbol), start_state),
                            RsmState(var, to_state),
                        )

        start_symbol = rsm.initial_label
        start_fa: DeterministicFiniteAutomaton = rsm.boxes[start_symbol].dfa
        self.start_state = RsmState(start_symbol, start_fa.start_state.value)

    def __init__(self, rsm: rsa.RecursiveAutomaton, graph: nx.DiGraph):
        self.nodes_edges: Dict[int, Dict[Symbol, Set[int]]] = {}
        self.rsm_data: Dict[Symbol, Dict[str, RsmStateData]] = {}
        self.start_state: RsmState

        self.rsm = rsm
        self.graph = graph

        self.initialize_graph_data(graph)
        self.initialize_rsm_data(rsm)

        self.gss_stack = GSStack()
        self.accepting_node = self.gss_stack.get_node(RsmState(Symbol("$"), "fin"), -1)

        self.unprocessed: Set[SPPFNode] = set()
        self.added: Set[SPPFNode] = set()

    def add_sppf_nodes(self, nodes: Set[SPPFNode]):
        nodes.difference_update(self.added)
        self.added.update(nodes)
        self.unprocessed.update(nodes)

    def filter_popped_nodes(
        self, nodes: Set[SPPFNode], prev_node: SPPFNode
    ) -> Tuple[Set[SPPFNode], Set[Tuple[int, int]]]:
        new_nodes = set()
        final_nodes = set()

        for node in nodes:
            if node.gss_node == self.accepting_node:
                start = prev_node.gss_node.node
                final = node.node
                final_nodes.add((start, final))
            else:
                new_nodes.add(node)

        return new_nodes, final_nodes

    def execute_step(self, sppf_node: SPPFNode) -> Set[Tuple[int, int]]:
        rsm_state = sppf_node.state
        rsm_data = self.rsm_data[rsm_state.variable][rsm_state.substate]

        def process_terminal():
            for term, new_rsm_state in rsm_data.terminal_edges.items():
                graph_terms = self.nodes_edges[sppf_node.node]
                if term in graph_terms:
                    new_nodes = {
                        SPPFNode(sppf_node.gss_node, new_rsm_state, node)
                        for node in graph_terms[term]
                    }
                    self.add_sppf_nodes(new_nodes)

        def process_variable() -> Set[Tuple[int, int]]:
            results = set()
            for var, (start_rsm, return_rsm) in rsm_data.variable_edges.items():
                inner_node = self.gss_stack.get_node(start_rsm, sppf_node.node)
                pop_nodes = inner_node.add_edge(return_rsm, sppf_node.gss_node)

                pop_nodes, start_end_pairs = self.filter_popped_nodes(
                    pop_nodes, sppf_node
                )
                self.add_sppf_nodes(pop_nodes)
                self.add_sppf_nodes({SPPFNode(inner_node, start_rsm, sppf_node.node)})
                results.update(start_end_pairs)
            return results

        def process_pop() -> Set[Tuple[int, int]]:
            popped_nodes = sppf_node.gss_node.pop(sppf_node.node)
            popped_nodes, result_pairs = self.filter_popped_nodes(
                popped_nodes, sppf_node
            )
            self.add_sppf_nodes(popped_nodes)
            return result_pairs

        process_terminal()
        result = process_variable()
        if rsm_data.is_final:
            result.update(process_pop())
        return result

    def solve_reachability(
        self, from_nodes: Set[int], to_nodes: Set[int]
    ) -> Set[Tuple[int, int]]:
        reachable_pairs = set()
        for node in from_nodes:
            gss_node = self.gss_stack.get_node(self.start_state, node)
            gss_node.add_edge(RsmState(Symbol("$"), "fin"), self.accepting_node)
            self.add_sppf_nodes({SPPFNode(gss_node, self.start_state, node)})

        while self.unprocessed:
            reachable_pairs.update(self.execute_step(self.unprocessed.pop()))

        return {(start, end) for start, end in reachable_pairs if end in to_nodes}


def gll_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
) -> Set[Tuple[int, int]]:
    if not start_nodes:
        start_nodes = set(graph.nodes())
    if not final_nodes:
        final_nodes = set(graph.nodes())

    solver = GllCFPQSolver(rsm, graph)
    return solver.solve_reachability(start_nodes, final_nodes)
