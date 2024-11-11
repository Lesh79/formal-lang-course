from itertools import product
import networkx as nx
import scipy.sparse as sp
from pyformlang import rsa, cfg as pycfg
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from scipy.sparse import csc_array

from project.AdjacencyMatrixFA import AdjacencyMatrixFA, intersect_automata
from project.task2 import graph_to_nfa
from typing import Set, Tuple


def cfg_to_rsm(cfg: pycfg.CFG) -> rsa.RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> rsa.RecursiveAutomaton:
    return rsa.RecursiveAutomaton.from_text(ebnf)


def bool_decomposed_rsm(rsm: rsa.RecursiveAutomaton) -> AdjacencyMatrixFA:
    nfa = NondeterministicFiniteAutomaton()

    for nonterminal, box in rsm.boxes.items():
        dfa_box = box.dfa

        for state in dfa_box.final_states | dfa_box.start_states:
            nfa.add_start_state(
                State((nonterminal, state))
            ) if state in dfa_box.start_states else None
            nfa.add_final_state(
                State((nonterminal, state))
            ) if state in dfa_box.final_states else None

        for u, v, label in dfa_box.to_networkx().edges(data="label"):
            nfa.add_transition(State((nonterminal, u)), label, State((nonterminal, v)))

    return AdjacencyMatrixFA(nfa)


def __compute_closure(decomposed_rsa, decomposed_graph, rsm):
    prev_nonzero_count, curr_nonzero_count = 0, None

    while prev_nonzero_count != curr_nonzero_count:
        prev_nonzero_count = curr_nonzero_count
        intersected = intersect_automata(decomposed_rsa, decomposed_graph)
        trans_closure = intersected.transitive_closure()

        for row_idx, col_idx in zip(*trans_closure.nonzero()):
            row_state = intersected.num_to_state[row_idx]
            col_state = intersected.num_to_state[col_idx]

            row_inner, row_graph_state = row_state.value
            row_label, row_rsm_state = row_inner.value
            col_inner, col_graph_state = col_state.value
            col_label, col_rsm_state = col_inner.value

            if (
                row_label == col_label
                and row_rsm_state in rsm.boxes[row_label].dfa.start_states
                and col_rsm_state in rsm.boxes[row_label].dfa.final_states
            ):
                row_graph_idx = decomposed_graph.states_to_num[row_graph_state]
                col_graph_idx = decomposed_graph.states_to_num[col_graph_state]
                decomposed_graph.adj_matrices[row_label][
                    row_graph_idx, col_graph_idx
                ] = True

        curr_nonzero_count = sum(
            mat.count_nonzero() for mat in decomposed_graph.adj_matrices.values()
        )


def tensor_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
    matrix_type=sp.csr_matrix,
) -> Set[Tuple[int, int]]:
    bool_rsm = bool_decomposed_rsm(rsm)
    bool_graph = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes), matrix_type=matrix_type
    )

    for nonterminal in rsm.boxes:
        for mat in (bool_graph, bool_rsm):
            if nonterminal not in mat.adj_matrices:
                mat.adj_matrices[nonterminal] = csc_array(
                    (mat.states_number, mat.states_number), dtype=bool
                )

    __compute_closure(bool_rsm, bool_graph, rsm)

    return {
        (bool_graph.num_to_state[start], bool_graph.num_to_state[finish])
        for start, finish in product(bool_graph.start_states, bool_graph.final_states)
        if bool_graph.adj_matrices[rsm.initial_label][start, finish]
    }
