from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import List, Iterable, Any, Optional

import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State
from scipy.sparse import csr_matrix, kron

from project.task2 import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    matrix_type = csr_matrix
    _symbol_matrices: dict[Symbol, matrix_type]
    _total_states: int
    _initial_states: set[int]
    _accept_states: set[int]
    _state_index_mapping: dict[State, int]

    @staticmethod
    def _index_elements(elements) -> dict[Any, int]:
        return {el: i for i, el in enumerate(elements)}

    def _build_symbol_matrices(self, nfa) -> dict[Symbol, np.ndarray]:
        adj_matrix_per_symbol = defaultdict(
            lambda: np.zeros((len(nfa.states), len(nfa.states)), dtype=bool)
        )
        for start, transitions in nfa.to_dict().items():
            for symbol, destinations in transitions.items():
                start_idx = self._state_index_mapping[start]
                if type(destinations) is State:
                    destinations = {destinations}
                for end in destinations:
                    end_idx = self._state_index_mapping[end]
                    adj_matrix_per_symbol[symbol][start_idx, end_idx] = True
        return adj_matrix_per_symbol

    def __init__(self, nfa: Optional[NondeterministicFiniteAutomaton]):
        self._symbol_matrices = dict()
        if nfa is None:
            self._total_states = 0
            self._initial_states = set()
            self._accept_states = set()
            return

        self._state_index_mapping = self._index_elements(nfa.states)
        self._total_states = len(nfa.states)
        self._initial_states = set(
            self._state_index_mapping[i] for i in nfa.start_states
        )
        self._accept_states = set(
            self._state_index_mapping[i] for i in nfa.final_states
        )

        symbol_matrices = self._build_symbol_matrices(nfa)

        for symbol, matrix in symbol_matrices.items():
            self._symbol_matrices[symbol] = self.matrix_type(matrix, dtype=bool)

    def accepts(self, word: Iterable[Symbol]) -> bool:
        @dataclass
        class StateConfig:
            remaining_word: List[Symbol]
            current_state: int

        stack = [
            StateConfig(list(word), start_state) for start_state in self._initial_states
        ]

        while stack:
            config = stack.pop()

            if not config.remaining_word:
                if config.current_state in self._accept_states:
                    return True
                continue

            next_symbol = config.remaining_word[0]
            if next_symbol not in self._symbol_matrices:
                continue

            for next_state in range(self._total_states):
                if self._symbol_matrices[next_symbol][config.current_state, next_state]:
                    stack.append(StateConfig(config.remaining_word[1:], next_state))

        return False

    @staticmethod
    def _compute_closure(matrix: matrix_type):
        mat_dense = matrix.todense()
        num_nodes = mat_dense.shape[0]
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    mat_dense[i, j] = mat_dense[i, j] or (
                        mat_dense[i, k] and mat_dense[k, j]
                    )
        return mat_dense

    def transitive_closure(self) -> np.ndarray:
        combined_matrix = self.matrix_type(sum(self._symbol_matrices.values()))
        return self._compute_closure(combined_matrix)

    def is_empty(self) -> bool:
        if not self._symbol_matrices:
            return True
        closure_matrix = self.transitive_closure()
        return not any(
            closure_matrix[s, e]
            for s, e in product(self._initial_states, self._accept_states)
        )

    @property
    def adj_matrices(self):
        return self._symbol_matrices

    @property
    def states_number(self):
        return self._total_states

    @property
    def states_to_num(self):
        return self._state_index_mapping

    @property
    def start_states(self):
        return self._initial_states

    @property
    def final_states(self):
        return self._accept_states


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    new_instance = AdjacencyMatrixFA(None)
    common_symbols = [
        symbol
        for symbol in set(automaton1.adj_matrices).intersection(automaton2.adj_matrices)
    ]

    def compute_kronecker(matrix1, matrix2):
        return AdjacencyMatrixFA.matrix_type(kron(matrix1, matrix2))

    new_instance._symbol_matrices = {
        symbol: compute_kronecker(
            automaton1.adj_matrices[symbol], automaton2.adj_matrices[symbol]
        )
        for symbol in common_symbols
    }

    def compute_state_index(st1, st2):
        return st1 * automaton2.states_number + st2

    def combine_states(set1, set2):
        return set(compute_state_index(st1, st2) for st1, st2 in product(set1, set2))

    new_instance._state_index_mapping = {
        State((s1[0], s2[0])): compute_state_index(s1[1], s2[1])
        for s1, s2 in product(
            automaton1.states_to_num.items(), automaton2.states_to_num.items()
        )
    }

    new_instance._initial_states = combine_states(
        automaton1.start_states, automaton2.start_states
    )
    new_instance._accept_states = combine_states(
        automaton1.final_states, automaton2.final_states
    )
    new_instance._total_states = automaton1.states_number * automaton2.states_number
    return new_instance


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    adj_regex = AdjacencyMatrixFA(regex_dfa)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    adj_graph = AdjacencyMatrixFA(graph_nfa)
    adj_intersect = intersect_automata(adj_graph, adj_regex)

    adj_closure = adj_intersect.transitive_closure()

    def get_state(st1, st2):
        return adj_intersect.states_to_num[State((st1, st2))]

    if adj_intersect.is_empty():
        result = set()
    else:
        result = set(
            state_pair
            for dfa_states_pair in product(
                regex_dfa.start_states, regex_dfa.final_states
            )
            for state_pair in product(graph_nfa.start_states, graph_nfa.final_states)
            if adj_closure[
                get_state(state_pair[0], dfa_states_pair[0]),
                get_state(state_pair[1], dfa_states_pair[1]),
            ]
        )

    if adj_intersect.accepts([]):
        result = result.union(
            set(
                i
                for i in product(graph_nfa.start_states, graph_nfa.final_states)
                if i[0] == i[1]
            )
        )

    return result
