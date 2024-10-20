from collections import defaultdict
from functools import reduce
from itertools import product

from networkx import MultiDiGraph
from scipy.sparse import (
    spmatrix,
    block_diag,
    identity,
    block_array,
    csr_matrix,
)

from project.fa_module import regex_to_dfa, graph_to_nfa
from project.AdjacencyMatrixFA import AdjacencyMatrixFA


def init_front(adj_dfa: AdjacencyMatrixFA, adj_nfa: AdjacencyMatrixFA) -> spmatrix:
    change = adj_dfa.states_number
    start_states = list(adj_nfa.start_states)
    left_vector = identity(change, dtype=bool)

    vectors = []
    for nfa_state_num in range(len(start_states)):
        right_vector = adj_nfa.matrix_type((change, adj_nfa.states_number), dtype=bool)
        for i in adj_dfa.start_states:
            right_vector[i, start_states[nfa_state_num]] = True
        vectors.append(block_array([[left_vector, right_vector]]))

    return reduce(lambda a, b: block_array([[a], [b]]), vectors).tocsr()


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_type: type(spmatrix) = csr_matrix,
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    adj_dfa = AdjacencyMatrixFA(regex_dfa, matrix_type=matrix_type)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    adj_nfa = AdjacencyMatrixFA(graph_nfa, matrix_type=matrix_type)

    change = adj_dfa.states_number
    start_states = list(adj_nfa.start_states)
    intersected_symbols = set(adj_dfa.adj_matrices.keys()).intersection(
        adj_nfa.adj_matrices.keys()
    )

    combined_matrices = {
        symbol: block_diag((adj_dfa.adj_matrices[symbol], adj_nfa.adj_matrices[symbol]))
        for symbol in intersected_symbols
    }

    init_front_matrix = init_front(adj_dfa, adj_nfa)
    front_left = init_front_matrix[:, :change]
    front_right = matrix_type(init_front_matrix[:, change:], dtype=bool)
    visited = matrix_type(front_right, dtype=bool)

    def update_front(front_right_to_update: spmatrix) -> spmatrix:
        def front_mul_matrix(cur_front, matrix) -> spmatrix:
            mul = matrix_type(cur_front @ matrix)
            diag_front_right = matrix_type(front_right_to_update.shape, dtype=bool)
            for i, j in zip(*mul[:, :change].nonzero()):
                diag_front_right[i // change * change + j, :] += mul[i, change:]
            return diag_front_right

        front = block_array([[front_left, front_right]])
        updated_front = reduce(
            lambda vector, matrix: vector + front_mul_matrix(front, matrix),
            combined_matrices.values(),
            matrix_type(front_right.shape, dtype=bool),
        )

        return updated_front

    while front_right.count_nonzero():
        front_right = update_front(front_right)
        front_right = front_right > visited
        visited += front_right

    start_to_reachable = defaultdict(set)
    for start_state, final_state in product(
        range(len(start_states)), adj_dfa.final_states
    ):
        reachable_states = set(
            visited[start_state * change + final_state, :].nonzero()[1]
        )
        start_to_reachable[start_states[start_state]].update(reachable_states)

    retrieved_states = {
        (start, final)
        for start, final in product(graph_nfa.start_states, graph_nfa.final_states)
        if adj_nfa.states_to_num[final]
        in start_to_reachable[adj_nfa.states_to_num[start]]
    }
    return retrieved_states
