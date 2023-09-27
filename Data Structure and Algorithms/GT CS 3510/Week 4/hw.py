from typing import List
from in_class import InClassAlgo

count = 0

class HWAlgo:

    @staticmethod
    def num_nodes_not_in_cycle(graph: List[List[int]]):
        scc = InClassAlgo.strongly_connected_components(graph)
        count = 0
        for _, components in scc.items():
            if len(components) == 1:
                count += 1
        return count
    
    @staticmethod
    def is_bipartite(graph: List[List[int]]) -> bool:
        visited = [0] * len(graph)
        for i in range(len(graph)):
            if visited[i] == 0 and not HWAlgo._dfs_bipartite(graph, i, 1, visited):
                return False
        return True
    
    @staticmethod
    def _dfs_bipartite(graph, cur_node, color, visited) -> bool:
        if visited[cur_node] != 0:
            return visited[cur_node] == color
        visited[cur_node] = color
        for nbr in graph[cur_node]:
            assign_color = 1 if color == 2 else 2
            if not HWAlgo._dfs_bipartite(graph, nbr, assign_color, visited):
                return False
        return True
    
    @staticmethod
    def half_connected(graph: List[List[int]]) -> bool:
        if InClassAlgo.graph_is_connected(graph):
            raise ValueError("Currently support only DAGs.")
        order = InClassAlgo.topological_sort(graph)
        for i in range(0, len(order) - 1):
            if i + 1 not in graph[i]:
                return False
        return True

if __name__ == '__main__':
    graph = [[1], [2], [4], [2], [5], [1]]
    print(HWAlgo.num_nodes_not_in_cycle(graph))
