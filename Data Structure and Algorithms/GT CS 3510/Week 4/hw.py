from typing import List

count = 0

class HWAlgo:

    @staticmethod
    def num_nodes_not_in_cycle(graph: List[List[int]]):
        global count
        visited = [0] * len(graph)
        for nd in range(len(graph)):
            if visited[nd] == 0:
                HWAlgo._dfs_cycle(graph, nd, visited)
        return len(graph) - count

    @staticmethod
    def _dfs_cycle(graph: List[List[int]], cur_node: int, visited: List[int]) -> bool:
        global count
        visited[cur_node] = 1
        for nbr in graph[cur_node]:
            if visited[nbr] == 1:
                count += 1
                return True
            if visited[nbr] == 0:
                child_cycle = HWAlgo._dfs_cycle(graph, nbr, visited)
                if child_cycle:
                    count += 1
                return True
        visited[cur_node] = 2
        return False
    
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


if __name__ == '__main__':
    graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
    print(HWAlgo.is_bipartite(graph))
