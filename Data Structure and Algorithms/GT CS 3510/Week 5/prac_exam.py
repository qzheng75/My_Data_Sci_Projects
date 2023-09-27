from helpers import Helpers
from typing import List, Optional
from collections import deque
from in_class import InClassAlgo
import heapq


class PracExamAlgo:
    @staticmethod
    def dual_source(graph: List[List[int]]) -> bool:
        in_degree = {}
        for node in range(len(graph)):
            in_degree[node] = 0
        for node in range(len(graph)):
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        count_dual_sources = 0
        for node in range(len(graph)):
            if in_degree[node] == 0:
                count_dual_sources += 1
                if count_dual_sources == 2:
                    return True
        return False

    @staticmethod
    def global_destination(graph: List[List[int]]) -> Optional[int]:
        topological_order = Helpers.topological_sort(graph)
        start_node = topological_order[-1]
        t_graph = Helpers._transpose_graph(graph)
        def dfs(graph, cur_node, visited):
            visited[cur_node] = True
            for nbr in graph[cur_node]:
                if not visited[nbr]:
                    dfs(graph, nbr, visited)
        visited = [False] * len(t_graph)
        dfs(t_graph, start_node, visited)
        return start_node if all(visited) == True else None
    
    @staticmethod
    def min_edge_flip(graph: List[List[int]], source, destination) -> int:
        def build_undirected(graph):
            undirected_g = [{} for _ in range(len(graph))]
            for i in range(len(graph)):
                for j in graph[i]:
                    undirected_g[i].update({j: 0})
                    undirected_g[j].update({i: 1})
            return undirected_g
        meta_graph = build_undirected(graph)
        distances = list(InClassAlgo.dijkstra(meta_graph, source))
        return next((item for item in distances if item[0] == destination), None)[1]
    
    @staticmethod
    def max_speed_to_drive(graph, start, destination):
        def max_spanning_tree(graph, start_node):
            n = len(graph)
            visited = [False] * n
            visited[start_node] = True
            mst = [{} for _ in range(n)]
            pq = []

            for to, dist in graph[start_node].items():
                heapq.heappush(pq, (-dist, start_node, to))
            num_edge = 0
        
            while num_edge < n - 1:
                least_weight_edge = heapq.heappop(pq)

                from_node = least_weight_edge[1]
                to_node = least_weight_edge[2]
                edge_weight = least_weight_edge[0]

                if visited[to_node]:
                    continue

                visited[to_node] = True
                mst[from_node].update({to_node: edge_weight})

                for nbr, weight in graph[to_node].items():
                    if not visited[nbr]:
                        heapq.heappush(pq, (-weight, to_node, nbr))
                num_edge += 1

            mst = [{key: -value for key, value in d.items()} for d in mst]

            for node_idx in range(len(mst)):
                for to, weight in mst[node_idx].items():
                    mst[to].update({node_idx: weight})
            return mst
        
        def dfs_find_min_weight_edge(graph, source, destination, visited=None, min_edge=None):
            if visited is None:
                visited = set()
            if min_edge is None:
                min_edge = float('inf')

            visited.add(source)

            if source == destination:
                return min_edge

            for neighbor, weight in graph[source].items():
                if neighbor not in visited:
                    min_edge = min(min_edge, weight)
                    result = dfs_find_min_weight_edge(graph, neighbor, destination, visited, min_edge)
                    if result < min_edge:
                        min_edge = result
            return min_edge
        
        mst = max_spanning_tree(graph, 0)
        return dfs_find_min_weight_edge(mst, start, destination)
    
    

if __name__ == '__main__':
    graph = [[1], [], [1, 3], [], [5], [1], [4, 3]]
    #print(PracExamAlgo.min_edge_flip(graph, 0, 6))

    graph = [{1: 3, 3: 7, 4: 8},
             {0: 3, 2: 1, 3: 4},
             {1: 1, 3: 2},
             {0: 7, 1: 4, 2: 2, 4: 3},
             {0: 8, 3: 3}]
    print(PracExamAlgo.max_speed_to_drive(graph, 0, 2))




