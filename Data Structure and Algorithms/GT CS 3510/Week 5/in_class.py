from typing import List, DefaultDict, Tuple
import heapq


class InClassAlgo:
    @staticmethod
    def dijkstra(graph: List[dict], start_idx: int) -> List[Tuple[int, float]]:
        """
        Calculate the shortest distances from a given starting node to all other nodes in a weighted graph
        using Dijkstra's algorithm.

        Args:
        graph (List[dict]): An adjacency list representation of the graph.
            Each element of the list is a dictionary where keys are neighboring node indices
            and values are corresponding edge weights.
        start_idx (int): The index of the starting node from which to calculate shortest distances.

        Returns:
        List[Tuple[int, float]]: A list of tuples, where each tuple contains the index of a node \
            and its corresponding shortest distance from the starting node.
        
        Note:
        - If a node is not reachable from the starting node, it will not appear in the result list.
        - The result includes the starting node itself with a distance of 0.
        - If there are multiple paths to a node with the same shortest distance, the path
          that was discovered first will be recorded in the result.

        Example:
        ```
        graph = [
            {1: 4, 2: 2, 3: 10},
            {3: 7, 4: 10},
            {3: 10},
            {4: 1},
            {}
        ]
        start_idx = 0

        result = InClassAlgo.dijkstra(graph, start_idx)
        # Output: dict_items([(0, 0), (1, 4), (2, 2), (3, 10), (4, 11)])
        ```
        """
        n = len(graph)
        visited = [False] * n
        dist_map = {}
        pq = []
        node_costs = DefaultDict(lambda: float('inf'))
        node_costs[start_idx] = 0
        heapq.heappush(pq, (0, start_idx))
    
        while pq:
            _, node = heapq.heappop(pq)
            visited[node] = True
    
            for nbr, weight in graph[node].items():
                if not visited[nbr]: 
                    new_cost = node_costs[node] + weight
                    if node_costs[nbr] > new_cost:
                        dist_map[nbr] = node
                        node_costs[nbr] = new_cost
                        heapq.heappush(pq, (new_cost, nbr))
        return node_costs.items()

if __name__ == '__main__':
    graph = [{1: 4, 2: 2, 3: 10}, {3: 7, 4: 10}, {3: 10}, {4: 1}, {}]
    print(InClassAlgo.dijkstra(graph, 0))
