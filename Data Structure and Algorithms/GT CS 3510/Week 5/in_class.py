from typing import List, DefaultDict, Tuple
import heapq


class UnionFindSet:
    def __init__(self, n: int):
        """
        Initialize a UnionFindSet instance with `n` elements.

        Args:
            n (int): The number of elements in the set.
        """
        self.parents = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, i: int) -> int:
        """
        Find the representative (root) of the subset to which element `i` belongs.

        Args:
            i (int): The element for which to find the representative.

        Returns:
            int: The representative element (root) of the subset.
        """
        if self.parents[i] != i:
            self.parents[i] = self.find(self.parents[i])
        return self.parents[i]
    
    def union(self, x: int, y: int) -> None:
        """
        Unites the subsets containing elements `x` and `y` by rank.

        Args:
            x (int): An element to be united.
            y (int): Another element to be united.

        Returns:
            None
        """
        if self.rank[x] < self.rank[y]:
            self.parents[x] = y
        elif self.rank[x] > self.rank[y]:
            self.parents[y] = x
        else:
            self.parents[y] = x
            self.rank[x] += 1

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
    
    @staticmethod
    def prim_MST(graph: List[dict], start_node: int) -> List[dict]:
        """
        Computes the Minimum Spanning Tree (MST) of a weighted graph using Prim's algorithm.

        Parameters:
            graph (List[dict]): An adjacency list representation of the weighted graph.
            start_node (int): The starting node for the MST construction.

        Returns:
            List[dict]: An adjacency list representation of the MST.
        """
        n = len(graph)
        visited = [False] * n
        visited[start_node] = True
        mst = [{} for _ in range(n)]
        pq = []

        for to, dist in graph[start_node].items():
            heapq.heappush(pq, (dist, start_node, to))
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
                    heapq.heappush(pq, (weight, to_node, nbr))
            num_edge += 1

        for node_idx in range(len(mst)):
            for to, weight in mst[node_idx].items():
                mst[to].update({node_idx: weight})
        return mst
    
    def kruskal_MST(graph: List[dict]) -> List[dict]:
        """
        Computes the Minimum Spanning Tree (MST) of a weighted graph using Kruskal's algorithm.

        Parameters:
            graph (List[dict]): An adjacency list representation of the weighted graph.

        Returns:
            List[dict]: An adjacency list representation of the MST.
        """
        edges = []
        for node_idx in range(len(graph)):
            for to, weight in graph[node_idx].items():
                edges.append([node_idx, to, weight])
        
        edges = sorted(edges, key=lambda x : x[2])
        union_find = UnionFindSet(len(graph))

        mst_edges = []
        while len(mst_edges) < len(graph) - 1:
            min_edge = edges.pop(0)
            x = union_find.find(min_edge[0])
            y = union_find.find(min_edge[1])
            if x != y:
                mst_edges.append(min_edge)
                union_find.union(x, y)

        mst = [{} for _ in range(len(graph))]
        for edge in mst_edges:
            from_node, to_node, weight = edge
            mst[from_node].update({to_node: weight})
            mst[to_node].update({from_node: weight})
            
        return mst


if __name__ == '__main__':
    graph = [{1: 4, 7: 8}, {0: 4, 2: 8, 7: 11}, {1: 8, 3: 7, 5: 4, 8: 2},
             {2: 7, 4: 9, 5: 14}, {3: 9, 5: 10}, {2: 4, 3: 14, 4: 10},
             {5: 2, 7: 1, 8: 6}, {0: 8, 1: 11, 6: 1, 8: 7},
             {2: 2, 6: 6, 7: 7}]
    print(InClassAlgo.kruskal_MST(graph))
    print(InClassAlgo.prim_MST(graph, 0))
