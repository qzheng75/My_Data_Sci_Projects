from typing import List


class InClassAlgo:

    @staticmethod
    def graph_is_connected(graph: List[List[int]]) -> bool:
        """
        Checks whether a given graph is connected.

        Parameters:
            - graph (List[List[int]]): An adjacency list representing the graph.

        Returns:
            - bool: True if the graph is connected, False otherwise.
        """
        visited = [False] * len(graph)
        InClassAlgo._dfs_connected(graph, 0, visited)
        return all(visited)

    @staticmethod
    def _dfs_connected(graph: List[List[int]], cur_node: int, visited: List[bool]) -> None:
        """
        A private helper method for depth-first search (DFS) to determine graph connectivity.

        Parameters:
            - graph (List[List[int]]): An adjacency list representing the graph.
            - cur_node (int): The current node being visited.
            - visited (List[bool]): A list of boolean values to track visited nodes.

        Returns:
            - None: This method does not return a value directly but updates the 'visited' list.
        """
        visited[cur_node] = True
        for nbr in graph[cur_node]:
            if not visited[nbr]:
                InClassAlgo._dfs_connected(graph, nbr, visited)


if __name__ == '__main__':
    graph = [[1, 2], [2, 3], [0, 1], [1]]
    print(InClassAlgo.graph_is_connected(graph))