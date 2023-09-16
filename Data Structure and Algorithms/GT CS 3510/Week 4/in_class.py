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

    @staticmethod
    def is_cyclic(graph):
        n = len(graph)
        visited = [0] * n
        for i in range(n):
            if visited[i] == 0 and InClassAlgo._dfs_cyclic(graph, i, visited):
                return True
        return False

    @staticmethod
    def _dfs_cyclic(graph, cur_node, visited):
        visited[cur_node] = 1
        for nbr in graph[cur_node]:
            if visited[nbr] == 1:
                return True
            elif visited[nbr] == 0 and InClassAlgo._dfs_cyclic(graph, nbr, visited):
                return True
        visited[cur_node] = 2
        return False

    time = 0
    @staticmethod
    def topological_sort(graph):
        global time
        assert not InClassAlgo.is_cyclic(graph), 'Input graph can\'t be cyclic.'
        time = 0
        n = len(graph)
        first_visit, last_visit, visited = [0] * n, [0] * n, [False] * n
        InClassAlgo._dfs_topological_sort(graph, 0, first_visit, last_visit, visited)
        visit_dict = dict(zip(range(len(last_visit)), last_visit))
        return sorted(visit_dict, key=lambda x: visit_dict[x], reverse=True)

    @staticmethod
    def _dfs_topological_sort(graph, cur_node, first_visit, last_visit, visited):
        global time
        visited[cur_node] = True
        first_visit[cur_node] = time
        time += 1
        for nbr in graph[cur_node]:
            if not visited[nbr]:
                InClassAlgo._dfs_topological_sort(graph, nbr, first_visit, last_visit, visited)
        last_visit[cur_node] = time
        time += 1


if __name__ == '__main__':
    graph = [[1, 3], [2], [0], [1]]
    print(InClassAlgo.topological_sort(graph))