# https://leetcode.com/problems/all-paths-from-source-to-target/
from typing import List


class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        n = len(graph)
        visited = [False] * n
        visited[0] = True
        paths = []
        self.dfs(graph, 0, n - 1, visited, [0], paths)
        return paths

    def dfs(self, graph, cur_node, target, visited, path, paths):
        if cur_node == target:
            paths.append(path)
            return
        for nbr in graph[cur_node]:
            if not visited[nbr]:
                visited[nbr] = True
                path.append(nbr)
                self.dfs(graph, nbr, target, visited, list(path), paths)
                path.pop()
                visited[nbr] = False

if __name__ == '__main__':
    graph = [[4,3,1],[3,2,4],[3],[4],[]]
    paths = Solution().allPathsSourceTarget(graph)
    print(paths)