# https://leetcode.com/problems/number-of-enclaves/
from typing import List


class Solution:

    def __init__(self) -> None:
        self.positions = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    def dfs(self, grid, i, j, visited):
        n, m = len(grid), len(grid[0])
        for pos in self.positions:
            new_i = i + pos[0]
            new_j = j + pos[1]
            if 0 <= new_i < n and 0 <= new_j < m \
                and not visited[new_i][new_j] and grid[new_i][new_j] == 1:
                visited[new_i][new_j] = True
                self.dfs(grid, new_i, new_j, visited)


    def numEnclaves(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        visited = [[False for _ in range(m)] for _ in range(n)]
        count = 0

        for row in range(n):
            if grid[row][0] == 1:
                visited[row][0] = True
                self.dfs(grid, row, 0, visited)
            if grid[row][m - 1] == 1:
                visited[row][m - 1] = True
                self.dfs(grid, row, m - 1, visited)

        for col in range(1, m - 1):
            if grid[0][col] == 1:
                visited[0][col] = True
                self.dfs(grid, 0, col, visited)
            if grid[n - 1][col] == 1:
                visited[n - 1][col] = True
                self.dfs(grid, n - 1, col, visited)

        for i in range(n):
            for j in range(m):
                if not visited[i][j] and grid[i][j] == 1:
                    count += 1
        return count