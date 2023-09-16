# https://leetcode.com/problems/max-area-of-island/
from typing import List


class Solution:

    def __init__(self):
        self.count = 0

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0

        m, n = len(grid), len(grid[0])
        visited = [[False for _ in range(n)] for _ in range(m)]

        result = 0
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == 1:
                    self.count = 0
                    self.dfs(grid, visited, i, j)
                    result = max(result, self.count)
        return result

    def dfs(self, grid, visited, x, y):
        if visited[x][y] or grid[x][y] == 0:
            return 
        visited[x][y] = True
        self.count += 1
        for new_x, new_y in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]): 
                self.dfs(grid, visited, new_x, new_y)