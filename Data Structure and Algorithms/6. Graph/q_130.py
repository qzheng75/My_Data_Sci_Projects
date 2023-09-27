# https://leetcode.com/problems/number-of-enclaves/
from typing import List


class Solution:
    dir_list = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    def solve(self, board: List[List[str]]) -> None:
        row_size = len(board)
        column_size = len(board[0])
        visited = [[False] * column_size for _ in range(row_size)]
        for i in range(column_size):
            if board[0][i] == "O" and not visited[0][i]:
                self.dfs(board, 0, i, visited)
            if board[row_size - 1][i] == "O" and not visited[row_size - 1][i]:
                self.dfs(board, row_size-1, i, visited)

        for i in range(1, row_size - 1):
            if board[i][0] == "O" and not visited[i][0]:
                self.dfs(board, i, 0, visited)
            if board[i][column_size - 1] == "O" and not visited[i][column_size - 1]:
                self.dfs(board, i, column_size-1, visited)
        
        for i in range(row_size):
            for j in range(column_size):
                if board[i][j] == "A":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"

    
    def dfs(self, board, x, y, visited):
        if visited[x][y] or board[x][y] == "X":
            return
        visited[x][y] = True
        board[x][y] = "A"
        for i in range(4):
            new_x = x + self.dir_list[i][0]
            new_y = y + self.dir_list[i][1]
            if new_x >= len(board) or new_y >= len(board[0]) or new_x < 0 or new_y < 0:
                continue
            self.dfs(board, new_x, new_y, visited)