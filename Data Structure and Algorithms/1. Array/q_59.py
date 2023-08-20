# https://leetcode.com/problems/spiral-matrix-ii/
from typing import List


class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        start_x, start_y = 0, 0
        offset = 1
        count = 1
        loop_num = n // 2
        mid = n // 2
        mat = [[0] * n for _ in range(n)]

        while loop_num >= 0:
            i, j = start_x, start_y

            # Top: left -> right
            for j in range(start_y, n - offset):
                mat[start_x][j] = count
                count += 1
            # Right: top -> down
            for i in range(start_x, n - offset):
                mat[i][n - offset] = count
                count += 1
            # Bottom: right -> left
            for j in range(n - offset, start_y, -1):
                mat[n - offset][j] = count
                count += 1
            # Left: down -> top
            for j in range(n - offset, start_x, -1):
                mat[j][start_y] = count
                count += 1

            start_x, start_y = start_x + 1, start_y + 1
            offset += 1
            loop_num -= 1

        if n % 2 != 0:
            mat[mid][mid] = count 
        return mat
