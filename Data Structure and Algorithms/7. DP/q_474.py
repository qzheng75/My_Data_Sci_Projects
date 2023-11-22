from typing import List


class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            zero_num = s.count('0')
            one_num = len(s) - zero_num
            for i in range(m, zero_num - 1, -1):
                for j in range(n, one_num - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zero_num][j - one_num] + 1)
        return dp[m][n]