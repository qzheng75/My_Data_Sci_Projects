from typing import List


class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        n = len(stones)
        _sum = sum(stones)
        target = _sum // 2
        dp = [0] * target
        for i in range(n):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return _sum - dp[target] - dp[target]


        