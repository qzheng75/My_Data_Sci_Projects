from typing import List


class Solution:
    def knapsack2D(self, budget: int, cost: List[int], values: List[int]) -> int:
        n = len(cost)
        m = budget
        dp = [[0] * (m + 1) for _ in range(n)]
        for j in range(m + 1):
            dp[0][j] = values[0] if cost[0] <= j else 0
        
        for i in range(1, n):
            for j in range(m + 1):
                if j < cost[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cost[i]] + values[i])
        return dp[n - 1][m]
    
    def knapsack1D(self, budget: int, cost: List[int], values: List[int]) -> int:
        n = len(cost)
        m = budget
        dp = [0] * (m + 1)
        for i in range(n):
            for j in range(budget, cost[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - cost[i]] + values[i])
        return dp[m]
    

if __name__ == '__main__':
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bagweight = 4
    print(Solution().knapsack2D(bagweight, weight, value))
    print(Solution().knapsack1D(bagweight, weight, value))
