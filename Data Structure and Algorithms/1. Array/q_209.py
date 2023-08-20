# https://leetcode.com/problems/minimum-size-subarray-sum/
from typing import List


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        cur_sum = 0
        min_len = 1e5 + 1
        i, j = 0, 0
        n = len(nums)
        while j < n:
            cur_sum += nums[j]
            while cur_sum > target:
                min_len = min(min_len, j - i)
                cur_sum -= nums[i]
                i += 1
            j += 1
        return min_len if min_len != 1e5 + 1 else 0