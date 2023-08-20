# https://leetcode.com/problems/find-pivot-index/
from typing import List


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        sum_arr = sum(nums)
        left_sum = 0
        for i, num in enumerate(nums):
            if sum_arr - left_sum - num == left_sum:
                return i
            left_sum += num
        return -1