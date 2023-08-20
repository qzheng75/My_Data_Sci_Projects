# https://leetcode.com/problems/squares-of-a-sorted-array/
from typing import List


class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        n = len(nums)
        i, j, k = 0, n - 1, n - 1
        res = [-1] * n
        while i <= j:
            if nums[i] ** 2 > nums[j] ** 2:
                res[k] = nums[i] ** 2
                i += 1
            else:
                res[k] == nums[j] ** 2
                j -= 1
            k -= 1
        return res
            
