# https://leetcode.com/problems/sort-array-by-parity-ii/
from typing import List


class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        res = [0] * len(nums)
        even_idx = 0
        odd_idx = 1
        for i in range(len(nums)):
            if nums[i] % 2 == 0:
                res[even_idx] = nums[i]
                even_idx += 2
            else:
                res[odd_idx] = nums[i]
                odd_idx += 2
        return res