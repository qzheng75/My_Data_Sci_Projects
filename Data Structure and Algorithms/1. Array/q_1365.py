# https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/
from typing import List


class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        res = sorted(nums)
        hash = dict()
        for i, num in enumerate(res):
            if num not in hash.keys():
                hash[num] = i       
        for i, num in enumerate(nums):
            res[i] = hash[num]
        return res