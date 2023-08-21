# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left_border = self.find_left(nums, target)
        right_border = self.find_right(nums, target)
        if left_border == -2 and right_border == -2:
            return [-1, -1]
        if right_border - left_border > 1:
            return [left_border + 1, right_border - 1]
        return [-1, -1]

    def find_left(self, nums, target):
        left, right = 0, len(nums) - 1
        left_border = -2
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                left_border = right
        return left_border
    
    def find_right(self, nums, target):
        left, right = 0, len(nums) - 1
        right_border = -2
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
                right_border = left
        return right_border


print(Solution().searchRange([5, 7, 7, 8, 8, 10], 8))