# https://leetcode.com/problems/4sum/
from typing import List


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        res = []
        nums.sort()
        n = len(nums)

        for i in range(n):
            if i != 0 and nums[i - 1] == nums[i]:
                continue
            if nums[i] > target and nums[i] > 0 and target > 0:
                break
            for j in range(i + 1, n):
                if j > i + 1 and nums[j - 1] == nums[j]:
                    continue
                if nums[i] + nums[j] > target and target > 0:
                    break
                left, right = j + 1, n - 1
                while left < right:
                    sum_ = nums[i] + nums[j] + nums[left] + nums[right]
                    if sum_ < target:
                        left += 1
                    elif sum_ > target:
                        right -= 1
                    else:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
        return res


print(Solution().fourSum([2,2,2,2,2], 8))