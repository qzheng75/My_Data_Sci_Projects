# https://leetcode.com/problems/sliding-window-maximum/
from typing import List
from collections import deque


class MaxQueue:
    def __init__(self):
        self.q = deque()
    
    def push(self, val):
        while len(self.q) != 0 and self.q[-1] < val:
            self.q.pop()
        self.q.append(val)

    def pop(self, element):
        if len(self.q) != 0 and self.q[0] == element:
            self.q.popleft()
    
    def front(self):
        return self.q[0]


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = MaxQueue()
        res = []
        for i in range(k):
            q.push(nums[i])
        res.append(q.front())
        for i in range(k, len(nums)):
            q.pop(nums[i - k])
            q.push(nums[i])
            res.append(q.front())
        return res