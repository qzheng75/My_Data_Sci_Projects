# https://leetcode.com/problems/reorder-list/
from typing import Optional
from collections import deque


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        dq = deque()
        cur = head
        while cur.next:
            dq.append(cur.next)
            cur = cur.next
        cur = head
        while len(dq) > 0:
            cur.next = dq.pop()
            cur = cur.next
            if len(dq):
                cur.next = dq.popleft()
                cur = cur.next
        cur.next = None
