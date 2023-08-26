# https://leetcode.com/problems/swap-nodes-in-pairs/
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        next = head.next
        new_head = self.swapPairs(next.next)
        next.next = head
        head.next = new_head
        return next
        