# https://leetcode.com/problems/remove-nth-node-from-end-of-list/
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def len_ll(self, head):
        cur = head
        cnt = 0
        while cur is not None:
            cnt += 1
            cur = cur.next
        return cnt
    
    def moveForward(self, head, steps) -> ListNode:
        while steps > 0:
            head = head.next
            steps -= 1
        return head

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        len_a, len_b = self.len_ll(headA), self.len_ll(headB)
        dis = len_a - len_b

        if dis > 0:
            headA = self.moveForward(headA, dis)
        else:
            headB = self.moveForward(headB, abs(dis))

        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        return None
        