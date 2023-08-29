# https://leetcode.com/problems/minimum-depth-of-binary-tree/
from typing import List, Optional
from TreeNode import TreeNode
from collections import deque


class Solution:
    def mixDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        depth = 0
        q = deque([root])
        while len(q) != 0:
            depth += 1
            n = len(q)
            for _ in range(n):
                cur = q.popleft()
                if not cur.left and not cur.right:
                    return depth
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return depth