# https://leetcode.com/problems/binary-tree-right-side-view/
from typing import List, Optional
from TreeNode import TreeNode
from collections import deque


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        q = deque()
        if not root:
            return res
        q.append(root)
        while len(q) != 0:
            level = []
            for _ in range(len(q)):
                cur = q.popleft()
                level.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            res.append(level)
        view = []
        for row in res:
            view.append(row[-1])
        return view