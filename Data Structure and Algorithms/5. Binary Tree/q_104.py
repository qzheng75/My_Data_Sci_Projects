# https://leetcode.com/problems/maximum-depth-of-binary-tree/
from typing import List, Optional
from TreeNode import TreeNode


class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))