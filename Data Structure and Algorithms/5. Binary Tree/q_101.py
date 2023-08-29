# https://leetcode.com/problems/symmetric-tree/
from typing import Optional
from TreeNode import TreeNode


class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        return self.compare(root.left, root.right)
        
    def compare(self, left, right):
        if not right and not left:
            return True
        elif not left or not right:
            return False
        return left.val == right.val and self.compare(left.left, right.right) and self.compare(left.right, right.left)