# https://leetcode.com/problems/symmetric-tree/
from typing import Optional
from TreeNode import TreeNode


class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.get_balance_height(root) != -1
    
    def get_balance_height(self, root):
        if not root:
            return 0
        left_height = self.get_balance_height(root.left)
        right_height = self.get_balance_height(root.right)
        height_diff = abs(left_height - right_height)
        if left_height == -1 or right_height == -1 or height_diff > 1:
            return -1
        return 1 + max(left_height, right_height)
        