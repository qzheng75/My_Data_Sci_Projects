# https://leetcode.com/problems/evaluate-reverse-polish-notation/
from operator import add, sub, mul
from typing import List


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        operators = {'+': add, '-': sub, '*': mul, '/': lambda x, y: x // y}
        st = []
        for token in tokens:
            if token not in operators.keys():
                st.append(int(token))
            else:
                operand2 = st.pop()
                operand1 = st.pop()
                st.append(operators[token](operand1, operand2))
        return st.pop()
