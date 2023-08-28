# https://leetcode.com/problems/valid-parentheses/


class Solution:
    def isValid(self, s: str) -> bool:
        st = []
        for c in s:
            if c == '(' or c == '[' or c == '{':
                st.append(c)
            if c == ')' or c == ']' or c == '}':
                if len(st) == 0:
                    return False
                if c == ')' and st[-1] != '(':
                    return False
                elif c == '}' and st[-1] != '{':
                    return False
                elif c == ']' and st[-1] != '[':
                    return False
                st.pop()
            else:
                return False
        return len(st) == 0