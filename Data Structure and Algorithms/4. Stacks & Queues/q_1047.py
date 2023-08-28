# https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/


class Solution:
    def removeDuplicates(self, s: str) -> str:
        st = []
        for c in s:
            if len(st) == 0:
                st.append(c)
                continue
            if c == st[-1]:
                st.pop()
            else:
                st.append(c)
        return ''.join(st)