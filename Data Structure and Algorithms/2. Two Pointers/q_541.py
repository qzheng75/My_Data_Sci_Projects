# https://leetcode.com/problems/reverse-string-ii/


class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        def reverse(text):
            i, j = 0, len(text) - 1
            while i < j:
                text[i], text[j] = text[j], text[i]
                i += 1
                j -= 1
            return text
        res = list(s)
        for i in range(0, len(s), 2 * k):
            res[i : i + k] = reverse(res[i : i + k])
        return ''.join(res)
    

print(Solution().reverseStr('abcdefgh', 2))
