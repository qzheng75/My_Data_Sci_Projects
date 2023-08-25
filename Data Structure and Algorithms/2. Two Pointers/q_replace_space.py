# Given a string, replace all spaces in it with another given string


class Solution:
    def replaceSpace(self, s: str, to_replace: str) -> str:
        k = len(to_replace)
        cnt = s.count(' ')
        res = list(s)
        res.extend([' '] * cnt * (k - 1))
        i, j = len(s) - 1, len(res) - 1

        while i >= 0:
            if res[i] != ' ':
                res[j] = res[i]
                j -= 1
            else:
                res[j - k + 1 : j + 1] = to_replace
                j -= k
            i -= 1
        return ''.join(res)
    

print(f"'{Solution().replaceSpace('We are r', '2023!')}'")