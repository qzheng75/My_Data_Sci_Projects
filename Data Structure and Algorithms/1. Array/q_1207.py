# https://leetcode.com/problems/unique-number-of-occurrences/
from typing import List


class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        hash = [0] * 2002
        for num in arr:
            hash[num + 1000] += 1
        freq = [False] * 1002
        for num in hash:
            if num > 0:
                if not freq[num]:
                    freq[num] = True
                else:
                    return False
        return True