# https://leetcode.com/problems/redundant-connection/
from typing import List


class UnionFind:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.ranks = [0 for _ in range(n)]

    def find(self, i):
        return i if self.parents[i] == i else self.find(self.parents[i])
    
    def union(self, x, y):
        if self.ranks[x] > self.ranks[y]:
            self.parents[y] = x
        elif self.ranks[y] > self.ranks[x]:
            self.parents[x] = y
        else:
            self.parents[y] = x
            self.ranks[x] += 1


class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        uf_set = UnionFind()