# https://leetcode.com/problems/top-k-frequent-elements/
import heapq
from typing import List


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        map_ = {}
        for num in nums:
            map_[num] = map_.get(num, 0) + 1

        pq = []
        for key, freq in map_:
            heapq.heappush(pq, (key, freq))
            if len(pq) > k:
                heapq.heappop(pq)

        res = [0] * k
        for i in range(k-1, -1, -1):
            res[i] = heapq.heappop(pq)[1]
        return res
        
