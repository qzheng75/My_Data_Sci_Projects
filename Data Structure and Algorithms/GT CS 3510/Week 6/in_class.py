from typing import List


class InClassAlgo:
    @staticmethod
    def arrange_tasks(weights: List[int], sizes: List[int]):
        assert len(weights) == len(sizes)
        ratios = [w / s for w, s in zip(weights, sizes)]
        return [index for index, _ in sorted(enumerate(ratios), key=lambda x: x[1])]
    

