from typing import List


class InClassAlgo:
    @staticmethod
    def find_reversed_pairs(arr: List[int]) -> List[List[int]]:
        """
        Find pairs of reversed elements in a list.

        Given a list of integers, this method identifies and returns pairs of elements
        [arr[i], arr[j]] such that i < j and arr[i] > arr[j].

        Parameters:
        - arr (List[int]): The input list of integers.

        Returns:
        - List[List[int]]: A list of lists, where each inner list represents a reversed pair.
          Each inner list contains two integers, [x, y], where x > y.
        """
        res = set()
        InClassAlgo._merge_sort(res, arr)
        return list(res)

    @staticmethod
    def _merge_sort(pairs: set(List[int]), arr: List[int]) -> List[int]:
        """
        Perform a merge sort on a list and find reversed pairs.

        This is a helper method for the `find_reversed_pairs` method. It recursively
        performs a merge sort on the input list while identifying reversed pairs.

        Parameters:
        - pairs (set(List[int])): A set to store reversed pairs.
        - arr (List[int]): The input list of integers.

        Returns:
        - List[int]: A sorted list of integers after merge sort.

        Note:
        - This method is not intended for direct use outside of the main algorithm.
        """
        n = len(arr)
        if n == 1:
            return arr
        mid_idx = n // 2
        left_arr = arr[0 : mid_idx]
        right_arr = arr[mid_idx : n]
        left_arr = InClassAlgo._merge_sort(pairs, left_arr)
        right_arr = InClassAlgo._merge_sort(pairs, right_arr)
        res_arr = [0] * n
        i, j, idx = 0, 0, 0
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] > right_arr[j]:
                for k in range(i, mid_idx):
                    pairs.append([left_arr[k], right_arr[j]])
                res_arr[idx] = right_arr[j]
                j += 1
            else:
                res_arr[idx] = left_arr[i]
                i += 1
            idx += 1
        while i < len(left_arr):
            res_arr[idx] = left_arr[i]
            i += 1
            idx += 1
        while j < len(right_arr):
            res_arr[idx] = right_arr[j]
            j += 1
            idx += 1
        return res_arr
