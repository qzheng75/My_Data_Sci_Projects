from typing import List
import numpy as np
import math
import random


class InClassAlgorithms:
    @staticmethod
    def int_to_binary_list(num: int, pad=32):
        binary_string = bin(num)[2:]
        padding_length = pad - len(binary_string)
        padded_binary_string = '0' * padding_length + binary_string
        return np.array([int(bit) for bit in padded_binary_string])
    
    @staticmethod
    def multiply_binary_numbers(x: np.ndarray, y: np.ndarray) -> int:
        """
        Multiply two n-bit binary-represented numbers.

        Given two n-bit binary numbers A and B, represented as arrays of 0s and 1s,
        this function calculates their product A * B using a divide-and-conquer approach.

        Parameters:
        - x (np.ndarray): The first n-bit binary number represented as a NumPy array.
        - y (np.ndarray): The second n-bit binary number represented as a NumPy array.

        Returns:
        - int: The result of A * B as an integer.

        Constraints:
        - The lengths of both input arrays x and y must be equal.
        - The length n of the binary numbers must be even.

        Time complexity:
        - Worst casw: O(n^{log3})
        """
        assert all(i in {0, 1} for i in x), 'x is not a binary-represented number.'
        assert all(i in {0, 1} for i in y), 'y is not a binary-represented number.'
        assert len(x) == len(y), f'Two arrays must be of same size, but got {len(x)} and {len(y)}'
        n = len(x)
        assert n % 2 == 0, f'Representation of numbers has to have even number of bits.'
        if n == 0:
            return 0
        if n == 1:
            return x[0] * y[0]
        mid_idx = n // 2
        x_left, x_right = x[0 : mid_idx], x[mid_idx :]
        y_left, y_right = y[0 : mid_idx], y[mid_idx :]
        term1 = (pow(2, n) - pow(2, n // 2)) * InClassAlgorithms.multiply_binary_numbers(x_left, y_left)
        term2 = pow(2, n //2) * InClassAlgorithms.multiply_binary_numbers(x_left + x_right, y_left + y_right)
        term3 = (1 - pow(2, n // 2)) * InClassAlgorithms.multiply_binary_numbers(x_right, y_right)
        return term1 + term2 + term3
    

    @staticmethod
    def k_select(arr: List[int], k: int, M=5) -> int:
        """
        Select the kth smallest element from an unsorted list using the k-select algorithm.

        This method finds the kth smallest element in the given list 'arr'
        using the k-select algorithm, which leverages median-of-medians for pivot selection.

        Parameters:
        - arr (List[int]): The unsorted list of integers.
        - k (int): The index (0-based) of the desired smallest element to find.
        - M (int, optional): The size of each group in finding good pivot. Default to be 5.
        Setting M < 5 can cause inefficiency.

        Returns:
        - int: The kth smallest element from the list 'arr'.

        Constraints:
        - The k value must be a non-negative integer, and it should be in range [0, n - 1]

        Time complexity:
        - Worst case: O(n)
        """
        n = len(arr)
        assert k < n and k >= 0, f'{k} is out of bounds for array of length {n}.'
        if n <= M * M:
            return InClassAlgorithms._brute_force_k_select(arr, k)
        num_groups = math.ceil(n / M)
        chunks = [arr[i : i + M] for i in range(0, n, M)]
        chunk_medians = [InClassAlgorithms._brute_force_k_select(subarr, M // 2)
                         for subarr in chunks]
        median = InClassAlgorithms.k_select(chunk_medians, num_groups // 2)
        pivot_idx = InClassAlgorithms._partition(arr, median)
        if k == pivot_idx:
            return median
        elif k < pivot_idx:
            return InClassAlgorithms.k_select(arr[0 : pivot_idx], k, M)
        else:
            return InClassAlgorithms.k_select(arr[pivot_idx :], k - pivot_idx, M)

    @staticmethod
    def _partition(arr, pivot):
        """
        Helper method to partition an array around a pivot value.

        This method rearranges the elements of the input array `arr` in such a way that
        all elements less than `pivot` are moved to the left, and all elements greater
        than `pivot` are moved to the right. The pivot elements equal to `pivot` remain
        between the two partitions.

        Parameters:
        - arr (List): The input list of elements to be partitioned.
        - pivot: The value around which the partitioning is performed.

        Returns:
        - int: The index at which the left partition ends. All elements at indices
        less than this value are less than or equal to `pivot`, and elements at
        indices greater than or equal to this value are greater than `pivot`.

        Time complexity:
        - Worst case: O(n)
        """
        left = 0
        right = len(arr) - 1
        i = 0
        while i <= right:
            if arr[i] == pivot:
                i += 1
            elif arr[i] < pivot:
                arr[left], arr[i] = arr[i], arr[left]
                left += 1
                i += 1
            else:
                arr[right], arr[i] = arr[i], arr[right]
                right -= 1
        return left

    @staticmethod
    def _brute_force_k_select(arr: List[int], k: int) -> int:
        """
        Find the kth smallest element in a list of integers using a brute-force method.

        This method sorts the input list in ascending order and returns the kth smallest element.
        
        Parameters:
        - arr (List[int]): A list of integers.
        - k (int): The position of the desired smallest element (0-indexed).

        Returns:
        - int: The kth smallest element from the sorted list.

        Time complexity:
        - Worst case: O(nlogn). Yet in median of medians algorithm, we consider it
        to be O(1) since each sort is performed on constant-length arrays.
        """
        sorted_arr = sorted(arr)
        return sorted_arr[k]

    

if __name__ == '__main__':
    # x = InClassAlgorithms.int_to_binary_list(11)
    # y = InClassAlgorithms.int_to_binary_list(259)
    # print(InClassAlgorithms.multiply_binary_numbers(x, y))
    test_arr = list(range(1, 31))
    random.shuffle(test_arr)
    print(test_arr)
    for i in range(30):
        print(InClassAlgorithms.k_select(test_arr, i))