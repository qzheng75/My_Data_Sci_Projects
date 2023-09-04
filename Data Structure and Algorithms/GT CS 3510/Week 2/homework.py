from typing import List
import numpy as np


class Homework02:
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
        - Worst casw: O(n^{log_3(6)})
        """
        assert all(i in {0, 1} for i in x), 'x is not a binary-represented number.'
        assert all(i in {0, 1} for i in y), 'y is not a binary-represented number.'
        assert len(x) == len(y), f'Two arrays must be of same size, but got {len(x)} and {len(y)}'
        return Homework02._multiply_binary_numbers_helper(x, y)

    @staticmethod
    def _multiply_binary_numbers_helper(x: np.ndarray, y: np.ndarray) -> int:
        """
        Helper method for multipling two n-bit binary-represented numbers.

        Parameters:
        - x (np.ndarray): The first n-bit binary number represented as a NumPy array.
        - y (np.ndarray): The second n-bit binary number represented as a NumPy array.

        Returns:
        - int: The result of A * B as an integer.
        """
        n = len(x)
        if n == 0:
            return 0
        if n == 1:
            return x[0] * y[0]
        one_third_idx = n // 3
        two_third_idx = n * 2 // 3
        x_left, x_mid, x_right = x[0 : one_third_idx], x[one_third_idx : two_third_idx], x[two_third_idx :]
        y_left, y_mid, y_right = y[0 : one_third_idx], y[one_third_idx : two_third_idx], y[two_third_idx :]
        term1 = (pow(2, n * 4 // 3) - pow(2, n * 2 // 3) - pow(2, n)) \
            * Homework02._multiply_binary_numbers_helper(x_left, y_left)
        term2 = (pow(2, n * 2 // 3) - pow(2, n // 3) - pow(2, n)) \
            * Homework02._multiply_binary_numbers_helper(x_mid, y_mid)
        term3 = (1 - pow(2, n * 2 // 3) - pow(2, n // 3)) \
            * Homework02._multiply_binary_numbers_helper(x_right, y_right)
        term4 = pow(2, n) * Homework02._multiply_binary_numbers_helper(x_left + x_mid, y_mid + y_left)
        term5 = pow(2, n * 2 // 3) \
            * Homework02._multiply_binary_numbers_helper(x_left + x_right, y_right + y_left)
        term6 = pow(2, n // 3) \
            * Homework02._multiply_binary_numbers_helper(x_mid + x_right, y_right + y_mid)
        return term1 + term2 + term3 + term4 + term5 + term6
    
    @staticmethod
    def k_select_of_sorted_arrays(nums1: List[int], nums2: List[int], k: int) -> int:
        assert nums1 == sorted(nums1) and nums2 == sorted(nums2), "Input arrays must be sorted."
        assert len(nums1) + len(nums2) > k, f"{k} is out of bounds for {len(nums1) + len(nums2)}."
        return Homework02._k_select_helper(nums1, 0, len(nums1) - 1,
                                           nums2, 0, len(nums2) - 1, k)

    @staticmethod
    def _k_select_helper(nums1, nums1_start, nums1_end,
                         nums2, nums2_start, nums2_end, k):
        if nums1_start > nums1_end:
            return nums2[k - nums1_start]
        if nums2_start > nums2_end:
            return nums1[k - nums2_start]
        nums1_mid = (nums1_start + nums1_end) // 2
        nums2_mid = (nums2_start + nums2_end) // 2
        if nums1_mid + nums2_mid < k:
            if nums1[nums1_mid] > nums2[nums2_mid]:
                return Homework02._k_select_helper(nums1, nums1_start, nums1_end,
                                                   nums2, nums2_mid + 1, nums2_end, k)
            else:
                return Homework02._k_select_helper(nums1, nums1_mid + 1, nums1_end,
                                                   nums2, nums2_start, nums2_end, k)
        else:
            if nums1[nums1_mid] > nums2[nums2_mid]:
                return Homework02._k_select_helper(nums1, nums1_start, nums1_mid - 1,
                                                   nums2, nums2_start, nums2_end, k)
            else:
                return Homework02._k_select_helper(nums1, nums1_start, nums1_end,
                                                   nums2, nums2_start, nums2_mid - 1, k)
