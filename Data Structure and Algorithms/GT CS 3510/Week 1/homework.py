from typing import List
import numpy as np


class Homework01Algorithms:
    @staticmethod
    def int_to_binary_list(num: int, pad=32):
        binary_string = bin(num)[2:]
        padding_length = pad - len(binary_string)
        padded_binary_string = '0' * padding_length + binary_string
        return [int(bit) for bit in padded_binary_string]

    @staticmethod
    def compare_binary_numbers(x: List[int], y: List[int]) -> str:
        """
        Compare two non-negative integers represented in binary format.

        Given two non-negative integers, x and y, each stored as binary arrays of n bits,
        this function determines their relationship: whether x is greater than y, y is greater than x,
        or if they are equal.

        Parameters:
        - x (List[int]): Binary representation of the first non-negative integer.
        - y (List[int]): Binary representation of the second non-negative integer.

        Returns:
        - str: A string indicating the comparison result: 'x > y', 'x < y', or 'x = y'.

        Constraints:
        - Both input numbers have to be binary represented.
        - The lengths of both input arrays x and y must be equal.

        Time Complexity:
        - Worst case: O(n)
        """
        assert all(i in {0, 1} for i in x), 'x is not a binary-represented number.'
        assert all(i in {0, 1} for i in y), 'y is not a binary-represented number.'
        assert len(x) == len(y), 'Numbers have to be represented in same number of bits.'
        n = len(x)
        for i in range(n):
            if x[i] != y[i]:
                return 'x > y' if x[i] == 1 else 'x < y'
        return 'x = y'
    
    @staticmethod
    def star_matrix_multiplication(n: int, v: np.ndarray) -> np.ndarray:
        """
        For any number n, such that n is a power of 2, the Star matrix,
        S_n is defined as follows:
            if n = 1, S_n = [1]
            if n > 1, S_n =
                (3S_{n/2} I_{n/2}
                 S_{n/2}  -2S_{n/2})
            Where I_k denotes the k * k identity matrix

        This method implements an algorithm that calculates S_n * v,
        where n is a power of 2 and v is a vector of length n.

        Parameters:
        - n (int): Size of star matrix involved in calculation, must be a power of 2.
        - v (np.ndarray): A vector of length n.

        Returns:
        - np.ndarray: The result of S_n * v.

        Constraints:
        - n must be a power of 2.
        - Length of input vector v has to be n.

        Time complexity:
        - Worst case: O(nlogn)
        """
        assert (n & (n - 1)) == 0, f"{n} is not a power of 2."
        assert len(v) == n, f'Shape mismatch: n = {n}, yet vector has length {len(v)}.'
        if n == 1:
            return np.array(v)
        mid_idx = n // 2
        v_upper = v[0 : mid_idx]
        v_lower = v[mid_idx :]
        S_v1 = Homework01Algorithms.star_matrix_multiplication(n // 2, v_upper)
        S_v2 = Homework01Algorithms.star_matrix_multiplication(n // 2, v_lower)
        S_upper = 3 * S_v1 + v_lower.reshape(-1, 1)
        S_lower = S_v1 - 2 * S_v2
        return np.vstack((S_upper, S_lower))


if __name__ == '__main__':
    x = 68
    y = 69
    # print(Homework01Algorithms.compare_binary_numbers(
    #     Homework01Algorithms.int_to_binary_list(x),
    #     Homework01Algorithms.int_to_binary_list(y)
    # ))
    n = 32
    # print(Homework01Algorithms.star_matrix_multiplication(n, [1] * n))
        
