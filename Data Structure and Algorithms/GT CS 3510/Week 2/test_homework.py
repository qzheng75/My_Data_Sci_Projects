from homework import Homework02
import unittest
import numpy as np


class TestInClassAlgorithms(unittest.TestCase):
    def test_multiply_binary_numbers_01(self):
        x = Homework02.int_to_binary_list(1, pad=27)
        y = Homework02.int_to_binary_list(0, pad=27)
        result = Homework02.multiply_binary_numbers(x, y)
        self.assertEqual(result, 0)

    def test_multiply_binary_numbers_02(self):
        x = Homework02.int_to_binary_list(7, pad=27)
        y = Homework02.int_to_binary_list(3, pad=27)
        result = Homework02.multiply_binary_numbers(x, y)
        self.assertEqual(result, 21)

    def test_multiply_binary_numbers_03(self):
        x = Homework02.int_to_binary_list(0, pad=27)
        y = Homework02.int_to_binary_list(0, pad=27)
        result = Homework02.multiply_binary_numbers(x, y)
        self.assertEqual(result, 0)

    def test_multiply_binary_numbers_04(self):
        x = Homework02.int_to_binary_list(63, pad=27)
        y = Homework02.int_to_binary_list(15, pad=27)
        result = Homework02.multiply_binary_numbers(x, y)
        self.assertEqual(result, 945)

    def test_multiply_binary_numbers_05(self):
        x = Homework02.int_to_binary_list(255, pad=27)
        y = Homework02.int_to_binary_list(170, pad=27)
        result = Homework02.multiply_binary_numbers(x, y)
        self.assertEqual(result, 43350)

    def test_multiply_binary_numbers_06(self):
        x = Homework02.int_to_binary_list(97, pad=27)
        y = Homework02.int_to_binary_list(256, pad=27)
        result = Homework02.multiply_binary_numbers(x, y)
        self.assertEqual(result, 24832)

    def test_k_select_of_sorted_arrays_01(self):
        nums1 = list(range(0, 5))
        nums2 = list(range(5, 9))
        for i in range(9):
            self.assertEqual(Homework02.k_select_of_sorted_arrays(nums1, nums2, i), i,
                             msg=f"Fail on case when k = {i}")
            
    def test_k_select_of_sorted_arrays_02(self):
        nums1 = list(range(0, 21, 2))
        nums2 = list(range(1, 20, 2))
        for i in range(20):
            self.assertEqual(Homework02.k_select_of_sorted_arrays(nums1, nums2, i), i,
                             msg=f"Fail on case when k = {i}")
            
    def test_k_select_of_sorted_arrays_03(self):
        all_nums = np.array(range(10000))
        ratio = 0.1
        while ratio < 1:
            arr1_size = int(10000 * ratio)
            arr1 = sorted(np.random.choice(all_nums, arr1_size, replace=False))
            arr2 = sorted(np.setdiff1d(all_nums, arr1))
            for i in range(100):
                self.assertEqual(Homework02.k_select_of_sorted_arrays(arr1, arr2, i), i,
                                msg=f"Fail on case when k = {i}")
            ratio += 0.02


if __name__ == '__main__':
    np.random.seed(1)
    unittest.main()
    