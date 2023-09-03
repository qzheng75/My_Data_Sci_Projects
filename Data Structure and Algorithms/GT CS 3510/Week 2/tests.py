from class_algorithms import InClassAlgorithms
import unittest
import numpy as np

class TestInClassAlgorithms(unittest.TestCase):

    def test_multiply_binary_numbers_01(self):
        x = InClassAlgorithms.int_to_binary_list(1)
        y = InClassAlgorithms.int_to_binary_list(0)
        result = InClassAlgorithms.multiply_binary_numbers(x, y)
        self.assertEqual(result, 0)

    def test_multiply_binary_numbers_02(self):
        x = InClassAlgorithms.int_to_binary_list(7)
        y = InClassAlgorithms.int_to_binary_list(3)
        result = InClassAlgorithms.multiply_binary_numbers(x, y)
        self.assertEqual(result, 21)

    def test_multiply_binary_numbers_03(self):
        x = InClassAlgorithms.int_to_binary_list(0)
        y = InClassAlgorithms.int_to_binary_list(0)
        result = InClassAlgorithms.multiply_binary_numbers(x, y)
        self.assertEqual(result, 0)

    def test_multiply_binary_numbers_04(self):
        x = InClassAlgorithms.int_to_binary_list(63)
        y = InClassAlgorithms.int_to_binary_list(15)
        result = InClassAlgorithms.multiply_binary_numbers(x, y)
        self.assertEqual(result, 945)

    def test_multiply_binary_numbers_05(self):
        x = InClassAlgorithms.int_to_binary_list(255)
        y = InClassAlgorithms.int_to_binary_list(170)
        result = InClassAlgorithms.multiply_binary_numbers(x, y)
        self.assertEqual(result, 43350)

    def test_multiply_binary_numbers_06(self):
        x = InClassAlgorithms.int_to_binary_list(97)
        y = InClassAlgorithms.int_to_binary_list(256)
        result = InClassAlgorithms.multiply_binary_numbers(x, y)
        self.assertEqual(result, 24832)

    def test_k_select_M5(self):
        test_arr = list(range(1007))
        np.random.shuffle(test_arr)
        for i in range(1007):
            self.assertEqual(i, InClassAlgorithms.k_select(test_arr, i, 5),
                             msg=f'Fail at case when k = {i}')

    def test_k_select_M7(self):
        test_arr = list(range(1007))
        np.random.shuffle(test_arr)
        for i in range(1007):
            self.assertEqual(i, InClassAlgorithms.k_select(test_arr, i, 7),
                             msg=f'Fail at case when k = {i}')
            
    def test_k_select_special_case(self):
        # Test when length of array < M
        test_arr = [2, 3, 0, 1, 4]
        for i in range(len(test_arr)):
            self.assertEqual(i, InClassAlgorithms.k_select(test_arr, i, 7),
                             msg=f'Fail at case when k = {i}')


if __name__ == '__main__':
    unittest.main()
