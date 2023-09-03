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
        test_arr = list(range(507))
        np.random.shuffle(test_arr)
        for i in range(507):
            self.assertEqual(i, InClassAlgorithms.k_select(test_arr, i, 5),
                             msg=f'Fail at case when k = {i}')

    def test_k_select_M7(self):
        test_arr = list(range(507))
        np.random.shuffle(test_arr)
        for i in range(507):
            self.assertEqual(i, InClassAlgorithms.k_select(test_arr, i, 7),
                             msg=f'Fail at case when k = {i}')
            
    def test_k_select_special_case(self):
        # Test when length of array < M
        test_arr = [2, 3, 0, 1, 4]
        for i in range(len(test_arr)):
            self.assertEqual(i, InClassAlgorithms.k_select(test_arr, i, 7),
                             msg=f'Fail at case when k = {i}')
            
    def test_strassen_01(self):
        matrix1 = np.array([[1, 2], [3, 4]])
        matrix2 = np.array([[5, 6], [7, 8]])
        result = InClassAlgorithms.strassen_fast_matmul(matrix1, matrix2)
        self.assertTrue(np.array_equal(result, np.matmul(matrix1, matrix2)))

    def test_strassen_02(self):
        matrix1 = np.array([[1, 1, 1, 1],
                            [2, 2, 2, 2],
                            [3, 3, 3, 3],
                            [2, 2, 2, 2]])
        matrix2 = np.array([[1, 1, 1, 1],
                            [2, 2, 2, 2],
                            [3, 3, 3, 3],
                            [2, 2, 2, 2]])
        result = InClassAlgorithms.strassen_fast_matmul(matrix1, matrix2)
        self.assertTrue(np.array_equal(result, np.matmul(matrix1, matrix2)))

    def test_strassen_03(self):
        matrix1 = np.array([[5]])
        matrix2 = np.array([[2]])
        result = InClassAlgorithms.strassen_fast_matmul(matrix1, matrix2)
        self.assertTrue(np.array_equal(result, np.matmul(matrix1, matrix2)))

    def test_strassen_04(self):
        matrix1 = np.random.randint(0, 11, size=(8, 8), dtype=int)
        matrix2 = np.random.randint(0, 11, size=(8, 8), dtype=int)

        result = InClassAlgorithms.strassen_fast_matmul(matrix1, matrix2)
        correct_result = np.matmul(matrix1, matrix2)
        self.assertTrue(np.array_equal(result, correct_result))

    def test_strassen_05_large_matrices(self):
        matrix1 = np.random.randint(0, 11, size=(128, 128), dtype=int)
        matrix2 = np.random.randint(0, 11, size=(128, 128), dtype=int)

        result = InClassAlgorithms.strassen_fast_matmul(matrix1, matrix2)
        correct_result = np.matmul(matrix1, matrix2)
        self.assertTrue(np.array_equal(result, correct_result))

    def test_fast_power(self):
        base = 2
        exp = 5
        self.assertEqual(InClassAlgorithms.fast_pow(base, exp), pow(base, exp))

        exp = 6
        self.assertEqual(InClassAlgorithms.fast_pow(base, exp), pow(base, exp))


if __name__ == '__main__':
    unittest.main()
