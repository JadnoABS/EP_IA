import unittest

from matrix import multiply_matrix, transpose_matrix


class TestMatrixMultiplication(unittest.TestCase):
    def test_raise_exception(self):
        first = [
            [1, 0],
            [1, 0],
        ]

        second = [
            [1, 0]
        ]

        with self.assertRaises(ArithmeticError) as error:
            multiply_matrix(first, second)

        self.assertIsNotNone(error)

    def test_success(self):
        second = [
            [1, 0],
            [0, 1],
        ]

        first = [
            [1, 0],
            [1, 0],
        ]

        self.assertEqual(first, multiply_matrix(first, second))


class TestTransposeMatrix(unittest.TestCase):
    def test_sucess(self):
        m = [
            [1, 3],
            [2, 4],
        ]

        expected = [
            [1, 2],
            [3, 4],
        ]

        self.assertEqual(expected, transpose_matrix(m))


if __name__ == '__main__':
    unittest.main()
