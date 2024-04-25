import unittest
from io import StringIO
from time import time
from unittest.mock import patch

from src.main.utils.progress import ProgressBar


class TestProgressBar(unittest.TestCase):

    @patch('sys.stdout', new_callable=StringIO)
    def test_progress_bar(self, mock_stdout):
        iterations = 10
        prev_time = time()
        for i in range(iterations):
            ProgressBar.run(iterations, i + 1, prev_time)
            prev_time = time()
        output = mock_stdout.getvalue()

        self.assertIn("100.00% Done", output)

    def test_median(self):
        expected_median = 8
        test_data = [3, 6, 8, 9, 10]
        actual_median = ProgressBar.median(test_data)

        self.assertEqual(actual_median, expected_median)

    def test_median_even(self):
        expected_median = 7.0
        test_data = [3, 6, 8, 9]
        actual_median = ProgressBar.median(test_data)

        self.assertEqual(actual_median, expected_median)


if __name__ == '__main__':
    unittest.main()
