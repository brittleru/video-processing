import io
import unittest
from unittest.mock import patch

from src.main.utils.display import readable_time, seconds_to_readable
from src.main.utils.logging import Color


class TestSecondsToReadable(unittest.TestCase):
    def test_seconds_to_readable(self):
        test_cases = [
            (3665, "01:01:05"),
            (0, "00:00:00"),
            (88400, "24:33:20"),
            (59, "00:00:59"),
            (3599, "00:59:59"),
            (3665.5, "01:01:05"),
            (3665.9, "01:01:05"),
            (3665.1, "01:01:05"),
        ]

        for input_seconds, expected_output in test_cases:
            with self.subTest(input_seconds=input_seconds, expected_output=expected_output):
                actual_output = seconds_to_readable(input_seconds)
                self.assertEqual(actual_output, expected_output)

    def test_negative_seconds(self):
        with self.assertRaises(ValueError):
            seconds_to_readable(-100)

    def test_formatting(self):
        self.assertRegex(seconds_to_readable(3665), r'^\d{2}:\d{2}:\d{2}$')


class TestReadableTime(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(readable_time(0, 3665, color_output=False, display=False), "01:01:05")

    def test_same_start_end_time(self):
        with self.assertRaises(ValueError):
            readable_time(100, 100, color_output=False, display=False)

    def test_end_time_less_than_start_time(self):
        with self.assertRaises(ValueError):
            readable_time(200, 100, color_output=False, display=False)

    def test_time_display(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            self.assertEqual(readable_time(0, 3665, color_output=False, display=True), "01:01:05")
            printed_output = fake_stdout.getvalue().strip()
            self.assertEqual(printed_output, 'Time took: 01:01:05 | 3665 seconds.')

    def test_time_display_and_color_output(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            self.assertEqual(readable_time(0, 3665, color_output=True, display=True), "01:01:05")
            printed_output = fake_stdout.getvalue().strip()
            self.assertEqual(printed_output, f"{Color.CYAN}Time took: 01:01:05 | 3665 seconds.{Color.RESET}")


if __name__ == '__main__':
    unittest.main()
