import os.path
import tempfile
import unittest
from unittest.mock import patch

from src.main.utils.files_manipulation import create_dir_if_doesnt_exist, get_path_of_files, purge_files_in_dir
from src.main.utils.logging import Color


class TestFileManipulation(unittest.TestCase):

    def setUp(self) -> None:
        self.test_dir = "./test_dir"
        self.empty_dir = "empty_directory"
        os.makedirs(self.empty_dir, exist_ok=True)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_files = ["1.jpg", "3.jpg", "10.jpg", "102.jpg", "2.jpg"]
        for filename in self.test_files:
            open(os.path.join(self.temp_dir.name, filename), 'a').close()

        self.test_dir_purge = tempfile.TemporaryDirectory()
        self.test_files_purge = ["test1.jpg", "test2.jpg", "test3.txt"]
        for file in self.test_files_purge:
            open(os.path.join(self.test_dir_purge.name, file), 'w').close()

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
        if os.path.exists(self.empty_dir):
            os.rmdir(self.empty_dir)

        self.temp_dir.cleanup()
        self.test_dir_purge.cleanup()

    @patch('sys.stdout', None)
    def test_create_dir_if_doesnt_exist_creation(self):
        directory_created = create_dir_if_doesnt_exist(self.test_dir)
        self.assertTrue(directory_created)
        self.assertTrue(os.path.exists(self.test_dir))

    @patch('sys.stdout', None)
    def test_create_dir_if_doesnt_exist_existing(self):
        os.makedirs(self.test_dir, exist_ok=True)
        directory_created = create_dir_if_doesnt_exist(self.test_dir)
        self.assertFalse(directory_created)

    def test_get_path_of_files(self):
        result = get_path_of_files(self.temp_dir.name)
        expected_result = [
            os.path.join(self.temp_dir.name, "1.jpg"),
            os.path.join(self.temp_dir.name, "2.jpg"),
            os.path.join(self.temp_dir.name, "3.jpg"),
            os.path.join(self.temp_dir.name, "10.jpg"),
            os.path.join(self.temp_dir.name, "102.jpg")
        ]
        self.assertEqual(expected_result, result)

    def test_different_file_type(self):
        result = get_path_of_files(self.temp_dir.name, file_type=".mp4")
        self.assertEqual([], result)

    def test_empty_directory(self):
        result = get_path_of_files("/nonexistent/path")
        self.assertEqual([], result)

    def test_purge_files_in_dir_existing_directory(self):
        with patch('builtins.print') as mocked_print:
            result = purge_files_in_dir(self.test_dir_purge.name)
            self.assertTrue(result)

            mocked_print.assert_called_once_with(
                f"{Color.GREEN}Successfully deleted all '.jpg' contents in {Color.BOLD}'"
                f"{os.path.basename(self.test_dir_purge.name)}'{Color.RESET}{Color.GREEN} directory.{Color.RESET}"
            )

    def test_purge_files_in_dir_non_existing_directory(self):
        with patch('builtins.print') as mocked_print:
            result = purge_files_in_dir("non_existing_directory")
            self.assertFalse(result)
            mocked_print.assert_called_once_with(
                f"{Color.RED}Directory {Color.BOLD}'non_existing_directory'{Color.RESET}"
                f"{Color.RED} doesn't exist.{Color.RESET}"
            )

    def test_purge_files_in_dir_empty_directory(self):
        with patch('builtins.print') as mocked_print:
            result = purge_files_in_dir(self.empty_dir)
            self.assertFalse(result)
            mocked_print.assert_called_once_with(
                f"{Color.YELLOW}Directory {Color.BOLD}'{os.path.basename(self.empty_dir)}'{Color.RESET}"
                f"{Color.YELLOW} is empty. Or has no {Color.BOLD}'.jpg'{Color.RESET}{Color.YELLOW} files.{Color.RESET}"
            )


if __name__ == '__main__':
    unittest.main()
