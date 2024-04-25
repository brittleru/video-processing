import sys
from time import time

from src.main.utils.display import seconds_to_readable
from src.main.utils.logging import Color


class ProgressBar:
    __avg_times = []

    @staticmethod
    def run(
            num_iterations: int,
            current_iteration: int,
            prev_time: float,
            bar_width: int = 40,
            display_iter_width: int = 3
    ) -> None:
        """
        Outputs a progress bar in the terminal. It has the number of iterations that has been passed, the time took
        per iteration and an average time computed as the median of the previous iterations.

        :param num_iterations: The number of your iterations.
        :param current_iteration: The number of the current iteration.
        :param prev_time: The time from the previous iteration
        :param bar_width: The width of the bar to output in terminal, by default is 40.
        :param display_iter_width: The number of spaces that current iteration and number of iteration can be formatted.
                                   This is needed to have a smoother display, by default is 4. E.g., if your iteration has
                                   10k elements then change it to 5.
        :return: Void
        """
        current_iteration_percentage = float(current_iteration) / float(num_iterations)
        finished_value = ""
        color = Color.BLUE

        if current_iteration_percentage >= 1.:
            current_iteration_percentage = 1
            finished_value = "\r\n"
            color = Color.GREEN

        num_finished_blocks = int(round(bar_width * current_iteration_percentage))
        curr_time = time()
        bar_content = \
            f"\r{color}[{'#' * num_finished_blocks}{'-' * (bar_width - num_finished_blocks)}]{Color.RESET} | " \
            f"{color}{current_iteration:{display_iter_width}d}/{num_iterations:{display_iter_width}d} {Color.RESET}| " \
            f"{color}{Color.BOLD}{current_iteration_percentage * 100:.02f}% Done{Color.RESET}"

        if color is Color.BLUE:
            iteration_time = curr_time - prev_time
            ProgressBar.__avg_times.append(iteration_time)
            avg_time = ProgressBar.median(ProgressBar.__avg_times)
            bar_content += f" |{color} Time per iteration: {iteration_time:.4f}s{Color.RESET} | " \
                           f"{color}Estimated time to finish: " \
                           f"{seconds_to_readable(avg_time * (num_iterations - current_iteration))}" \
                           f"{Color.RESET}{finished_value}"
        else:
            bar_content += f"{finished_value}"

        sys.stdout.write(bar_content)
        sys.stdout.flush()

    @staticmethod
    def median(arr: list):
        """
        Method to compute the median from an array.

        :param arr: The given array.
        :return: The median of the array.
        """
        arr.sort()
        arr_len = len(arr)
        if arr_len % 2 == 0:
            median1 = arr[arr_len // 2]
            median2 = arr[arr_len // 2 - 1]
            return (median1 + median2) / 2
        return arr[arr_len // 2]


if __name__ == '__main__':
    from time import sleep

    iterations = 150
    for i in range(iterations):
        iter_time = time()
        sleep(.1)
        ProgressBar.run(iterations, i + 1, iter_time)
    print("Completed.")
