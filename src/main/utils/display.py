import datetime

from src.main.utils.logging import Color


def seconds_to_readable(seconds: float) -> str:
    """
    This method will return the human-readable format of the seconds given.

    :param seconds: Number of seconds. Must be a positive float.
    :return: A string representing a human-readable format of the seconds e.g., 02:33:44.
    """
    if seconds < 0:
        raise ValueError(f"Seconds ({seconds}) can't be a negative number.")

    minutes = seconds / 60
    ss = seconds % 60
    hh = minutes / 60
    mm = minutes % 60
    return "%02d:%02d:%02d" % (hh, mm, ss)


def readable_time(start_time: float, end_time: float, color_output: bool = True, display: bool = True) -> str:
    """
    This method would display a human-readable format of the seconds given as input, with the format of HH:MM:SS.

    :param start_time: The start time in seconds. Must be less than end time.
    :param end_time: The end time in seconds. Must be higher than end time.
    :param color_output: Binary value that if set to true will apply color to the displayed output.
    :param display: Binary value that if set to true will display the output in terminal with both the number of seconds
                    passed and the human-readable format.
    :return: A string representing a human-readable format of the seconds e.g., 02:33:44
    """
    if start_time >= end_time:
        raise ValueError(f"Start time ({start_time}) can't be larger than the end time ({end_time})")
    total_time = seconds_to_readable(end_time - start_time)

    if display:
        total_time_beautify = f"Time took: {total_time} | {round(end_time - start_time, 2)} seconds."
        if color_output:
            total_time_beautify = f"{Color.CYAN}{total_time_beautify}{Color.RESET}"
        print(total_time_beautify)

    return total_time


def date_time():
    """
    This method will compute the time at the runtime and return it into a string that can be passed as the name of
    directories or files.

    :return: Time as string.
    """
    current_time = datetime.datetime.now()

    return f"{current_time.day}-{current_time.month}-{current_time.year}_" \
           f"{current_time.hour}-{current_time.minute}-{current_time.second}"


if __name__ == '__main__':
    sample_time = 1_673_175_131
    end_sample = sample_time + 123_456
    readable_time(sample_time, end_sample)
