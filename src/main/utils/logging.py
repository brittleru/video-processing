from typing import Final


class Color:
    RESET: Final[str] = '\033[0m'
    BOLD: Final[str] = '\033[1m'
    UNDERLINE: Final[str] = '\033[4m'
    BLACK: Final[str] = "\033[30m"
    MAGENTA: Final[str] = "\033[35m"
    DARKCYAN: Final[str] = '\033[36m'
    WHITE: Final[str] = "\033[37m"
    RED: Final[str] = '\033[91m'
    GREEN: Final[str] = '\033[92m'
    YELLOW: Final[str] = '\033[93m'
    BLUE: Final[str] = '\033[94m'
    PURPLE: Final[str] = '\033[95m'
    CYAN: Final[str] = '\033[96m'


if __name__ == '__main__':
    print(f"This looks like {Color.BOLD}color{Color.RESET} text.")
    print(f"This looks like {Color.UNDERLINE}color{Color.RESET} text.")
    print(f"This looks like {Color.BLACK}color{Color.RESET} text.")
    print(f"This looks like {Color.MAGENTA}color{Color.RESET} text.")
    print(f"This looks like {Color.DARKCYAN}color{Color.RESET} text.")
    print(f"This looks like {Color.WHITE}color{Color.RESET} text.")
    print(f"This looks like {Color.RED}color{Color.RESET} text.")
    print(f"This looks like {Color.GREEN}color{Color.RESET} text.")
    print(f"This looks like {Color.YELLOW}color{Color.RESET} text.")
    print(f"This looks like {Color.BLUE}color{Color.RESET} text.")
    print(f"This looks like {Color.PURPLE}color{Color.RESET} text.")
    print(f"This looks like {Color.CYAN}color{Color.RESET} text.")
