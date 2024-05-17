from typing import Final, Tuple


class QuiltingTypes:
    RANDOM_PLACEMENT: Final[str] = "random_placement"
    NEIGHBORING_BLOCKS: Final[str] = "neighboring_blocks"
    MINIMUM_ERROR: Final[str] = "minimum_error"
    SAMPLE_CORRELATION: Final[str] = "sample_correlation"
    SAMPLE_OVERLAP: Final[str] = "sample_overlap"


class ColorBGR:
    GREEN: Final[Tuple[int, int, int]] = (0, 255, 0)
