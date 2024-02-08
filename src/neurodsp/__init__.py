import ibldsp, sys
from warnings import warn

sys.modules["neurodsp"] = ibldsp
warn(
    "neurodsp has been renamed to ibldsp and the old name will be deprecated on 01-Sep-2024.",
    FutureWarning,
)
