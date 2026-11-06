"""
Persistent dtypes for the ``/DST/Pointlike`` HDF5 table.

Peak identifiers are signed to retain the ``-1`` missing-peak sentinel. Times
and widths are in microseconds, positions in millimeters, and integrals are
unweighted sample sums. Floating fields can store NaN for missing properties.
These types do not define PMAP, blame or event-count tables.
"""

import numpy as np

dst_types = dict(
    event           = np.uint32,
    s1no            = np.int16,
    s2no            = np.int16,
    s1_width        = np.float32,
    s1_height       = np.float32,
    s1_time         = np.float32,
    s1_integral     = np.float32,
    s1_multiplicity = np.uint16,
    s2_width        = np.float32,
    s2_height       = np.float32,
    s2_time         = np.float32,
    s2_integral     = np.float32,
    s2_multiplicity = np.uint16,
    drift_time      = np.float32,
    x               = np.float32,
    y               = np.float32,
)
