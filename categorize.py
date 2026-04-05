import re

with open('/tmp/telemetry.rs', 'r') as f:
    text = f.read()

# Instead of splitting by lines, let's use the line ranges we found earlier.
# This will be perfectly safe.
types_ranges = [(12, 38), (44, 297), (783, 797), (1054, 1137), (1273, 1286)]
aggregation_ranges = [(298, 703), (903, 1049), (1138, 1268), (1287, 1402)]
export_ranges = [(704, 782), (798, 902)]
events_ranges = [(1407, 1645)]
init_ranges = []

# Tests ranges
# 1651-2800 is the huge block of tests, let's just grep for it. Wait, the file has 2718 lines!

def get_block(text, start_re, end_re=None):
    pass
