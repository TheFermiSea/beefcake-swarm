import os

with open('/tmp/telemetry.rs', 'r') as f:
    lines = f.readlines()

def get_block(lines, start_line):
    # finds the end of an item
    # Not needed, we already have exact line numbers from get_item_ranges.py!
    pass
