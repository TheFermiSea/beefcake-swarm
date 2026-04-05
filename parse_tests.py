import os

with open('/tmp/telemetry.rs', 'r') as f:
    lines = f.readlines()

# All tests are from line 1646 to end.
# We'll just put them in aggregation.rs since it has only 800 lines so far and with 1000 lines of test it might go over 1500. Let's check wc -l.
