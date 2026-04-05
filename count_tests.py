import re

def count():
    with open('/tmp/telemetry.rs', 'r') as f:
        lines = f.readlines()

    test_start = 1646
    print(len(lines[test_start:]))

count()
