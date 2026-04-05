import re

def get_test_blocks():
    with open('/tmp/telemetry.rs', 'r') as f:
        lines = f.readlines()

    tests_start = 1646
    test_lines = lines[tests_start:]

    # We will just split tests by `#[test]`
    blocks = []
    current_block = []

    for line in test_lines:
        if line.strip() == '#[test]' or line.strip() == '#[tokio::test]':
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        else:
            if current_block:
                current_block.append(line)
            else:
                # Pre-test stuff (like `use super::*;`)
                pass
    if current_block:
        blocks.append(current_block)

    for i, b in enumerate(blocks):
        print(f"Test {i}: {len(b)} lines, {b[1].strip() if len(b)>1 else ''}")

get_test_blocks()
