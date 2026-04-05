import os
import re

with open('/tmp/telemetry.rs', 'r') as f:
    text = f.read()

# We can use regex to find items and move them.
# Even better, we can just split the file by the section headers!
# I observed the file has 12 parts separated by "// ───..."

parts = re.split(r'\n// ──+\n', text)
for i, part in enumerate(parts):
    with open(f'/tmp/part_{i}.rs', 'w') as f:
        f.write(part)
