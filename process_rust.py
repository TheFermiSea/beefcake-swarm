import re

with open('/tmp/telemetry.rs', 'r') as f:
    lines = f.readlines()

blocks = []
current_block = []
brace_count = 0
in_item = False

for line in lines:
    is_item_start = re.match(r'^(pub )?(struct|enum|fn|const|impl|mod) ', line.strip()) or line.startswith('#[cfg(test)]')

    if is_item_start and brace_count == 0:
        if current_block:
            blocks.append(''.join(current_block))
        current_block = []
        in_item = True

        # Pull in previous comments/attributes
        # This is hard to do line-by-line forward, let's just do it.
