path = 'crates/swarm-agents/src/telemetry/aggregation.rs'
with open(path, 'r') as f:
    lines = f.readlines()

# Let's remove the extra closing brace at the end
# Or we can just count braces and remove the unbalanced one.
if lines[-1].strip() == '}':
    lines.pop()
if lines[-1].strip() == '}': # there might be a newline
    pass

with open(path, 'w') as f:
    f.writelines(lines[:-1]) if lines[-1].strip() == '}' else f.writelines(lines)
