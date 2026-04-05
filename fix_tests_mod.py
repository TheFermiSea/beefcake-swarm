import os

for fname in ['aggregation.rs', 'export.rs', 'types.rs', 'events.rs']:
    path = f'crates/swarm-agents/src/telemetry/{fname}'
    with open(path, 'r') as f:
        content = f.read()

    # We appended the second mod tests to aggregation.rs. Let's fix it by renaming.
    count = 0
    def replacer(m):
        global count
        count += 1
        return f"mod tests_{count} {{"

    content = content.replace("mod tests {", "mod tests_main {", 1)
    if "mod tests {" in content:
         content = content.replace("mod tests {", "mod tests_extra {")

    with open(path, 'w') as f:
        f.write(content)
