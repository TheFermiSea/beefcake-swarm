import re

with open('/tmp/telemetry.rs', 'r') as f:
    text = f.read()

# Strip out imports, we'll put them in all files to be safe, then clippy/fmt can sort it out.
# Let's extract items one by one.
import ast
# We don't have Rust AST parser.
