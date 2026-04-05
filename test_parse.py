import re

def process():
    with open('/tmp/telemetry.rs', 'r') as f:
        code = f.read()

    # We will use simple regexes to find blocks, or just use tree-sitter or syn if we could, but a python parser is easier.
    # Actually, we can use rust analyzer or rustc to parse, but let's just do line by line.

process()
