import os

def run():
    with open('/tmp/telemetry.rs', 'r') as f:
        lines = f.readlines()

    def get_code(ranges):
        out = []
        for s, e in ranges:
            # We want to grab previous empty lines and comments
            start_idx = s
            while start_idx > 0 and (lines[start_idx-1].strip() == '' or lines[start_idx-1].strip().startswith('//') or lines[start_idx-1].strip().startswith('#[')):
                start_idx -= 1
            out.append("".join(lines[start_idx:e+1]))
        return "\n".join(out)

    # Note: We need a reliable way to map out where each function goes.
    # What if we just use the sections separated by `// ───`?
    # section 0: imports + ArtifactAction + ArtifactRecord (types)
    # section 2: Execution Artifacts types, session metrics, MetricsCollector (aggregation), write_session_metrics (export) ... this is mixed!
    # So we CANNOT use sections directly.
    pass
run()
