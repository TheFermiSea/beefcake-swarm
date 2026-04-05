import os

os.system('cp /tmp/telemetry.rs crates/swarm-agents/src/telemetry.rs')
os.system('rm -rf crates/swarm-agents/src/telemetry/')
