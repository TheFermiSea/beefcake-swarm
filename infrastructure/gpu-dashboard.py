#!/usr/bin/env python3
import asyncio
import subprocess
import json
import time
import sys
import os
from datetime import datetime

# Configuration
NODES = [
    {"name": "qe-node1",  "ip": "10.0.0.10", "user": "ubuntu", "pass": "ubuntu"},
    {"name": "qe-node2",  "ip": "10.0.0.11", "user": "ubuntu", "pass": "ubuntu"},
    {"name": "qe-node3",  "ip": "10.0.0.12", "user": "ubuntu", "pass": "ubuntu"},
    {"name": "vasp-01",   "ip": "10.0.0.20", "user": "root",   "pass": "adminadmin"},
    {"name": "vasp-02",   "ip": "10.0.0.21", "user": "root",   "pass": "adminadmin"},
    {"name": "vasp-03",   "ip": "10.0.0.22", "user": "root",   "pass": "adminadmin"},
]
REFRESH_RATE = 2

# ANSI Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

def clear_screen():
    print("\033[H\033[J", end="")

async def get_gpu_stats(node):
    """
    Connects to node via SSH and retrieves GPU stats.
    Returns a dict with status and data.
    """
    # Command to get CSV data: index, name, temperature.gpu, utilization.gpu, memory.used, memory.total
    # We use -o ConnectTimeout=2 to fail fast if node is down
    cmd = [
        "sshpass", "-p", node['pass'],
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=2", "-o", "LogLevel=QUIET",
        f"{node['user']}@{node['ip']}",
        "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            err_msg = stderr.decode().strip()
            # Handle standard SSH errors gracefully
            if "No route to host" in err_msg or "timed out" in err_msg:
                return {"node": node, "status": "down", "msg": "Host Unreachable"}
            if "Permission denied" in err_msg:
                return {"node": node, "status": "auth_fail", "msg": "Auth Failed"}
            # Return full error if unknown
            return {"node": node, "status": "error", "msg": err_msg or "Command failed (likely node down)"}
            
        # Parse output
        output = stdout.decode().strip()
        if not output:
             return {"node": node, "status": "empty", "msg": "No GPU found"}
        
        gpus = []
        for line in output.split('\n'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "temp": int(parts[2]),
                    "util": int(parts[3]),
                    "mem_used": int(parts[4]),
                    "mem_total": int(parts[5])
                })
        
        return {"node": node, "status": "ok", "gpus": gpus}

    except Exception as e:
        return {"node": node, "status": "error", "msg": str(e)}

def print_dashboard(results):
    clear_screen()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{BOLD}=== Beefcake GPU Cluster Dashboard === {RESET} {timestamp}")
    print(f"{ 'Node':<12} {'IP':<12} {'GPU':<22} {'Temp':<6} {'Util':<6} {'Memory (MiB)':<20} {'Bar':<20}")
    print("-" * 105)

    active_gpus = 0
    total_gpus = 0

    for res in results:
        node = res['node']
        name_str = f"{node['name']}"
        ip_str = f"{node['ip']}"
        
        status = res['status']
        if status == "ok":
            for gpu in res['gpus']:
                total_gpus += 1
                active_gpus += 1
                
                # Color coding for utilization
                util = gpu['util']
                util_color = GREEN
                if util > 50: util_color = YELLOW
                if util > 85: util_color = RED
                
                # Color coding for memory
                mem_pct = (gpu['mem_used'] / gpu['mem_total']) * 100
                mem_color = GREEN
                if mem_pct > 50: mem_color = YELLOW
                if mem_pct > 85: mem_color = RED

                # Progress bar
                bar_len = 20
                filled = int(util / 100 * bar_len)
                bar = "[" + "#" * filled + " " * (bar_len - filled) + "]"
                
                print(f"{name_str:<12} {ip_str:<12} {gpu['name']:<22} {gpu['temp']}C   {util_color}{util:>3}%{RESET}  {mem_color}{gpu['mem_used']:>5} / {gpu['mem_total']:<5}{RESET}    {util_color}{bar}{RESET}")
        elif status == "down":
             print(f"{name_str:<12} {ip_str:<12} {DIM}{'Host Down / Unreachable':<60}{RESET}")
        else:
             print(f"{name_str:<12} {ip_str:<12} {RED}{res.get('msg', 'Error'):<60}{RESET}")
             
    print("-" * 105)
    print(f"Summary: {active_gpus} Active GPUs / {len(NODES)} Nodes Monitored")
    print(f"{DIM}Press Ctrl+C to exit{RESET}")

async def main_loop():
    print("Initializing connection to nodes...")
    while True:
        tasks = [get_gpu_stats(node) for node in NODES]
        results = await asyncio.gather(*tasks)
        
        # Sort results by name to keep stable order
        results.sort(key=lambda x: x['node']['name'])
        
        print_dashboard(results)
        time.sleep(REFRESH_RATE)

if __name__ == "__main__":
    # Check for sshpass
    if subprocess.call(["which", "sshpass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print(f"{RED}Error: sshpass is not installed.{RESET}")
        print("Please install it: brew install sshpass (Mac) or apt install sshpass (Linux)")
        sys.exit(1)

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)