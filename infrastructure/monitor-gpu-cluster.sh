#!/bin/bash
# Wrapper script to run the GPU dashboard
# Ensures dependencies are met

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found"
    exit 1
fi

# Check for sshpass
if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass could not be found"
    echo "Install with: brew install sshpass (Mac) or apt install sshpass (Linux)"
    exit 1
fi

# Run the python script
exec python3 "$DIR/gpu_dashboard.py"
