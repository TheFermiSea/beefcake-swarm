 #!/bin/bash
 ###############################################################################
 # Qwen3-Coder-Next Deployment Script with MoE CPU Offloading
 #
 # This script deploys Qwen3-Coder-Next (80B MoE, 3B active) on a single node
 # using llama.cpp with MoE layer offloading to CPU.
 #
 # Hardware Requirements:
 #   - GPU: 32GB VRAM (V100S) - holds attention layers
 #   - RAM: 256GB - holds MoE expert layers
 #   - CPU: 40 cores - for MoE computation
 #
 # Usage:
 #   ./deploy-qwen3-coder-next.sh                    # Validate + deploy on vasp-01
 #   ./deploy-qwen3-coder-next.sh --validate-only    # Only run validation
 #   ./deploy-qwen3-coder-next.sh --node vasp-02     # Deploy on specific node
 #   ./deploy-qwen3-coder-next.sh --download-only    # Only download model
 #   ./deploy-qwen3-coder-next.sh --quant Q5_K_XL    # Use different quantization
 #
 # Model: unsloth/Qwen3-Coder-Next-GGUF (UD-Q4_K_XL by default, ~49GB)
 # Docs: https://unsloth.ai/docs/models/qwen3-coder-next
 ###############################################################################
 
 set -euo pipefail
 
 # Script location
 SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 
 # Colors
 RED='\033[0;31m'
 GREEN='\033[0;32m'
 YELLOW='\033[1;33m'
 BLUE='\033[0;34m'
 NC='\033[0m'
 
 # Default configuration
 TARGET_NODE="${TARGET_NODE:-vasp-01}"
 QUANT="${QUANT:-Q4_K_XL}"
 MODEL_NAME="Qwen3-Coder-Next"
 MODEL_REPO="unsloth/Qwen3-Coder-Next-GGUF"
 MODEL_DIR="/scratch/ai/models"
 CONTAINER="/cluster/shared/containers/llama-server.sif"
 PORT="${PORT:-8002}"
 CTX_SIZE="${CTX_SIZE:-32768}"
 
 # MoE offloading configuration
 # -ot ".ffn_.*_exps.=CPU" offloads all MoE layers to CPU
 # This keeps attention on GPU for speed, experts on CPU for capacity
 MOE_OFFLOAD_PATTERN=".ffn_.*_exps.=CPU"
 
 # Minimum requirements
 MIN_RAM_GB=200
 MIN_VRAM_GB=28
 MIN_DISK_GB=60
 
 # Parse arguments
 VALIDATE_ONLY=false
 DOWNLOAD_ONLY=false
 DEPLOY=true
 
 while [[ $# -gt 0 ]]; do
     case $1 in
         --validate-only)
             VALIDATE_ONLY=true
             DEPLOY=false
             shift
             ;;
         --download-only)
             DOWNLOAD_ONLY=true
             DEPLOY=false
             shift
             ;;
         --node)
             TARGET_NODE="$2"
             shift 2
             ;;
         --quant)
             QUANT="$2"
             shift 2
             ;;
         --port)
             PORT="$2"
             shift 2
             ;;
         --ctx-size)
             CTX_SIZE="$2"
             shift 2
             ;;
         --help|-h)
             echo "Usage: $0 [OPTIONS]"
             echo ""
             echo "Options:"
             echo "  --validate-only    Only run pre-deployment validation"
             echo "  --download-only    Only download the model"
             echo "  --node NODE        Target node (default: vasp-01)"
             echo "  --quant QUANT      Quantization level (default: Q4_K_XL)"
             echo "                     Options: Q4_K_XL (~49GB), Q5_K_XL (~57GB),"
             echo "                              Q6_K (~66GB), Q8_0 (~85GB)"
             echo "  --port PORT        Server port (default: 8002)"
             echo "  --ctx-size SIZE    Context size (default: 32768)"
             echo ""
             echo "Quantization Memory Requirements:"
             echo "  Q4_K_XL: ~49GB (recommended for 256GB RAM + 32GB VRAM)"
             echo "  Q5_K_XL: ~57GB (better quality, slightly slower)"
             echo "  Q6_K:    ~66GB (high quality)"
             echo "  Q8_0:    ~85GB (highest quality)"
             exit 0
             ;;
         *)
             echo "Unknown option: $1"
             exit 1
             ;;
     esac
 done
 
 # Model filename based on quantization
 MODEL_FILE="${MODEL_NAME}-UD-${QUANT}.gguf"
 MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
 
 # Validation counters
 PASSED=0
 FAILED=0
 WARNINGS=0
 
 pass() {
     echo -e "${GREEN}[PASS]${NC} $1"
     ((PASSED++))
 }
 
 fail() {
     echo -e "${RED}[FAIL]${NC} $1"
     ((FAILED++))
 }
 
 warn() {
     echo -e "${YELLOW}[WARN]${NC} $1"
     ((WARNINGS++))
 }
 
 info() {
     echo -e "${BLUE}[INFO]${NC} $1"
 }
 
 section() {
     echo ""
     echo -e "${BLUE}=== $1 ===${NC}"
 }
 
 ###############################################################################
 # Pre-Deployment Validation
 ###############################################################################
 
 validate_node() {
     section "Node Connectivity: ${TARGET_NODE}"
     
     # Check SSH connectivity
     if ssh -o ConnectTimeout=10 -o BatchMode=yes "${TARGET_NODE}" "echo ok" &>/dev/null; then
         pass "SSH connectivity to ${TARGET_NODE}"
     else
         fail "Cannot SSH to ${TARGET_NODE}"
         return 1
     fi
     
     # Check node is not running other AI workloads on target port
     if ssh "${TARGET_NODE}" "ss -tlnp 2>/dev/null | grep -q ':${PORT} '" 2>/dev/null; then
         fail "Port ${PORT} already in use on ${TARGET_NODE}"
     else
         pass "Port ${PORT} is available"
     fi
 }
 
 validate_hardware() {
     section "Hardware Requirements: ${TARGET_NODE}"
     
     # Check RAM
     RAM_KB=$(ssh "${TARGET_NODE}" "grep MemTotal /proc/meminfo | awk '{print \$2}'")
     RAM_GB=$((RAM_KB / 1024 / 1024))
     if [[ $RAM_GB -ge $MIN_RAM_GB ]]; then
         pass "RAM: ${RAM_GB}GB (required: ${MIN_RAM_GB}GB)"
     else
         fail "RAM: ${RAM_GB}GB (required: ${MIN_RAM_GB}GB)"
     fi
     
     # Check available RAM (not just total)
     AVAIL_KB=$(ssh "${TARGET_NODE}" "grep MemAvailable /proc/meminfo | awk '{print \$2}'")
     AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
     if [[ $AVAIL_GB -ge 50 ]]; then
         pass "Available RAM: ${AVAIL_GB}GB"
     else
         warn "Available RAM: ${AVAIL_GB}GB (may be tight, consider freeing memory)"
     fi
     
     # Check GPU
     if ssh "${TARGET_NODE}" "nvidia-smi &>/dev/null"; then
         GPU_NAME=$(ssh "${TARGET_NODE}" "nvidia-smi --query-gpu=name --format=csv,noheader" | head -1)
         VRAM_MB=$(ssh "${TARGET_NODE}" "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits" | head -1)
         VRAM_GB=$((VRAM_MB / 1024))
         
         if [[ $VRAM_GB -ge $MIN_VRAM_GB ]]; then
             pass "GPU: ${GPU_NAME} with ${VRAM_GB}GB VRAM"
         else
             fail "GPU VRAM: ${VRAM_GB}GB (required: ${MIN_VRAM_GB}GB)"
         fi
         
         # Check GPU memory usage
         VRAM_USED=$(ssh "${TARGET_NODE}" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | head -1)
         VRAM_FREE=$((VRAM_MB - VRAM_USED))
         VRAM_FREE_GB=$((VRAM_FREE / 1024))
         if [[ $VRAM_FREE_GB -ge 20 ]]; then
             pass "GPU memory available: ${VRAM_FREE_GB}GB"
         else
             warn "GPU memory available: ${VRAM_FREE_GB}GB (may conflict with other workloads)"
         fi
     else
         fail "nvidia-smi not available - GPU not detected"
     fi
     
     # Check CPU cores
     CPU_CORES=$(ssh "${TARGET_NODE}" "nproc")
     if [[ $CPU_CORES -ge 32 ]]; then
         pass "CPU cores: ${CPU_CORES}"
     else
         warn "CPU cores: ${CPU_CORES} (32+ recommended for MoE offloading)"
     fi
 }
 
 validate_storage() {
     section "Storage Requirements"
     
     # Check model directory exists
     if ssh "${TARGET_NODE}" "test -d ${MODEL_DIR}"; then
         pass "Model directory exists: ${MODEL_DIR}"
     else
         info "Creating model directory: ${MODEL_DIR}"
         ssh "${TARGET_NODE}" "mkdir -p ${MODEL_DIR}"
         pass "Model directory created: ${MODEL_DIR}"
     fi
     
     # Check available disk space
     DISK_AVAIL=$(ssh "${TARGET_NODE}" "df -BG ${MODEL_DIR} | tail -1 | awk '{print \$4}' | tr -d 'G'")
     if [[ $DISK_AVAIL -ge $MIN_DISK_GB ]]; then
         pass "Disk space available: ${DISK_AVAIL}GB (required: ${MIN_DISK_GB}GB)"
     else
         fail "Disk space available: ${DISK_AVAIL}GB (required: ${MIN_DISK_GB}GB)"
     fi
     
     # Check if model already exists
     if ssh "${TARGET_NODE}" "test -f ${MODEL_PATH}"; then
         MODEL_SIZE=$(ssh "${TARGET_NODE}" "du -h ${MODEL_PATH} | cut -f1")
         pass "Model already downloaded: ${MODEL_FILE} (${MODEL_SIZE})"
     else
         warn "Model not found: ${MODEL_FILE} (will need to download)"
     fi
 }
 
 validate_software() {
     section "Software Requirements"
     
     # Check container exists
     if ssh "${TARGET_NODE}" "test -f ${CONTAINER}"; then
         CONTAINER_SIZE=$(ssh "${TARGET_NODE}" "du -h ${CONTAINER} | cut -f1")
         pass "Container exists: ${CONTAINER} (${CONTAINER_SIZE})"
     else
         fail "Container not found: ${CONTAINER}"
         info "Build with: cd ${SCRIPT_DIR} && ./build.sh --push"
     fi
     
     # Check llama-server supports MoE offloading
     if ssh "${TARGET_NODE}" "apptainer exec ${CONTAINER} llama-server --help 2>&1 | grep -q '\-ot'"; then
         pass "llama-server supports -ot (tensor offloading)"
     else
         warn "Cannot confirm -ot support (may need updated llama.cpp)"
     fi
     
     # Check apptainer/singularity
     if ssh "${TARGET_NODE}" "command -v apptainer &>/dev/null || command -v singularity &>/dev/null"; then
         APPTAINER_VER=$(ssh "${TARGET_NODE}" "apptainer --version 2>/dev/null || singularity --version 2>/dev/null")
         pass "Container runtime: ${APPTAINER_VER}"
     else
         fail "apptainer/singularity not found"
     fi
     
     # Check huggingface-cli for download
     if ssh "${TARGET_NODE}" "command -v huggingface-cli &>/dev/null"; then
         pass "huggingface-cli available"
     else
         warn "huggingface-cli not found (needed for model download)"
         info "Install with: pip install huggingface_hub"
     fi
 }
 
 validate_network() {
     section "Network Configuration"
     
     # Check if we can reach HuggingFace (for model download)
     if ssh "${TARGET_NODE}" "curl -sf --connect-timeout 5 https://huggingface.co/api/health &>/dev/null"; then
         pass "HuggingFace API reachable"
     else
         warn "Cannot reach HuggingFace API (may need proxy for model download)"
     fi
     
     # Check hostname resolution
     TARGET_FQDN=$(ssh "${TARGET_NODE}" "hostname -f")
     pass "Node FQDN: ${TARGET_FQDN}"
 }
 
 validate_conflicts() {
     section "Workload Conflicts"
     
     # Check for running SLURM jobs on node
     SLURM_JOBS=$(ssh slurm-ctl "squeue -w ${TARGET_NODE} -h 2>/dev/null | wc -l" || echo "0")
     if [[ "$SLURM_JOBS" -gt 0 ]]; then
         warn "Active SLURM jobs on ${TARGET_NODE}: ${SLURM_JOBS}"
         ssh slurm-ctl "squeue -w ${TARGET_NODE} 2>/dev/null" || true
     else
         pass "No active SLURM jobs on ${TARGET_NODE}"
     fi
     
     # Check for other inference servers
     LLAMA_PROCS=$(ssh "${TARGET_NODE}" "pgrep -f 'llama-server|llama-cli' 2>/dev/null | wc -l" || echo "0")
     if [[ "$LLAMA_PROCS" -gt 0 ]]; then
         warn "Existing llama processes on ${TARGET_NODE}: ${LLAMA_PROCS}"
         ssh "${TARGET_NODE}" "ps aux | grep -E 'llama-server|llama-cli' | grep -v grep" || true
     else
         pass "No conflicting llama processes"
     fi
     
     # Check GPU processes
     GPU_PROCS=$(ssh "${TARGET_NODE}" "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l" || echo "0")
     if [[ "$GPU_PROCS" -gt 0 ]]; then
         warn "GPU processes running: ${GPU_PROCS}"
         ssh "${TARGET_NODE}" "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv" || true
     else
         pass "GPU is idle"
     fi
 }
 
 run_validation() {
     echo "==========================================="
     echo "Qwen3-Coder-Next Pre-Deployment Validation"
     echo "==========================================="
     echo "Target Node: ${TARGET_NODE}"
     echo "Model: ${MODEL_FILE}"
     echo "Port: ${PORT}"
     echo "Context Size: ${CTX_SIZE}"
     echo "==========================================="
     
     validate_node || return 1
     validate_hardware
     validate_storage
     validate_software
     validate_network
     validate_conflicts
     
     # Summary
     section "Validation Summary"
     echo -e "${GREEN}Passed:${NC}   ${PASSED}"
     echo -e "${RED}Failed:${NC}   ${FAILED}"
     echo -e "${YELLOW}Warnings:${NC} ${WARNINGS}"
     
     if [[ $FAILED -gt 0 ]]; then
         echo ""
         echo -e "${RED}Validation failed. Fix issues above before deploying.${NC}"
         return 1
     fi
     
     if [[ $WARNINGS -gt 0 ]]; then
         echo ""
         echo -e "${YELLOW}Validation passed with warnings. Review above.${NC}"
     else
         echo ""
         echo -e "${GREEN}All checks passed!${NC}"
     fi
     
     return 0
 }
 
 ###############################################################################
 # Model Download
 ###############################################################################
 
 download_model() {
     section "Model Download"
     
     # Check if model already exists
     if ssh "${TARGET_NODE}" "test -f ${MODEL_PATH}"; then
         MODEL_SIZE=$(ssh "${TARGET_NODE}" "du -h ${MODEL_PATH} | cut -f1")
         info "Model already exists: ${MODEL_PATH} (${MODEL_SIZE})"
         echo -n "Re-download? [y/N]: "
         read -r response
         if [[ ! "$response" =~ ^[Yy]$ ]]; then
             return 0
         fi
     fi
     
     info "Downloading ${MODEL_FILE} to ${TARGET_NODE}:${MODEL_DIR}..."
     info "This may take 30-60 minutes depending on connection speed."
     
     # Install huggingface_hub if needed
     ssh "${TARGET_NODE}" "pip install -q huggingface_hub 2>/dev/null || pip3 install -q huggingface_hub" || true
     
     # Download model
     ssh "${TARGET_NODE}" "huggingface-cli download ${MODEL_REPO} \
         --local-dir ${MODEL_DIR}/Qwen3-Coder-Next-GGUF \
         --include '*UD-${QUANT}*'"
     
     # Move to standard location
     ssh "${TARGET_NODE}" "mv ${MODEL_DIR}/Qwen3-Coder-Next-GGUF/${MODEL_FILE} ${MODEL_PATH} 2>/dev/null || true"
     
     # Verify download
     if ssh "${TARGET_NODE}" "test -f ${MODEL_PATH}"; then
         MODEL_SIZE=$(ssh "${TARGET_NODE}" "du -h ${MODEL_PATH} | cut -f1")
         pass "Model downloaded successfully: ${MODEL_SIZE}"
     else
         fail "Model download failed"
         return 1
     fi
 }
 
 ###############################################################################
 # Deployment
 ###############################################################################
 
 deploy_server() {
     section "Deploying Qwen3-Coder-Next Server"
     
     # Verify model exists
     if ! ssh "${TARGET_NODE}" "test -f ${MODEL_PATH}"; then
         fail "Model not found: ${MODEL_PATH}"
         info "Run with --download-only first"
         return 1
     fi
     
     info "Starting llama-server with MoE CPU offloading..."
     info "  Model: ${MODEL_PATH}"
     info "  Port: ${PORT}"
     info "  Context: ${CTX_SIZE}"
     info "  MoE Offload: ${MOE_OFFLOAD_PATTERN}"
     
     # Create endpoint registration
     ENDPOINT_DIR="/cluster/shared/ai/endpoints"
     ssh "${TARGET_NODE}" "mkdir -p ${ENDPOINT_DIR}"
     
     # Generate startup script on target
     ssh "${TARGET_NODE}" "cat > /tmp/qwen3-coder-start.sh << 'STARTUP_EOF'
 #!/bin/bash
 set -euo pipefail
 
 MODEL_PATH='${MODEL_PATH}'
 CONTAINER='${CONTAINER}'
 PORT=${PORT}
 CTX_SIZE=${CTX_SIZE}
 MOE_OFFLOAD='${MOE_OFFLOAD_PATTERN}'
 ENDPOINT_DIR='${ENDPOINT_DIR}'
 LOG_DIR='/scratch/ai/logs'
 
 mkdir -p \"\${LOG_DIR}\"
 
 # Register endpoint
 cat > \"\${ENDPOINT_DIR}/qwen3-coder-next.json\" <<EOF
 {
     \"model\": \"Qwen3-Coder-Next\",
     \"mode\": \"moe-offload\",
     \"node\": \"\$(hostname)\",
     \"host\": \"\$(hostname -f)\",
     \"port\": \${PORT},
     \"endpoint\": \"http://\$(hostname -f):\${PORT}/v1/chat/completions\",
     \"models_endpoint\": \"http://\$(hostname -f):\${PORT}/v1/models\",
     \"context_size\": \${CTX_SIZE},
     \"started_at\": \"\$(date -Iseconds)\"
 }
 EOF
 
 echo \"[\$(date)] Starting Qwen3-Coder-Next with MoE CPU offloading...\"
 echo \"[\$(date)] Endpoint: http://\$(hostname -f):\${PORT}/v1\"
 
 # Run with MoE offloading
 # -ot offloads MoE expert layers to CPU, keeps attention on GPU
 # --threads for CPU inference threads
 # --flash-attn for efficient attention (if supported)
 exec apptainer run --nv \\
     --bind \"\$(dirname \${MODEL_PATH})\":\"\$(dirname \${MODEL_PATH})\":ro \\
     --bind \"\${LOG_DIR}\":\"\${LOG_DIR}\":rw \\
     \"\${CONTAINER}\" \\
     --model \"\${MODEL_PATH}\" \\
     --alias \"Qwen3-Coder-Next\" \\
     -ot \"\${MOE_OFFLOAD}\" \\
     --host 0.0.0.0 \\
     --port \${PORT} \\
     --ctx-size \${CTX_SIZE} \\
     --threads 32 \\
    --flash-attn auto \\
     --temp 1.0 \\
     --top-p 0.95 \\
     --min-p 0.01 \\
     --top-k 40 \\
     --cont-batching \\
     --metrics \\
     --jinja
 STARTUP_EOF
 chmod +x /tmp/qwen3-coder-start.sh"
     
     info "Starting server in background..."
     ssh "${TARGET_NODE}" "nohup /tmp/qwen3-coder-start.sh > /scratch/ai/logs/qwen3-coder-next.log 2>&1 &"
     
     # Wait for server to start
     info "Waiting for server to initialize (this may take 2-5 minutes for model loading)..."
     for i in {1..60}; do
         sleep 5
         if ssh "${TARGET_NODE}" "curl -sf http://localhost:${PORT}/health &>/dev/null"; then
             pass "Server is healthy!"
             break
         fi
         echo -n "."
         if [[ $i -eq 60 ]]; then
             fail "Server failed to start within 5 minutes"
             info "Check logs: ssh ${TARGET_NODE} 'tail -100 /scratch/ai/logs/qwen3-coder-next.log'"
             return 1
         fi
     done
     
     echo ""
     section "Deployment Complete"
     
     TARGET_FQDN=$(ssh "${TARGET_NODE}" "hostname -f")
     echo "API Endpoint: http://${TARGET_FQDN}:${PORT}/v1/chat/completions"
     echo "Models List:  http://${TARGET_FQDN}:${PORT}/v1/models"
     echo "Health Check: http://${TARGET_FQDN}:${PORT}/health"
     echo "Metrics:      http://${TARGET_FQDN}:${PORT}/metrics"
     echo ""
     echo "Test with:"
     echo "  curl -X POST http://${TARGET_FQDN}:${PORT}/v1/chat/completions \\"
     echo "    -H 'Content-Type: application/json' \\"
     echo "    -d '{\"model\":\"Qwen3-Coder-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a hello world in Rust\"}]}'"
     echo ""
     echo "Logs: ssh ${TARGET_NODE} 'tail -f /scratch/ai/logs/qwen3-coder-next.log'"
 }
 
 ###############################################################################
 # Main
 ###############################################################################
 
 main() {
     echo ""
     echo "==========================================="
     echo "Qwen3-Coder-Next Deployment"
     echo "==========================================="
     echo "Model:   ${MODEL_NAME} (${QUANT})"
     echo "Node:    ${TARGET_NODE}"
     echo "Port:    ${PORT}"
     echo "Context: ${CTX_SIZE} tokens"
     echo "==========================================="
     
     # Always run validation
     if ! run_validation; then
         exit 1
     fi
     
     if $VALIDATE_ONLY; then
         exit 0
     fi
     
     if $DOWNLOAD_ONLY; then
         download_model
         exit $?
     fi
     
     if $DEPLOY; then
         # Check if model needs download
         if ! ssh "${TARGET_NODE}" "test -f ${MODEL_PATH}" 2>/dev/null; then
             echo ""
             echo -n "Model not found. Download now? [Y/n]: "
             read -r response
             if [[ ! "$response" =~ ^[Nn]$ ]]; then
                 download_model || exit 1
             else
                 echo "Cannot deploy without model. Exiting."
                 exit 1
             fi
         fi
         
         deploy_server
     fi
 }
 
 main "$@"
