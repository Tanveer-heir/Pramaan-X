#!/usr/bin/env bash

set -u

# Run this script from the repository root or keep it inside the repository root.
# Expected folders: CPH, Detection, frontend, prnu-device-attribution-api-handoff

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "Error: gnome-terminal is not installed."
  echo "Install it with: sudo apt install gnome-terminal"
  exit 1
fi

for required_dir in CPH Detection frontend prnu-device-attribution-api-handoff; do
  if [ ! -d "$REPO_ROOT/$required_dir" ]; then
    echo "Error: missing repository directory: $REPO_ROOT/$required_dir"
    exit 1
  fi
done

start_terminal() {
  local title="$1"
  local command_text="$2"

  gnome-terminal --title="$title" -- bash -lc "
    $command_text
    exit_code=\$?
    echo
    echo \"[$title] stopped with exit code \$exit_code\"
    read -r -p 'Press Enter to close this terminal...' _
    exit \$exit_code
  "
}

SHARED_MEDIA_DIR="$REPO_ROOT/CPH/data/shared_media"
INVESTIGATION_OUTPUT_DIR="$REPO_ROOT/CPH/data/investigations"

mkdir -p "$SHARED_MEDIA_DIR" "$INVESTIGATION_OUTPUT_DIR"

# Terminal 1: PRNU device attribution service
start_terminal "Pramaan-X PRNU :8002" "
  cd '$REPO_ROOT/prnu-device-attribution-api-handoff'
  source .venv/bin/activate
  export SHARED_MEDIA_ROOT='$SHARED_MEDIA_DIR'
  export ALLOW_DEMO_MODEL=false
  echo 'Starting PRNU service on http://127.0.0.1:8002'
  python -m prnu_attribution.api \\
    --host 127.0.0.1 \\
    --port 8002 \\
    --dataset dataset \\
    --model-dir models
"

# Terminal 2: multimodal Detection service
start_terminal "Pramaan-X Detection :8001" "
  cd '$REPO_ROOT/Detection'
  source ../.venv/bin/activate
  if [ -f .env ]; then
    set -a
    source .env
    set +a
  fi
  export PRAMAAN_DEVICE=\${PRAMAAN_DEVICE:-cuda}
  export PRAMAAN_FUSION_CHECKPOINT='$REPO_ROOT/Detection/checkpoints/fusion/pramaan_x_hackathon_final.pth'
  export DETECTION_ALLOWED_MEDIA_ROOT='$SHARED_MEDIA_DIR'
  echo \"Starting Detection service on http://127.0.0.1:8001 using \$PRAMAAN_DEVICE\"
  python -m uvicorn service.main:app \\
    --host 127.0.0.1 \\
    --port 8001
"

# Terminal 3: CPH gateway
start_terminal "Pramaan-X CPH :8000" "
  cd '$REPO_ROOT/CPH'
  source .venv/bin/activate
  if [ -f .env ]; then
    set -a
    source .env
    set +a
  fi
  export DETECTION_SERVICE_URL=http://127.0.0.1:8001
  export PRNU_SERVICE_URL=http://127.0.0.1:8002
  export SHARED_MEDIA_DIR='$SHARED_MEDIA_DIR'
  export INVESTIGATION_OUTPUT_DIR='$INVESTIGATION_OUTPUT_DIR'
  echo 'Starting CPH gateway on http://127.0.0.1:8000'
  python -m uvicorn src.api.main:app \\
    --host 127.0.0.1 \\
    --port 8000
"

# Terminal 4: frontend
start_terminal "Pramaan-X Frontend :5173" "
  cd '$REPO_ROOT/frontend'
  if [ ! -d node_modules ]; then
    echo 'node_modules not found. Installing frontend dependencies...'
    if [ -f package-lock.json ]; then
      npm ci
    else
      npm install
    fi
  fi
  echo 'Starting frontend on http://127.0.0.1:5173'
  npm run dev -- --host 127.0.0.1
"

echo
echo 'Started four service terminals:'
echo '  PRNU       http://127.0.0.1:8002'
echo '  Detection  http://127.0.0.1:8001'
echo '  CPH        http://127.0.0.1:8000'
echo '  Frontend   http://127.0.0.1:5173'
echo
echo 'Open http://127.0.0.1:5173 in your browser.'
