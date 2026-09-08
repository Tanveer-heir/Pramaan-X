#!/usr/bin/env bash
# ==============================================================================
# Chandigarh Police Hackathon (CPH Section 2) - Docker Quick Runner
# ==============================================================================

set -e

CMD=${1:-"api"}
TARGET=${2:-"https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"}

case "$CMD" in
  build)
    echo "[*] Building CPH Source Attribution Docker image..."
    docker compose build
    ;;
  api|up)
    echo "[*] Starting Source Attribution microservice on http://localhost:8000..."
    docker compose up source-attribution
    ;;
  test_astar)
    echo "[*] Running A* Recursive Attribution on: $TARGET"
    docker compose run --rm source-attribution python test_astar_attribution.py "$TARGET"
    ;;
  test)
    echo "[*] Running automated test suite inside container..."
    docker compose run --rm source-attribution pytest tests/ -v
    ;;
  shell)
    echo "[*] Launching interactive container shell..."
    docker compose run --rm source-attribution bash
    ;;
  down)
    echo "[*] Stopping container..."
    docker compose down
    ;;
  *)
    echo "Usage: ./docker-run.sh [build | api | test_astar | test | shell | down] [target_url_or_path]"
    ;;
esac
