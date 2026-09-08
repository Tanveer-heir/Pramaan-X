# ==============================================================================
# Chandigarh Police Hackathon (CPH Section 2) - Docker Quick Runner (PowerShell)
# ==============================================================================

param (
    [string]$Command = "api",
    [string]$Target = "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"
)

switch ($Command) {
    "build" {
        Write-Host "[*] Building CPH Source Attribution Docker image..." -ForegroundColor Cyan
        docker compose build
    }
    "api" {
        Write-Host "[*] Starting Source Attribution microservice on http://localhost:8000..." -ForegroundColor Cyan
        docker compose up source-attribution
    }
    "up" {
        Write-Host "[*] Starting Source Attribution microservice on http://localhost:8000..." -ForegroundColor Cyan
        docker compose up source-attribution
    }
    "test_astar" {
        Write-Host "[*] Running A* Recursive Attribution on: $Target" -ForegroundColor Green
        docker compose run --rm source-attribution python test_astar_attribution.py "$Target"
    }
    "test" {
        Write-Host "[*] Running automated test suite inside container..." -ForegroundColor Yellow
        docker compose run --rm source-attribution pytest tests/ -v
    }
    "shell" {
        Write-Host "[*] Launching interactive container shell..." -ForegroundColor Magenta
        docker compose run --rm source-attribution bash
    }
    "down" {
        Write-Host "[*] Stopping container..." -ForegroundColor Yellow
        docker compose down
    }
    Default {
        Write-Host "Usage: .\docker-run.ps1 [-Command <build|api|test_astar|test|shell|down>] [-Target <url_or_path>]" -ForegroundColor Red
    }
}
