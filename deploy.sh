#!/usr/bin/env bash
# ==============================================================================
# Autonomous Data Scientist (DataPilot) - Production Deployment Supervisor
# Orchestrates FastAPI (8000), Streamlit (8501), React (3000), and MLflow (5001)
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

LOGS_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"
mkdir -p "$LOGS_DIR" "$PID_DIR"

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
UVICORN_BIN="$PROJECT_ROOT/.venv/bin/uvicorn"
STREAMLIT_BIN="$PROJECT_ROOT/.venv/bin/streamlit"
MLFLOW_BIN="$PROJECT_ROOT/.venv/bin/mlflow"

COLOR_ORANGE='\033[38;5;208m'
COLOR_GREEN='\033[0;32m'
COLOR_BLUE='\033[0;34m'
COLOR_RED='\033[0;31m'
COLOR_NC='\033[0m'

banner() {
    echo -e "${COLOR_ORANGE}"
    echo "======================================================================"
    echo "       🧠 DataPilot • Autonomous Data Scientist Platform            "
    echo "======================================================================"
    echo -e "${COLOR_NC}"
}

start_service() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"
    local logfile="$LOGS_DIR/$name.log"
    local cmd="$2"

    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo -e "  ● $name is already running [PID $(cat "$pidfile")]"
        return
    fi

    echo -ne "  ⏳ Launching $name... "
    eval "$cmd > '$logfile' 2>&1 &"
    local pid=$!
    echo "$pid" > "$pidfile"
    sleep 1.5

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${COLOR_GREEN}ACTIVE [PID $pid]${COLOR_NC} (log: logs/$name.log)"
    else
        echo -e "${COLOR_RED}FAILED${COLOR_NC}"
        tail -n 10 "$logfile"
    fi
}

stop_service() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"

    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo -ne "  ⏳ Stopping $name [PID $pid]... "
            kill "$pid" 2>/dev/null || true
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            echo -e "${COLOR_GREEN}STOPPED${COLOR_NC}"
        else
            echo -e "  ○ $name was not running."
        fi
        rm -f "$pidfile"
    fi
}

check_http() {
    local name="$1"
    local url="$2"
    local status=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    if [ "$status" = "200" ]; then
        echo -e "  ● $name: ${COLOR_GREEN}ONLINE (HTTP 200)${COLOR_NC} -> $url"
    else
        echo -e "  ○ $name: ${COLOR_RED}OFFLINE ($status)${COLOR_NC} -> $url"
    fi
}

start_all() {
    banner
    echo -e "${COLOR_BLUE}▶ Starting all 4 platform services...${COLOR_NC}\n"

    # 1. FastAPI Gateway (Port 8000)
    start_service "api" "$UVICORN_BIN app.api.main:app --host 127.0.0.1 --port 8000"

    # 2. Streamlit Workstation (Port 8501)
    start_service "streamlit" "$STREAMLIT_BIN run dashboard/app.py --server.port 8501 --server.headless true"

    # 3. React Vite UI (Port 3000)
    start_service "frontend" "npm run dev -- --host 127.0.0.1 --port 3000"

    # 4. MLflow UI (Port 5001)
    start_service "mlflow" "$MLFLOW_BIN ui --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db"

    echo ""
    echo -e "${COLOR_GREEN}✔ All services started successfully!${COLOR_NC}\n"
    check_status
}

stop_all() {
    banner
    echo -e "${COLOR_BLUE}▶ Stopping all platform services...${COLOR_NC}\n"
    stop_service "frontend"
    stop_service "streamlit"
    stop_service "api"
    stop_service "mlflow"

    # Also clean any orphan ports
    for port in 8000 8501 3000 5001; do
        lsof -ti :$port | xargs kill -9 2>/dev/null || true
    done
    echo -e "\n${COLOR_GREEN}✔ All platform services have been stopped.${COLOR_NC}\n"
}

check_status() {
    echo -e "${COLOR_ORANGE}--- Service Health Checks ---${COLOR_NC}"
    check_http "React Control Center " "http://127.0.0.1:3000"
    check_http "Streamlit Workstation" "http://127.0.0.1:8501"
    check_http "FastAPI API Gateway  " "http://127.0.0.1:8000/health"
    check_http "MLflow Tracking UI   " "http://127.0.0.1:5001"
    echo ""
    echo -e "Access Points:"
    echo -e "  🌐 Frontend:  http://localhost:3000"
    echo -e "  📊 Studio:    http://localhost:8501"
    echo -e "  ⚡ API Docs:  http://localhost:8000/docs"
    echo -e "  🧪 MLflow:    http://localhost:5001"
    echo ""
}

case "$1" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        banner
        check_status
        ;;
    *)
        echo "Usage: ./deploy.sh {start|stop|restart|status}"
        exit 1
        ;;
esac
