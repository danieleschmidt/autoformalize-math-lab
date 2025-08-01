#!/bin/bash
# Autonomous SDLC Enhancement Scheduler
# Schedules continuous value discovery and execution

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Configuration
PYTHON="${PYTHON:-python3}"
DISCOVERY_SCRIPT=".terragon/simple_discovery.py"
EXECUTION_SCRIPT="scripts/autonomous_execution.py"
LOG_DIR="logs"
LOCK_FILE=".terragon/autonomous.lock"

# Create necessary directories
mkdir -p "$LOG_DIR" ".terragon"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/autonomous.log"
}

# Check if another instance is running
check_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local lock_pid=$(cat "$LOCK_FILE")
        if kill -0 "$lock_pid" 2>/dev/null; then
            log "Another autonomous process is running (PID: $lock_pid)"
            exit 1
        else
            log "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
    trap 'rm -f "$LOCK_FILE"' EXIT
}

# Run value discovery
run_discovery() {
    log "🔍 Running value discovery..."
    if $PYTHON "$DISCOVERY_SCRIPT" > "$LOG_DIR/discovery.log" 2>&1; then
        log "✅ Value discovery completed"
        return 0
    else
        log "❌ Value discovery failed"
        return 1
    fi
}

# Run autonomous execution
run_execution() {
    log "🤖 Running autonomous execution..."
    if $PYTHON "$EXECUTION_SCRIPT" > "$LOG_DIR/execution.log" 2>&1; then
        log "✅ Autonomous execution completed"
        return 0
    else
        log "💭 No execution performed this cycle"
        return 1
    fi
}

# Main execution modes
case "${1:-run}" in
    "discovery")
        log "🚀 Starting autonomous value discovery..."
        check_lock
        run_discovery
        ;;
    
    "execution")
        log "🚀 Starting autonomous execution..."
        check_lock
        run_execution
        ;;
    
    "run"|"")
        log "🚀 Starting full autonomous cycle..."
        check_lock
        
        # Run discovery first
        if run_discovery; then
            # Then attempt execution
            run_execution
        fi
        ;;
    
    "schedule")
        log "📅 Setting up autonomous scheduling..."
        
        # Add cron jobs for different schedules
        echo "Setting up cron jobs..."
        
        # Every hour: value discovery
        (crontab -l 2>/dev/null || true; echo "0 * * * * cd $REPO_DIR && $0 discovery") | crontab -
        
        # Every 4 hours: full execution cycle
        (crontab -l 2>/dev/null || true; echo "0 */4 * * * cd $REPO_DIR && $0 run") | crontab -
        
        # Daily: comprehensive analysis
        (crontab -l 2>/dev/null || true; echo "0 2 * * * cd $REPO_DIR && make qa && $0 run") | crontab -
        
        log "✅ Cron jobs scheduled"
        crontab -l
        ;;
    
    "unschedule")
        log "🗑️  Removing autonomous scheduling..."
        crontab -l | grep -v "$REPO_DIR" | crontab - || true
        log "✅ Cron jobs removed"
        ;;
    
    "status")
        log "📊 Autonomous system status..."
        
        # Check if processes are running
        if [ -f "$LOCK_FILE" ]; then
            echo "🔒 Lock file exists (PID: $(cat $LOCK_FILE))"
        else
            echo "🔓 No active processes"
        fi
        
        # Show recent logs
        echo "📋 Recent activity:"
        if [ -f "$LOG_DIR/autonomous.log" ]; then
            tail -10 "$LOG_DIR/autonomous.log"
        else
            echo "No log file found"
        fi
        
        # Show current opportunities
        if [ -f "BACKLOG.md" ]; then
            echo "🎯 Current opportunities:"
            grep -A 5 "Next Best Value" BACKLOG.md || true
        fi
        ;;
    
    "install")
        log "🛠️  Installing autonomous system..."
        
        # Make scripts executable
        chmod +x "$0"
        chmod +x "$EXECUTION_SCRIPT" 2>/dev/null || true
        
        # Install Python dependencies if needed
        if command -v pip3 >/dev/null; then
            pip3 install --user --upgrade pip || true
        fi
        
        # Set up pre-commit hooks if available
        if [ -f ".pre-commit-config.yaml" ] && command -v pre-commit >/dev/null; then
            pre-commit install || true
        fi
        
        log "✅ Autonomous system installed"
        ;;
    
    "help"|"-h"|"--help")
        cat << EOF
Terragon Autonomous SDLC Enhancement Scheduler

Usage: $0 [COMMAND]

Commands:
    discovery     Run value discovery only
    execution     Run autonomous execution only
    run           Run full cycle (discovery + execution) [default]
    schedule      Set up cron jobs for autonomous operation
    unschedule    Remove autonomous cron jobs
    status        Show system status and recent activity
    install       Install and set up the autonomous system
    help          Show this help message

Environment Variables:
    PYTHON        Python executable to use (default: python3)

Files:
    logs/autonomous.log     Main log file
    .terragon/autonomous.lock    Lock file for process coordination
    BACKLOG.md              Generated value opportunities backlog

Examples:
    $0                      # Run full autonomous cycle
    $0 discovery            # Run value discovery only
    $0 schedule             # Set up hourly/daily automation
    $0 status               # Check system status
EOF
        ;;
    
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

log "🏁 Autonomous cycle completed"