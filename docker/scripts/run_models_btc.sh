#!/bin/bash

# Execute restart_queue script to make TRAINING models as PENDING
ASSET="BTC"
python -m app.ml_models.restart_queue "$ASSET"

# Function to create a tmux pane, name it, and run a command
create_pane() {
    local pane_number=$1
    local pane_name=$2
    local python_command=$3

    tmux select-pane -t $pane_number
    tmux send-keys "tmux select-pane -T '$pane_name'" C-m
    tmux send-keys "$python_command" C-m
}

# Create a new tmux session with 4 panes in a 2x2 layout
tmux new-session -d -s "ml_btc_models" -n "btc_models"

tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Run BTC models in each pane with descriptive names
create_pane 0 "BTC_1M" "python -m app.ml_models.prod.06_btc_1m_from_btc_8h_multi"
create_pane 1 "BTC_2M" "python -m app.ml_models.prod.07_btc_2m_from_btc_8h_multi"
create_pane 2 "BTC_3M" "python -m app.ml_models.prod.08_btc_3m_from_btc_7d_multi"
create_pane 3 "BTC_6M" "python -m app.ml_models.prod.09_btc_6m_from_btc_7d_multi"

# Prevent the container from exiting
tail -f /dev/null