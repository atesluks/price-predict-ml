#!/bin/bash

# Execute restart_queue script to make TRAINING models as PENDING
ASSET="ETH"
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
tmux new-session -d -s "ml_eth_models" -n "eth_models"

tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Run ETH models in each pane with descriptive names
create_pane 0 "ETH_1M" "python -m app.ml_models.prod.10_eth_1m_from_eth_8h_multi"
create_pane 1 "ETH_2M" "python -m app.ml_models.prod.11_eth_2m_from_eth_8h_multi"
create_pane 2 "ETH_3M" "python -m app.ml_models.prod.12_eth_3m_from_eth_7d_multi"
create_pane 3 "ETH_6M" "python -m app.ml_models.prod.13_eth_6m_from_eth_7d_multi"

# Prevent the container from exiting
tail -f /dev/null