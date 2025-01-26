#!/bin/bash

# Create a new tmux session with the first pane
tmux new-session -d -s data_workers -n workers

# Run the first command in the first pane
tmux send-keys -t data_workers:workers.0 "echo 'Starting 1_data pane'; python -m app.data_workers.coingecko_hourly_price_worker" C-m

# Split the window horizontally and run the second command in the new pane
tmux split-window -h -t data_workers:workers
tmux send-keys -t data_workers:workers.1 "echo 'Starting 2_data pane'; python -m app.data_workers.coingecko_daily_price_worker" C-m

echo "Both workers started and running in a tmux session. Connect to a container and use 'tmux a' command to connect to a session, then press 'control' + 'b', then 'd' to detach from the tmux session."

# Prevent the container from exiting
tail -f /dev/null