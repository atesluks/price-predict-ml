#!/bin/bash

# Execute restart_queue script to make TRAINING models as PENDING
python -m app.ml_models.restart_queue

# Get model configs from config.yaml
MODEL_CONFIGS=$(python -c "from app.utils.misc import get_model_configs; print('\n'.join([config['name'] for config in get_model_configs()]))")
NUM_MODELS=$(echo "$MODEL_CONFIGS" | wc -l)

# Function to create a tmux pane, name it, and run a command
create_pane() {
    local pane_number=$1
    local model_name=$2
    local python_command=$3

    tmux select-pane -t $pane_number
    tmux send-keys "tmux select-pane -T '$model_name'" C-m
    tmux send-keys "$python_command" C-m
}

# Calculate number of windows needed (each window has 4 panes)
NUM_WINDOWS=$(( ($NUM_MODELS + 3) / 4 ))

# Create session and windows, then run models
tmux new-session -d -s "models"
window=0
model_index=0

while IFS= read -r model_name; do
    # Create new window for every 4 models
    if (( $model_index % 4 == 0 )); then
        WINDOW_NAME="window_$window"
        if [ $window -gt 0 ]; then
            tmux new-window -t models:$window -n "$WINDOW_NAME"
        fi

        # Create 2x2 layout in reading order:
        # First split horizontally, then split each half vertically
        tmux split-window -v -p 50
        tmux select-pane -t 0
        tmux split-window -h -p 50
        tmux select-pane -t 2
        tmux split-window -h -p 50
        
        pane=0
    fi

    # Create pane with model name
    create_pane $pane "$model_name" "python -m app.ml_models.prod.general_model_training $model_index"

    ((model_index++))
    ((pane++))
    
    # Increment window counter every 4 models
    if (( $model_index % 4 == 0 )); then
        ((window++))
    fi
done <<< "$MODEL_CONFIGS"

# Prevent the container from exiting
tail -f /dev/null