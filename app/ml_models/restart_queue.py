import argparse
from app.utils.model_training_queue import reset_training_status_to_pending

# This is done so that when a model training is interrupt (e.g. the container is stopped), once the 
# training container is releaunched, it will replace status from TRAINING to PENDING, so that model 
# training is picked up again. Otherwise the script will think that the model training is still 
# running and will wait for it forever.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reset training status to pending.')
    parser.add_argument('asset', type=str, help='Asset name (BTC or ETH).')
    
    args = parser.parse_args()
    
    reset_training_status_to_pending(args.asset)
    print(f"Replaced status for running {args.asset} models from TRAINING to PENDING")