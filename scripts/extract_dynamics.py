import torch
import os
import argparse
import sys
from tqdm import tqdm

# Add root to path
sys.path.append(os.getcwd())
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dynamics_logger import DynamicsLogger
# Reuse loading logic from run.py or define similar
from tasks.mazes.analysis.run import _load_ctm_model
from data.custom_datasets import MazeImageFolder
from models.ctm import ContinuousThoughtMachine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt')
    parser.add_argument('--data_root', type=str, default='data/mazes/large/test')
    parser.add_argument('--output_file', type=str, default='data/dynamics_trajectories.h5')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='mps') # Default to mps for M3
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Device fallback
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load Model
    # We might need to ensure arguments match run.py expectations or hack them if run.py expects args object
    # _load_ctm_model expects checkpoint PATH and DEVICE
    model = _load_ctm_model(args.checkpoint, device)
    model.eval()
    
    # Load Data
    print(f"Loading data from {args.data_root}")
    # Using MazeImageFolder as in run.py
    # Note: run.py sets expand_range based on legacy_scaling. Assuming default (True -> expand_range=False? No wait)
    # run.py: expand_range=not args.legacy_scaling (default True) -> expand_range=False
    # Let's match run.py default
    test_data = MazeImageFolder(
        root=args.data_root, 
        which_set='test',
        maze_route_length=50, # Arbitrary max length for loader
        expand_range=False, 
        trunc=True
    )
    
    loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Logger
    logger = DynamicsLogger(args.output_file, expected_samples=args.num_samples)
    
    samples_processed = 0
    
    print(f"Starting extraction for {args.num_samples} samples...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            if samples_processed >= args.num_samples:
                break
                
            inputs = inputs.to(device)
            current_batch_size = inputs.size(0)
            
            # Truncate batch if we only need a few more
            if samples_processed + current_batch_size > args.num_samples:
                needed = args.num_samples - samples_processed
                inputs = inputs[:needed]
                targets = targets[:needed]
                current_batch_size = needed
            
            # Forward pass with tracking
            # model returns: predictions, certainties, (synch_out, synch_action), pre, post, attn
            predictions, certainties, synch, pre_acts, post_acts, attn = model(inputs, track=True)
            
            # Calculate metadata (Maze length)
            # targets is shape (B, Len) with 4 as padding?
            # From run.py: actual_lengths = (targets != 4).sum(dim=-1)
            lengths = (targets != 4).sum(dim=-1).cpu().numpy()
            
            metadata = {
                'maze_length': lengths
            }
            
            # Log
            logger.log_batch(predictions, certainties, synch, pre_acts, post_acts, metadata)
            
            samples_processed += current_batch_size
            
    logger.close()
    print(f"Extraction complete. Saved to {args.output_file}")
    
    # Verification of size
    file_size = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")

if __name__ == '__main__':
    main()
