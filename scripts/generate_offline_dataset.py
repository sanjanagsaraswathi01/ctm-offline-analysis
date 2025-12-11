import torch
import numpy as np
import h5py
import os
import sys
from tqdm import tqdm
import argparse

# Path setup
sys.path.append(os.getcwd())
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from offline_mode import OfflineCTMWrapper
from tasks.mazes.analysis.run import _load_ctm_model
from data.custom_datasets import MazeImageFolder

def generate_random_walk(start_states, steps, std=0.1):
    """Generates a random walk starting from start_states."""
    B, D = start_states.shape
    walks = [start_states]
    current = start_states
    for _ in range(steps - 1):
        step = torch.randn_like(current) * std
        current = current + step
        walks.append(current)
    return torch.stack(walks, dim=1) # (B, T, D)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt')
    parser.add_argument('--data_root', default='data/mazes/large/test')
    parser.add_argument('--output', default='data/offline_extended_trajectories.h5')
    parser.add_argument('--num_samples', type=int, default=200) # 200 is sufficient for stats
    parser.add_argument('--cutoff', type=int, default=25)
    parser.add_argument('--extend_factor', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load
    ctm = _load_ctm_model(args.checkpoint, device)
    wrapper = OfflineCTMWrapper(ctm)
    
    dataset = MazeImageFolder(args.data_root, which_set='test', maze_route_length=50, expand_range=False, trunc=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Store lists
    data_store = {k: [] for k in ['control', 'mask', 'noise', 'random_walk', 'labels']}
    
    processed = 0
    print(f"Generating Offline Data (Cutoff={args.cutoff}, Extend={args.extend_factor})...")
    
    with torch.no_grad():
        for x, y in tqdm(loader):
            if processed >= args.num_samples: break
            x = x.to(device)
            B = x.size(0)
            
            # 1. Control (Extended)
            # Use 'none' variant with extension
            _, s_control = wrapper.forward_offline(x, variant='none', cutoff=args.cutoff, extend_factor=args.extend_factor)
            
            # 2. Mask
            _, s_mask = wrapper.forward_offline(x, variant='mask', cutoff=args.cutoff, extend_factor=args.extend_factor)
            
            # 3. Noise
            _, s_noise = wrapper.forward_offline(x, variant='noise', cutoff=args.cutoff, extend_factor=args.extend_factor)
            
            # 4. Random Walk Baseline
            # Start from state at cutoff? Or start?
            # Metric usually compares entropy of entire trajectory.
            # Let's match s_control shape.
            # Start from initial state of control
            s_rw = generate_random_walk(s_control[:, 0, :], s_control.shape[1], std=0.05) # Small step size
            
            # Append
            data_store['control'].append(s_control.cpu().numpy())
            data_store['mask'].append(s_mask.cpu().numpy())
            data_store['noise'].append(s_noise.cpu().numpy())
            data_store['random_walk'].append(s_rw.cpu().numpy())
            
            # Labels (Length for now, or just class placeholder)
            # y is targets. Calculate length
            lengths = (y != 4).sum(dim=-1).cpu().numpy()
            data_store['labels'].append(lengths)
            
            processed += B
            
    # Concat
    print("Concatenating...")
    for k in data_store:
        data_store[k] = np.concatenate(data_store[k], axis=0)[:args.num_samples]
        
    # Save
    print(f"Saving to {args.output}")
    with h5py.File(args.output, 'w') as f:
        for k, v in data_store.items():
            # Use float16 for storage
            dtype = 'float16' if 'labels' not in k else 'int32'
            f.create_dataset(k, data=v, compression='gzip', dtype=dtype)
        f.attrs['cutoff'] = args.cutoff
        f.attrs['extend_factor'] = args.extend_factor
            
    print("Generation Complete.")

if __name__ == '__main__':
    main()
