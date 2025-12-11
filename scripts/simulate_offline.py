import torch
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
import h5py

# Path setup
sys.path.append(os.getcwd())
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tasks.mazes.analysis.run import _load_ctm_model
from data.custom_datasets import MazeImageFolder
from analysis.attractor_metrics import calc_convergence_rate

# Create custom CTM forward pass to support input intervention
# We can't easily hook into the loop of the existing CTM class without modifying it or subclassing?
# or we can copy/paste the forward loop here for the experiment.
# Copying is safer to avoid modifying core models.

def offline_forward(model, x, cutoff_step=10, replacement_val=0.0):
    """
    Runs CTM trace but replaces input 'kv' features after cutoff_step.
    """
    B = x.size(0)
    device = x.device
    
    # 1. Featurize Input
    # Only useful for first part
    kv_original = model.compute_features(x)
    
    # 2. Create "Offline" Input (Zero/Noise)
    # Zeroing out the KV might be the right way to simulate "no input" or "darkness"
    if replacement_val == 0.0:
        kv_offline = torch.zeros_like(kv_original)
    else:
        # random noise
        kv_offline = torch.randn_like(kv_original)
    
    # Init Recurrent State (Same as CTM.forward)
    state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1)
    activated_state = model.start_activated_state.unsqueeze(0).expand(B, -1)
    
    # Storage
    predictions = []
    post_activations = []
    
    decay_alpha_action, decay_beta_action = None, None
    r_action, r_out = torch.exp(-model.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-model.decay_params_out).unsqueeze(0).repeat(B, 1)
    
    _, decay_alpha_out, decay_beta_out = model.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
    
    for stepi in range(model.iterations):
        # *** INTERVENTION ***
        # If we are past cutoff, switch to offline KV
        if stepi >= cutoff_step:
            current_kv = kv_offline
        else:
            current_kv = kv_original
            
        # Standard Loop logic
        synchronisation_action, decay_alpha_action, decay_beta_action = model.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')
        
        q = model.q_proj(synchronisation_action).unsqueeze(1)
        # Use current_kv
        attn_out, _ = model.attention(q, current_kv, current_kv, average_attn_weights=False, need_weights=False)
        attn_out = attn_out.squeeze(1)
        
        pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
        
        state = model.synapses(pre_synapse_input)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
        
        activated_state = model.trace_processor(state_trace)
        
        synchronisation_out, decay_alpha_out, decay_beta_out = model.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')
        
        current_prediction = model.output_projector(synchronisation_out)
        
        predictions.append(current_prediction)
        post_activations.append(activated_state)
        
    return torch.stack(predictions, dim=1), torch.stack(post_activations, dim=1) # (B, T, ...)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt')
    parser.add_argument('--data_root', default='data/mazes/large/test')
    parser.add_argument('--output', default='data/offline_simulation.h5')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--cutoff', type=int, default=25) # Cutoff after 25 ticks (1/3rd of 75)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Model & Dataset
    model = _load_ctm_model(args.checkpoint, device)
    
    dataset = MazeImageFolder(args.data_root, which_set='test', maze_route_length=50, expand_range=False, trunc=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False)
    
    all_preds_std = []
    all_states_std = []
    all_preds_off = []
    all_states_off = []
    
    count = 0
    print(f"Simulating Offline Regime (Cutoff={args.cutoff})...")
    
    with torch.no_grad():
        for x, y in tqdm(loader):
            if count >= args.num_samples: break
            x = x.to(device)
            B = x.size(0)
            
            # 1. Standard Run
            # predictions, certainties, _, _, post_acts, _ = model(x, track=True)
            # Use our loop for consistency? Or standard forward? 
            # run.py forward returns (B, Out, T), (B, 2, T)
            # offline_forward returns (B, T, Out) !! Need to match dims.
            # Let's run offline_forward with cutoff > iterations for "Standard"
            
            p_std, s_std = offline_forward(model, x, cutoff_step=999)
            p_off, s_off = offline_forward(model, x, cutoff_step=args.cutoff)
            
            all_preds_std.append(p_std.cpu().numpy())
            all_states_std.append(s_std.cpu().numpy())
            all_preds_off.append(p_off.cpu().numpy())
            all_states_off.append(s_off.cpu().numpy())
            
            count += B
            
    # Concatenate
    all_preds_std = np.concatenate(all_preds_std, axis=0)[:args.num_samples]
    all_states_std = np.concatenate(all_states_std, axis=0)[:args.num_samples]
    all_preds_off = np.concatenate(all_preds_off, axis=0)[:args.num_samples]
    all_states_off = np.concatenate(all_states_off, axis=0)[:args.num_samples]
    
    # Save results
    print(f"Saving to {args.output}")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('preds_std', data=all_preds_std, compression='gzip', dtype='float16')
        f.create_dataset('states_std', data=all_states_std, compression='gzip', dtype='float16')
        f.create_dataset('preds_off', data=all_preds_off, compression='gzip', dtype='float16')
        f.create_dataset('states_off', data=all_states_off, compression='gzip', dtype='float16')
        f.attrs['cutoff'] = args.cutoff
        
    print("Offline Simulation Complete.")

if __name__ == '__main__':
    main()
