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
from models.baselines import SimpleRNN
from tasks.mazes.analysis.run import _load_ctm_model # We'll just instantiate directly for untrained
from models.ctm import ContinuousThoughtMachine
from data.custom_datasets import MazeImageFolder

def get_untrained_ctm(device):
    # Match parameters from verified checkpoint
    # D=2048, T=75, M=25, etc
    # We can get these from the loaded checkpoint's args too to be safe
    # But for untrained, manual instantiation is fine if consistent
    model = ContinuousThoughtMachine(
        iterations=75,
        d_model=2048,
        d_input=128,
        heads=4,
        n_synch_out=25,
        n_synch_action=25,
        synapse_depth=1,
        memory_length=1500, # Large memory from checkpoint name/config usually
        deep_nlms=True,
        memory_hidden_dims=2048,
        out_dims=5, # Mazes have 5 actions? Check task info
        # Check defaults from verify_setup or run.py
    ).to(device)
    return model

def get_untrained_rnn(device):
    model = SimpleRNN(
        d_input=128,
        d_model=2048,
        out_dims=5,
        iterations=75
    ).to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/mazes/large/test')
    parser.add_argument('--output', default='data/baseline_trajectories.h5')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--cutoff', type=int, default=25)
    parser.add_argument('--extend_factor', type=float, default=2.0)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Init Models
    print("Initializing Untrained Models...")
    # Need to verify params - simpler to load trained and re-init weights?
    # Or just use reasonable defaults. The dynamics structure check depends mostly on architecture.
    # Let's try to infer from checkpoint file just for parameters but NOT load state dict
    try:
        # Hack: Load checkpoint to get args, but don't apply state dict
        cp = torch.load('checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt', map_location='cpu', weights_only=False)
        args_model = cp['args']
        
        # Instantiate CTM
        # Logic copied from _load_ctm_model but without state dict load
        # Need to handle missing args defaults
        if not hasattr(args_model, 'neuron_select_type'): args_model.neuron_select_type = 'first-last'
        if not hasattr(args_model, 'n_random_pairing_self'): args_model.n_random_pairing_self = 0
        
        prediction_reshaper = [args_model.out_dims // 5, 5] if hasattr(args_model, 'out_dims') else None

        ctm_untrained = ContinuousThoughtMachine(
            iterations=args_model.iterations,
            d_model=args_model.d_model,
            d_input=args_model.d_input,
            heads=args_model.heads,
            n_synch_out=args_model.n_synch_out,
            n_synch_action=args_model.n_synch_action,
            synapse_depth=args_model.synapse_depth,
            memory_length=args_model.memory_length,
            deep_nlms=args_model.deep_memory,
            memory_hidden_dims=args_model.memory_hidden_dims,
            do_layernorm_nlm=args_model.do_normalisation,
            backbone_type=getattr(args_model, 'backbone_type', 'resnet18'), # simple default fallback
            positional_embedding_type=args_model.positional_embedding_type,
            out_dims=args_model.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=0,
            neuron_select_type=args_model.neuron_select_type,
            n_random_pairing_self=args_model.n_random_pairing_self,
        ).to(device)
        
        # Init RNN
        rnn_untrained = SimpleRNN(
             d_input=args_model.d_input,
             d_model=args_model.d_model,
             out_dims=args_model.out_dims,
             iterations=args_model.iterations
        ).to(device)
        
        print("Models Initialized from Config.")
        
    except Exception as e:
        print(f"Failed to init from config: {e}. Using defaults.")
        return

    # Data Loader
    dataset = MazeImageFolder(args.data_root, which_set='test', maze_route_length=50, expand_range=False, trunc=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False)
    
    # Wrappers
    ctm_wrapper = OfflineCTMWrapper(ctm_untrained)
    # RNN needs custom wrapper or just simple loop since it has no "Offline" features really
    # But we want to simulate input cutoff.
    # OfflineCTMWrapper calls `compute_features` which SimpleRNN has. 
    # It also relies on `compute_synchronisation` etc which RNN lacks.
    # So we need a simple OfflineRNNWrapper or logic here.
    
    # Store
    data_store = {'ctm_untrained': [], 'rnn_untrained': []}
    
    processed = 0
    print("Generating Baseline Data...")
    
    with torch.no_grad():
        for x, _ in tqdm(loader):
            if processed >= args.num_samples: break
            x = x.to(device)
            B = x.size(0)
            
            # 1. Untrained CTM (Mask Variant)
            _, s_ctm = ctm_wrapper.forward_offline(x, variant='mask', cutoff=args.cutoff, extend_factor=args.extend_factor)
            data_store['ctm_untrained'].append(s_ctm.cpu().numpy())
            
            # 2. Untrained RNN (Manual Cutoff logic)
            # Featurize
            kv_orig = rnn_untrained.compute_features(x)
            kv_zero = torch.zeros_like(kv_orig)
            
            # RNN Loop
            # We need to manually drive RNN to support cutoff
            # SimpleRNN forward doesn't support cutoff.
            # Let's just do it manually here roughly
            
            # Init State
            rnn_state = rnn_untrained.start_state.unsqueeze(0).expand(B, -1)
            states_rnn_list = []
            
            rnn_inputs_orig = kv_orig.mean(dim=1)
            rnn_inputs_zero = kv_zero.mean(dim=1)
            
            total_steps = int(args_model.iterations * args.extend_factor)
            
            for i in range(total_steps):
                curr_in = rnn_inputs_zero if i >= args.cutoff else rnn_inputs_orig
                rnn_state = rnn_untrained.lstm_cell(curr_in, rnn_state)
                states_rnn_list.append(rnn_state)
                
            s_rnn = torch.stack(states_rnn_list, dim=1) # (B, T, D)
            data_store['rnn_untrained'].append(s_rnn.cpu().numpy())
            
            processed += B
            
    # Save
    with h5py.File(args.output, 'w') as f:
        for k, v in data_store.items():
            data_cat = np.concatenate(v, axis=0)[:args.num_samples]
            f.create_dataset(k, data=data_cat, compression='gzip', dtype='float16')
            
    print(f"Baselines Generated. Saved to {args.output}")

if __name__ == '__main__':
    main()
