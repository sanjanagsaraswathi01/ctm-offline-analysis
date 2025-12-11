import torch
import torch.nn as nn
import numpy as np

class OfflineCTMWrapper:
    def __init__(self, model):
        self.model = model
        
    def forward_offline(self, x, variant='mask', cutoff=25, extend_factor=2):
        """
        Runs CTM with intervention at `cutoff` step.
        variant: 'mask' (zeros), 'noise' (gaussian), 'none' (standard)
        extend_factor: Multiplier for original iterations (e.g., 2x duration)
        """
        B = x.size(0)
        device = x.device
        
        # 1. Featurize Input (Standard)
        kv_original = self.model.compute_features(x)
        
        # 2. Prepare Intervention Components
        if variant == 'mask':
            kv_intervention = torch.zeros_like(kv_original)
        elif variant == 'noise':
            kv_intervention = torch.randn_like(kv_original)
        else: # 'none' or 'control'
            kv_intervention = kv_original
            
        # 3. Initialize State
        state_trace = self.model.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.model.start_activated_state.unsqueeze(0).expand(B, -1)
        
        predictions = []
        post_activations = []
        
        # Determine total steps
        total_steps = int(self.model.iterations * extend_factor)
        
        # Synchronization Params
        decay_alpha_action, decay_beta_action = None, None
        r_action = torch.exp(-self.model.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.model.decay_params_out).unsqueeze(0).repeat(B, 1)
        
        _, decay_alpha_out, decay_beta_out = self.model.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )
        
        # 4. Run Loop
        for stepi in range(total_steps):
            # Select Input
            if stepi >= cutoff:
                current_kv = kv_intervention
            else:
                current_kv = kv_original
                
            # --- Standard CTM Step Logic ---
            # 1. Synch Action
            synchronisation_action, decay_alpha_action, decay_beta_action = self.model.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )
            
            # 2. Attention
            q = self.model.q_proj(synchronisation_action).unsqueeze(1)
            # Use current_kv (Modified or Original)
            attn_out, _ = self.model.attention(q, current_kv, current_kv, need_weights=False)
            attn_out = attn_out.squeeze(1)
            
            # 3. Update State
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
            state = self.model.synapses(pre_synapse_input)
            
            # Update Trace (Slide window)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            
            # 4. Activate
            activated_state = self.model.trace_processor(state_trace)
            
            # 5. Synch Out & Predict
            synchronisation_out, decay_alpha_out, decay_beta_out = self.model.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )
            current_prediction = self.model.output_projector(synchronisation_out)
            
            # Store
            predictions.append(current_prediction)
            post_activations.append(activated_state)
            
        return torch.stack(predictions, dim=1), torch.stack(post_activations, dim=1)
