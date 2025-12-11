import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    A minimal RNN baseline with similar capacity/architecture to CTM.
    Uses CTM-like backbone but replaces the "Thought" process with a standard GRU.
    """
    def __init__(self, d_input, d_model, out_dims, iterations):
        super().__init__()
        self.d_model = d_model
        self.iterations = iterations
        
        # Backbone matching CTM's lazy structure
        self.backbone = nn.Sequential(
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.kv_proj = nn.Sequential(nn.LazyLinear(d_input), nn.LayerNorm(d_input))
        
        # Recurrent Core
        self.lstm_cell = nn.GRUCell(d_input, d_model) # Input is featured KV mean?
        
        # Output Projector
        self.output_projector = nn.Linear(d_model, out_dims)
        
        # Initialization match
        self.register_parameter('start_state', nn.Parameter(torch.zeros(d_model)))
        
    def compute_features(self, x):
        input_features = self.backbone(x)
        kv = self.kv_proj(input_features.flatten(2).transpose(1, 2))
        return kv

    def forward(self, x, track=False):
        # We need to match CTM signature for scripts to run
        B = x.size(0)
        device = x.device
        
        kv = self.compute_features(x)
        # Flatten KV for RNN input? 
        # CTM uses attention over KV. RNN usually takes vector.
        # Let's take mean over spatial dims to get vector input for RNN
        rnn_input = kv.mean(dim=1) # (B, d_input)
        
        state = self.start_state.unsqueeze(0).expand(B, -1)
        
        predictions = []
        states = []
        
        # Consistent output format with CTM:
        # predictions, certainties(dummy), _, _, post_acts(state), _
        
        for step in range(self.iterations):
            state = self.lstm_cell(rnn_input, state)
            pred = self.output_projector(state)
            
            predictions.append(pred)
            states.append(state)
            
        predictions = torch.stack(predictions, dim=-1) # (B, Out, T)
        states = torch.stack(states, dim=1) # (B, T, D)
        
        # Dummy Certainty (B, 2, T)
        certainties = torch.zeros(B, 2, self.iterations, device=device)
        
        # Matches CTM return signature for track=True
        # predictions, certainties, (synch), pre, post, attn
        if track:
             return predictions, certainties, (None, None), None, states, None
             
        return predictions, certainties, None
