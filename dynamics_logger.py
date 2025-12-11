import h5py
import numpy as np
import os

class DynamicsLogger:
    def __init__(self, output_path, expected_samples=1000):
        self.output_path = output_path
        self.expected_samples = expected_samples
        self.file = None
        self.current_idx = 0
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize file
        self.file = h5py.File(self.output_path, 'w')
        
        # We will initialize datasets on the first batch when we know dimensions
        self.datasets = {}

    def _init_datasets(self, predictions, certainties, synch, pre_acts, post_acts):
        # inputs are typically numpy arrays from a batch
        # dimensions: (Batch, ...)
        
        # synch is a tuple (synch_out, synch_action) or just synch_out?
        # Based on run.py: predictions, certainties, (synch_out, synch_action), pre, post, attn
        # We need to handle this structure.
        
        synch_out, synch_action = synch
        
        # Create datasets
        # Use simple compression 'gzip'
        
        # 1. Post-activations (z^t)
        # Shape: (Total, Iterations, Features) or similar.
        # pre_acts shape from run.py likely: (B, Iterations, Features) ?? 
        # Actually in ctm.py: pre_activations_tracking is list of (B, ...). 
        # run.py converts to np.array -> (Iterations, B, Features) usually if stacked?
        # Let's assume the incoming data is (B, Iterations, ...) for standard logging.
        # If it comes as (Iterations, B, ...), we might want to transpose for dataset consistency (Sample, ...)
        
        # Let's inspect shapes in the `log_batch` method dynamically.
        pass

    def log_batch(self, predictions, certainties, synch, pre_acts, post_acts, metadata=None):
        """
        Logs a batch of data.
        synch: tuple (synch_out, synch_action)
        """
        synch_out, synch_action = synch
        
        # Convert all to numpy if they aren't
        def to_np(x):
            return x.detach().cpu().numpy() if hasattr(x, 'detach') else x
            
        predictions = to_np(predictions)
        certainties = to_np(certainties)
        synch_out = to_np(synch_out)
        synch_action = to_np(synch_action)
        pre_acts = to_np(pre_acts)
        post_acts = to_np(post_acts)
        
        # In run.py/ctm.py, tracking returns np.array(list_of_tensors).
        # List len = Iterations. Tensor shape = (B, ...).
        # So np.array shape = (Iterations, B, ...).
        # We want (B, Iterations, ...) for storage usually.
        
        # Transpose helper
        def fix_dims(arr):
            # If shape is (Iterations, Batch, ...), swap to (Batch, Iterations, ...)
            # Assuming dim 0 is Iterations and dim 1 is Batch if len > 1
            if len(arr.shape) >= 2:
                return np.swapaxes(arr, 0, 1)
            return arr

        # predictions/certainties are already (B, ..., Iterations) in ctm.py!
        # wait, ctm.py: predictions[..., stepi] = current_prediction
        # so output is (B, OutDim, Iterations). 
        # Tranpose to (B, Iterations, OutDim) for consistency?
        predictions = np.moveaxis(predictions, -1, 1) # -> (B, Iterations, OutDim)
        certainties = np.moveaxis(certainties, -1, 1) # -> (B, Iterations, 2)
        
        # pre_acts, post_acts, synch: (Iterations, B, ...) -> (B, Iterations, ...)
        pre_acts = fix_dims(pre_acts)
        post_acts = fix_dims(post_acts)
        synch_out = fix_dims(synch_out)
        synch_action = fix_dims(synch_action)
        
        batch_size = predictions.shape[0]
        
        # Initialize datasets if needed
        if not self.datasets:
            self.datasets['predictions'] = self.file.create_dataset('predictions', shape=(self.expected_samples,) + predictions.shape[1:], dtype='float16', compression='gzip')
            self.datasets['certainties'] = self.file.create_dataset('certainties', shape=(self.expected_samples,) + certainties.shape[1:], dtype='float16', compression='gzip')
            self.datasets['post_activations'] = self.file.create_dataset('post_activations', shape=(self.expected_samples,) + post_acts.shape[1:], dtype='float16', compression='gzip')
            self.datasets['synch_out'] = self.file.create_dataset('synch_out', shape=(self.expected_samples,) + synch_out.shape[1:], dtype='float16', compression='gzip')
            # Add metadata datasets
            # metadata is dict {key: list_of_values}
            if metadata:
                for k, v in metadata.items():
                    # Assuming scalars for now (int/float)
                    if isinstance(v[0], int) or isinstance(v[0], np.integer):
                         self.datasets[k] = self.file.create_dataset(k, shape=(self.expected_samples,), dtype='int32')
                    else:
                         self.datasets[k] = self.file.create_dataset(k, shape=(self.expected_samples,), dtype='float32')

        # Write data
        end_idx = self.current_idx + batch_size
        
        self.datasets['predictions'][self.current_idx:end_idx] = predictions
        self.datasets['certainties'][self.current_idx:end_idx] = certainties
        self.datasets['post_activations'][self.current_idx:end_idx] = post_acts
        self.datasets['synch_out'][self.current_idx:end_idx] = synch_out
        
        if metadata:
            for k, v in metadata.items():
                self.datasets[k][self.current_idx:end_idx] = v
                
        self.current_idx = end_idx
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
