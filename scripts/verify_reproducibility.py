import os
import h5py
import numpy as np
import subprocess
import sys

def verify_reproducibility():
    print("Verifying reproducibility...")
    
    # Run 1
    file1 = "data/verify_run1.h5"
    if os.path.exists(file1): os.remove(file1)
    
    cmd1 = [
        "python", "scripts/extract_dynamics.py",
        "--output_file", file1,
        "--num_samples", "10",
        "--batch_size", "10"
    ]
    print(f"Executing Run 1: {' '.join(cmd1)}")
    subprocess.check_call(cmd1)
    
    # Run 2
    file2 = "data/verify_run2.h5"
    if os.path.exists(file2): os.remove(file2)
    
    cmd2 = [
        "python", "scripts/extract_dynamics.py",
        "--output_file", file2,
        "--num_samples", "10",
        "--batch_size", "10"
    ]
    print(f"Executing Run 2: {' '.join(cmd2)}")
    subprocess.check_call(cmd2)
    
    # Compare
    print("Comparing files...")
    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')
    
    keys = ['predictions', 'certainties', 'post_activations', 'synch_out']
    all_match = True
    
    for k in keys:
        d1 = f1[k][:]
        d2 = f2[k][:]
        
        if not np.array_equal(d1, d2):
            print(f"[FAIL] Mismatch in {k}")
            print(f"  Shape: {d1.shape}")
            print(f"  Max diff: {np.max(np.abs(d1 - d2))}")
            all_match = False
        else:
            print(f"[PASS] {k} matches exactly.")
            
    f1.close()
    f2.close()
    
    # Cleanup
    if all_match:
        os.remove(file1)
        os.remove(file2)
        print("\nReproducibility verified!")
    else:
        print("\nReproducibility FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    verify_reproducibility()
