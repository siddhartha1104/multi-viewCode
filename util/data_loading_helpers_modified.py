import numpy as np

def load_matlab_string(ref):
    """
    Extracts string content from an HDF5 object reference (used in .mat files saved with MATLAB v7.3).
    Handles both ASCII (uint16) and object-based encodings.
    """
    if isinstance(ref, np.ndarray):
        # Sometimes it's already a numpy array of ASCII codes
        return ''.join(chr(int(c)) for c in ref)
    
    # For actual HDF5 datasets
    try:
        return ''.join(chr(int(c[0])) for c in ref[:])
    except Exception:
        return str(ref[:])
