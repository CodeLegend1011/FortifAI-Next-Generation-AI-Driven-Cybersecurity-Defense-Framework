import numpy as np

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types - NUMPY 2.0 COMPATIBLE"""
    if obj is None:
        return None
    
    # Handle numpy boolean
    if isinstance(obj, (np.bool_, bool)) and hasattr(np, 'bool_'):
        return bool(obj)
    if type(obj).__name__ == 'bool_':
        return bool(obj)
    
    # Handle numpy integers
    if isinstance(obj, np.integer):
        return int(obj)
    if type(obj).__name__ in ('int_', 'int8', 'int16', 'int32', 'int64'):
        return int(obj)
    
    # Handle numpy floats - THIS IS THE KEY FIX
    if isinstance(obj, np.floating):
        return float(obj)
    if type(obj).__name__ in ('float_', 'float16', 'float32', 'float64'):
        return float(obj)
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle dicts recursively
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    
    # Handle lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    
    return obj
