import dis
import pickle
import tempfile
import os
import subprocess
import time
import polars as pl
import pandas as pd
import numpy as np

def show_global(func):
    # Get the bytecode instructions of the function
    bytecode = dis.Bytecode(func)

    # Extract and print only the LOAD_GLOBAL instructions
    for instr in bytecode:
        if instr.opname == "LOAD_GLOBAL":
            print(f"{instr.opname}: {instr.argval}")

def ensure_dir(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def write_pickle_to_local(obj, file_path):
    """Write pickle file to local directory"""
    ensure_dir(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle_from_local(file_path):
    """Read pickle file from local directory"""
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
            
            # Verify data consistency after loading
            if hasattr(obj, 'result') and isinstance(obj.result, pl.DataFrame):
                print(f"Loaded result shape: {obj.result.shape}")
            
            return obj
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading pickle file: {e}")

def view_data(data):
    """
    Open data in default spreadsheet application
    
    Parameters:
    -----------
    data : pandas.DataFrame, numpy.ndarray, or polars.DataFrame
        Data to be viewed
    """
    # Convert input to polars DataFrame
    if isinstance(data, pd.DataFrame):
        df = pl.from_pandas(data)
    elif isinstance(data, np.ndarray):
        df = pl.from_numpy(data)
    elif isinstance(data, pl.DataFrame):
        df = data
    else:
        raise TypeError("Input must be pandas DataFrame, numpy array or polars DataFrame")
    
    # Create temp file with .csv extension
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    temp_path = temp.name
    temp.close()
    
    try:
        # Write DataFrame to CSV
        df.write_csv(temp_path)
        
        # Open CSV with default application
        if os.name == 'nt':  # Windows
            os.startfile(temp_path)
        elif os.name == 'posix':  # macOS/Linux
            subprocess.call(('open', temp_path))
            
        # Give Excel time to open the file before deleting
        time.sleep(5)
            
    except Exception as e:
        print(f"Error saving/opening CSV: {e}")
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass  # File may already be deleted
