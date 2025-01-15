import dis
import pickle
import tempfile
import os
import subprocess
import time
import polars as pl
import pandas as pd
import numpy as np

import boto3
from botocore.exceptions import NoCredentialsError

session = boto3.Session(profile_name="football-markov")
s3 = session.client("s3")


def show_global(func):
    # Get the bytecode instructions of the function
    bytecode = dis.Bytecode(func)

    # Extract and print only the LOAD_GLOBAL instructions
    for instr in bytecode:
        if instr.opname == "LOAD_GLOBAL":
            print(f"{instr.opname}: {instr.argval}")


def folder_exists(bucket, folder):
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=folder, Delimiter="/")
        for content in response.get("Contents", []):
            if content["Key"] == folder:
                return True
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def create_s3_folder_if_not_exists(bucket, folder):
    if not folder_exists(bucket=bucket, folder=folder):
        try:
            # Create an empty object with the folder name
            s3.put_object(Bucket=bucket, Key=folder)
            print(f"Successfully created folder s3://{bucket}/{folder}")
        except NoCredentialsError:
            print("Credentials not available")
    else:
        print(f"Folder s3://{bucket}/{folder} already exists")


def write_pickle_to_s3(obj, bucket, prefix, file_name):
    try:
        # Pickle the object
        pickled_data = pickle.dumps(obj)

        # Form the full key
        key = f"{prefix}{file_name}"

        # Upload the pickled object to S3
        s3.put_object(Bucket=bucket, Key=key, Body=pickled_data)
        print(f"Successfully uploaded pickled object to s3://{bucket}/{key}")
    except NoCredentialsError:
        print("Credentials not available")


def read_pickle_from_s3(bucket, prefix, file_name):
    try:
        # Form the full key
        key = f"{prefix}{file_name}"

        # Download the pickled object from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        pickled_data = response["Body"].read()

        # Unpickle the object
        obj = pickle.loads(pickled_data)
        print(f"Successfully read pickled object from s3://{bucket}/{key}")
        return obj
    except NoCredentialsError:
        print("Credentials not available")

def br(data) -> None:
    """
    Convert pandas DataFrame, polars DataFrame, or numpy array to polars DataFrame,
    save to temporary CSV file and open it.
    
    Args:
        data: A pandas DataFrame, polars DataFrame, or numpy array to convert, save and open
    """

    # Convert input to polars DataFrame if needed
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
            pass # File may already be deleted
