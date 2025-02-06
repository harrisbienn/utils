from pathlib import Path
import pandas as pd

def load_csv(file_path):
    """Load a CSV file into a Pandas DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    print(f"✅ Loaded file: {file_path}")
    return pd.read_csv(file_path)

def save_csv(df, output_path):
    """Save a DataFrame to CSV."""
    output_path = Path(output_path)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved file: {output_path}")

def list_files(directory, extension="*"):
    """List all files in a directory with a given extension."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"❌ Directory not found: {directory}")
    return list(directory.glob(f"*.{extension}"))
    
# Define a function to handle both range, single values, and text entries
def split_range(range_str):
    if pd.isnull(range_str):
        return (None, None)
    
    # Split on hyphen if present, otherwise assume a single value
    try:
        if '-' in range_str:
            min_val, max_val = map(str.strip, range_str.split('-'))
        else:
            min_val = max_val = range_str.strip()

        # Attempt to convert string parts to integers
        return int(min_val), int(max_val)
    except ValueError:
        # Return None for non-integer or unconvertible strings
        return (None, None)
        
# Function to convert Folder to Hyperlink
def make_hyperlink(domain,path):
    # Skip None values
    if path is None:
        return "None"  # or simply return None if you prefer
    else:
        url = domain + path 
        #return f'<a href="{url}" target="_blank">{url}</a>'
        return f'<a href="{url}">Linked Data</a>'