import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os

def load_embeddings(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    elif ext == ".parquet":
        table = pq.read_table(path)
        # Assume the first column or a column named 'embedding' contains the vectors
        if "embedding" in table.column_names:
            col = table.column("embedding")
        else:
            col = table.column(0)
        
        # Convert to numpy. Parquet embeddings are often lists of floats.
        data = col.to_numpy()
        if isinstance(data[0], (list, np.ndarray)):
            return np.stack(data).astype(np.float32)
        return data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
