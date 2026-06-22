import json
import os
import pandas as pd

DEFAULT_PATH = r"C:\Users\anandha.kumar\OneDrive - ClaySys Technologies\Desktop\Code Base\Global search\data\raw\evaluation_dataset.json"

def eva_data(file_path: str = DEFAULT_PATH) -> list:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation dataset file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    try:
        data = eva_data()
        print(f"Successfully loaded evaluation data. Number of samples: {len(data)}")
        # Show first 5 records as a DataFrame preview
        df = pd.DataFrame(data)
        print("\nPreview of the first 5 records:")
        print(df.head())
    except Exception as e:
        print(f"An error occurred: {e}")
