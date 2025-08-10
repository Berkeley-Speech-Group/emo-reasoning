import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

def process_single_annotator_file(file_path: str) -> pd.DataFrame:
    """
    Reads a single raw annotation file, skips the header, and averages
    ratings per second.

    Args:
        file_path (str): Path to the raw annotation text file.

    Returns:
        pd.DataFrame: A DataFrame with 'second' and 'rating' columns.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find the start of the data section, marked by '%%%%%%,%%%%%%'
    try:
        start_idx = next((i for i, line in enumerate(lines) if "%%%%%%,%%%%%%" in line)) + 1
    except StopIteration:
        print(f"Warning: Delimiter not found in {file_path}. Skipping file.")
        return pd.DataFrame()

    df = pd.read_csv(file_path, skiprows=start_idx, names=["second", "rating"])
    df["second"] = df["second"].astype(int)  # Group by integer second
    
    # Average ratings within each second
    return df.groupby("second")["rating"].mean().reset_index()

def process_dimension(conversation_id: str, dimension: str, annotations_dir: str) -> pd.DataFrame:
    """
    Processes all annotator files for a given emotion dimension (e.g., 'Arousal')
    and averages them into a single time-series.

    Args:
        conversation_id (str): The ID of the conversation (e.g., 'MSP-Conversation_0001').
        dimension (str): The emotion dimension ('Arousal', 'Valence', or 'Dominance').
        annotations_dir (str): The root directory of the raw annotations.

    Returns:
        pd.DataFrame: A DataFrame with averaged ratings for the dimension.
    """
    dim_path = os.path.join(annotations_dir, dimension)
    annotator_files = [os.path.join(dim_path, f) for f in os.listdir(dim_path) if f.startswith(conversation_id)]

    all_annotator_dfs = [process_single_annotator_file(p) for p in annotator_files]
    valid_dfs = [df for df in all_annotator_dfs if not df.empty]
    
    if not valid_dfs:
        return pd.DataFrame()
        
    # Average across all annotators for each second
    combined_df = pd.concat(valid_dfs).groupby("second")["rating"].mean().reset_index()
    combined_df.rename(columns={"rating": dimension.lower()}, inplace=True)
    return combined_df.set_index("second")


def main():
    parser = argparse.ArgumentParser(description="Process MSP-Conversation annotations into clean CSVs.")
    parser.add_argument('--segmented_audio_dir', type=str, required=True, help="Directory of segmented .wav files to determine which files to process.")
    parser.add_argument('--annotations_dir', type=str, required=True, help="Path to the root of the raw Annotations directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the processed .csv label files.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dimensions = ["Arousal", "Valence", "Dominance"]
    audio_files = [f for f in os.listdir(args.segmented_audio_dir) if f.endswith(".wav")]

    for audio_file in tqdm(audio_files, desc="Processing Annotations"):
        filename = os.path.splitext(audio_file)[0]
        # e.g., "MSP-Conversation_0001_1_1" -> "MSP-Conversation_0001"
        conversation_id = "_".join(filename.split("_")[:2])

        dimension_data = [process_dimension(conversation_id, dim, args.annotations_dir) for dim in dimensions]
        valid_dimension_data = [df for df in dimension_data if not df.empty]
        
        if not valid_dimension_data:
            print(f"Warning: No valid annotation data found for {filename}. Skipping.")
            continue
            
        # Merge all dimensions into a single DataFrame
        final_df = pd.concat(valid_dimension_data, axis=1, join='outer').reset_index()
        
        # Interpolate and fill missing values to create a dense time-series
        max_seconds = final_df["second"].max()
        full_time_index = pd.DataFrame(np.arange(0, max_seconds + 1), columns=["second"])
        final_df = full_time_index.merge(final_df, on="second", how="left")
        final_df = final_df.sort_values("second").fillna(method='ffill').fillna(method='bfill')
        
        output_csv_path = os.path.join(args.output_dir, f"{filename}.csv")
        final_df.to_csv(output_csv_path, index=False, columns=["second", "arousal", "valence", "dominance"])

    print("Annotation processing complete.")

if __name__ == "__main__":
    main()