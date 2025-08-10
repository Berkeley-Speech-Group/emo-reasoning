import torch
import os
import argparse
from tqdm import tqdm
from WavLM import WavLM, WavLMConfig
import torchaudio
import pandas as pd
import numpy as np
from CSER import CSERM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def average_features_per_second(features: torch.Tensor, step_size: int = 50) -> torch.Tensor:
    """
    Averages frame-level features into second-level features.
    WavLM outputs features at a rate of 50 frames per second.
    """
    batch_size, time_steps, feature_dim = features.shape
    padding_needed = (step_size - (time_steps % step_size)) % step_size
    if padding_needed > 0:
        padding = torch.zeros(batch_size, padding_needed, feature_dim, device=features.device)
        features = torch.cat([features, padding], dim=1)
    num_seconds = features.shape[1] // step_size
    return features.view(batch_size, num_seconds, step_size, -1).mean(dim=2)

def run_batch_inference(input_dir: str, output_dir: str, wavlm: torch.nn.Module, cser_model: torch.nn.Module, batch_size: int):
    """
    Performs batch inference on all .wav files in a directory, predicting VAD values
    and saving them to corresponding CSV files.

    Args:
        input_dir (str): Directory with .wav files.
        output_dir (str): Directory to save the output .csv files.
        wavlm (torch.nn.Module): The pre-trained WavLM model for feature extraction.
        cser_model (torch.nn.Module): The trained CSERM for VAD prediction.
        batch_size (int): Number of files to process per batch.
    """
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    if not audio_files:
        print(f"No .wav files found in {input_dir}. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    target_sample_rate = 16000
    
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Running inference"):
        batch_files = audio_files[i:i+batch_size]
        wav_forms, original_lengths = [], []
        
        for audio_file in batch_files:
            audio_path = os.path.join(input_dir, audio_file)
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] != 1: waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
            wav_forms.append(waveform.squeeze(0))
            original_lengths.append(waveform.shape[1])

        # Pad batch to the max length in the batch
        padded_wavs = torch.nn.utils.rnn.pad_sequence(wav_forms, batch_first=True, padding_value=0.0)
        padding_mask = (padded_wavs == 0.0)

        padded_wavs = padded_wavs.to(device)
        padding_mask = padding_mask.to(device)

        # Step 1: Extract features with WavLM
        with torch.no_grad():
            if wavlm.cfg.normalize:
                padded_wavs = torch.nn.functional.layer_norm(padded_wavs, padded_wavs.shape)
            features, _ = wavlm.extract_features(padded_wavs, padding_mask=padding_mask)

        # Step 2: Average features per second
        avg_features = average_features_per_second(features, step_size=50)

        # Step 3: Predict VAD values with CSERM
        with torch.no_grad():
            predictions = cser_model(avg_features)

        # Step 4: Save predictions to CSV, removing padding
        for j, audio_file in enumerate(batch_files):
            num_frames_original = original_lengths[j]
            num_seconds = -(-num_frames_original // target_sample_rate) # Ceiling division
            
            valid_predictions = predictions[j][:num_seconds].cpu().numpy()
            
            output_file = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.csv")
            pred_df = pd.DataFrame(valid_predictions, columns=['arousal', 'valence', 'dominance'])
            pred_df.to_csv(output_file, index=False)

def main(args):
    # Load WavLM model for feature extraction
    print("Loading WavLM model...")
    checkpoint_wavlm = torch.load(args.wavlm_path, map_location=device)
    cfg = WavLMConfig(checkpoint_wavlm['cfg'])
    wavlm = WavLM(cfg)
    wavlm.load_state_dict(checkpoint_wavlm['model'])
    wavlm.to(device)
    wavlm.eval()
    print("WavLM model loaded.")

    # Load trained CSER model for prediction
    print("Loading CSERM...")
    bilstm_params = {'hidden_size': 512, 'num_layers': 2}
    cser_model = CSERM(bilstm_params).to(device)
    checkpoint_cser = torch.load(args.cser_model_path, map_location=device)
    cser_model.load_state_dict(checkpoint_cser)
    cser_model.eval()
    print("CSERM loaded.")

    # Run inference on all subdirectories if specified, otherwise on the main directory
    if args.recursive:
        for root, _, files in os.walk(args.input_dir):
            if any(f.endswith('.wav') for f in files):
                relative_path = os.path.relpath(root, args.input_dir)
                current_output_dir = os.path.join(args.output_dir, relative_path)
                print(f"Processing directory: {root}")
                run_batch_inference(root, current_output_dir, wavlm, cser_model, args.batch_size)
    else:
        run_batch_inference(args.input_dir, args.output_dir, wavlm, cser_model, args.batch_size)

    print("Batch inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference to predict VAD values for audio files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .wav files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save .csv prediction files.')
    parser.add_argument('--wavlm_path', type=str, required=True, help='Path to the WavLM-Large.pt model.')
    parser.add_argument('--cser_model_path', type=str, required=True, help='Path to the trained CSERM checkpoint.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference.')
    parser.add_argument('--recursive', action='store_true', help='Process all subdirectories recursively.')
    
    args = parser.parse_args()
    main(args)