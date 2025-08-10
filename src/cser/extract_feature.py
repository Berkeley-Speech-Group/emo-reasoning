import torch
import os
import argparse
from tqdm import tqdm
from wavlm.WavLM import WavLM, WavLMConfig
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def average_features_per_second(features: torch.Tensor, step_size: int = 50) -> torch.Tensor:
    """
    Averages frame-level features into second-level features.
    WavLM outputs features at a rate of 50 frames per second.

    Args:
        features (torch.Tensor): Raw features from WavLM. Shape: [b, t, 1024]
        step_size (int): Number of frames per second (default is 50 for WavLM).

    Returns:
        torch.Tensor: Averaged features. Shape: [b, t/50, 1024]
    """
    batch_size, time_steps, feature_dim = features.shape
    
    # Pad features to be divisible by step_size
    padding_needed = (step_size - (time_steps % step_size)) % step_size
    if padding_needed > 0:
        padding = torch.zeros(batch_size, padding_needed, feature_dim, device=features.device)
        features = torch.cat([features, padding], dim=1)
        
    num_seconds = features.shape[1] // step_size
    
    # Reshape and average
    avg_features = features.view(batch_size, num_seconds, step_size, -1).mean(dim=2)
    
    return avg_features

def extract_and_save_features(audio_dir: str, output_dir: str, wavlm: nn.Module, batch_size: int):
    """
    Extracts features from audio files using WavLM, averages them per second,
    and saves them to disk. Skips files that have already been processed.

    Args:
        audio_dir (str): Directory containing .wav audio files.
        output_dir (str): Directory to save the extracted features (.pt files).
        wavlm (nn.Module): The pre-trained WavLM model.
        batch_size (int): Number of audio files to process in a single batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a list of files to process by excluding already processed ones
    existing_features = set(os.listdir(output_dir))
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') and f.replace('.wav', '_features.pt') not in existing_features]

    if not audio_files:
        print("All audio files have already been processed. Nothing to do.")
        return
        
    # Process audio files in batches
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Extracting features"):
        batch_files = audio_files[i:i+batch_size]
        wav_inputs = []
        
        for audio_file in batch_files:
            audio_path = os.path.join(audio_dir, audio_file)
            waveform, sample_rate = torchaudio.load(audio_path)

            # Ensure single-channel (mono) audio
            if waveform.shape[0] != 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            wav_inputs.append(waveform.squeeze(0))

        # Pad all waveforms in the batch to the same length
        padded_wavs = torch.nn.utils.rnn.pad_sequence(wav_inputs, batch_first=True, padding_value=0.0)
        padding_mask = (padded_wavs == 0.0)

        padded_wavs = padded_wavs.to(device)
        padding_mask = padding_mask.to(device)

        # Extract features using WavLM
        with torch.no_grad():
            if wavlm.cfg.normalize:
                padded_wavs = torch.nn.functional.layer_norm(padded_wavs, padded_wavs.shape)
            features, _ = wavlm.extract_features(padded_wavs, padding_mask=padding_mask)

        # Average features per second
        avg_features = average_features_per_second(features, step_size=50)        

        # Save the features for each file in the batch
        for j, audio_file in enumerate(batch_files):
            # Detach from graph and move to CPU before saving
            feature_to_save = avg_features[j].cpu()
            feature_save_path = os.path.join(output_dir, f"{audio_file.replace('.wav', '_features.pt')}")
            torch.save(feature_to_save, feature_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WavLM features from audio files.")
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted features.')
    parser.add_argument('--wavlm_path', type=str, required=True, help='Path to the WavLM-Large.pt model file.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for feature extraction.')
    
    args = parser.parse_args()

    # Load WavLM model
    print("Loading WavLM model...")
    checkpoint = torch.load(args.wavlm_path, map_location=device)
    cfg = WavLMConfig(checkpoint['cfg'])
    wavlm = WavLM(cfg)
    wavlm.load_state_dict(checkpoint['model'])
    wavlm.to(device)
    wavlm.eval()
    print("WavLM model loaded.")

    extract_and_save_features(args.audio_dir, args.output_dir, wavlm, args.batch_size)
    print("Feature extraction complete.")