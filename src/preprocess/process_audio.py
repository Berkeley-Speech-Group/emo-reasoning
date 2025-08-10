import os
import argparse
from pydub import AudioSegment
from tqdm import tqdm

def read_conversation_offsets(conversations_file: str) -> dict:
    """
    Reads the conversation start time offsets from the conversations.txt file.
    This is needed to align segment timestamps with the audio file's start.

    Args:
        conversations_file (str): Path to conversations.txt.

    Returns:
        dict: A mapping from conversation_id to its start offset in seconds.
    """
    offsets = {}
    with open(conversations_file, "r") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) == 3:
                conversation_id, start_offset, _ = parts
                offsets[conversation_id] = float(start_offset)
    return offsets

def segment_all_conversations(base_audio_path: str, parts_file: str, offsets: dict, output_path: str):
    """
    Segments full conversation audio files into smaller turn-level chunks based on
    the timing information in conversation_parts.txt.

    Args:
        base_audio_path (str): Path to the directory with full .wav files.
        parts_file (str): Path to conversation_parts.txt.
        offsets (dict): A dictionary of conversation start offsets.
        output_path (str): Directory to save the segmented .wav files.
    """
    segments_to_process = []
    with open(parts_file, "r") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) == 3:
                filename, start_time, end_time = parts
                # Extract conversation ID, e.g., "MSP-Conversation_0001"
                conversation_id = "_".join(filename.split("_")[:2])
                segments_to_process.append((conversation_id, filename, float(start_time), float(end_time)))

    print(f"Found {len(segments_to_process)} audio segments to process.")
    for conversation_id, filename, start_time, end_time in tqdm(segments_to_process, desc="Segmenting Audio"):
        start_offset = offsets.get(conversation_id, 0.0)
        if start_offset == 0.0:
            print(f"Warning: No offset found for {conversation_id}, using original times.")

        # Adjust segment times by the conversation's start offset
        adjusted_start_ms = (start_time - start_offset) * 1000
        adjusted_end_ms = (end_time - start_offset) * 1000

        audio_file = os.path.join(base_audio_path, f"{conversation_id}.wav")
        if os.path.exists(audio_file):
            try:
                audio = AudioSegment.from_wav(audio_file)
                segment_audio = audio[int(adjusted_start_ms):int(adjusted_end_ms)]
                output_file = os.path.join(output_path, f"{filename}.wav")
                segment_audio.export(output_file, format="wav")
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
        else:
            print(f"Warning: Missing audio file {audio_file}")


def main():
    parser = argparse.ArgumentParser(description="Segment MSP-Conversation audio files into turns.")
    parser.add_argument('--base_audio_dir', type=str, required=True, help="Path to the directory containing full conversation .wav files (e.g., .../Audio/Audio).")
    parser.add_argument('--time_labels_dir', type=str, required=True, help="Path to the directory containing Time_Labels files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the segmented audio files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    conversations_file = os.path.join(args.time_labels_dir, "conversations.txt")
    conversation_parts_file = os.path.join(args.time_labels_dir, "conversation_parts.txt")

    print("Reading conversation offsets...")
    offsets = read_conversation_offsets(conversations_file)
    
    print("Starting audio segmentation...")
    segment_all_conversations(args.base_audio_dir, conversation_parts_file, offsets, args.output_dir)
    print("Audio segmentation complete.")

if __name__ == "__main__":
    main()