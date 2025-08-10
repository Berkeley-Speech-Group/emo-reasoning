import numpy as np
from fastdtw import fastdtw
import pandas as pd
import os
from glob import glob
import argparse
from typing import List, Tuple

# --- Metric Definitions ---

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Calculates the Dynamic Time Warping (DTW) distance between two 1D sequences."""
    dist, _ = fastdtw(seq1, seq2)
    return dist

def ecs(user_seq: np.ndarray, agent_seq: np.ndarray, K: float = 5.0) -> float:
    """
    Emotional Contagion Score (ECS) for a single turn.
    Measures the similarity between user and agent emotional trajectories (V, A).
    A higher score indicates the agent's emotion is more aligned with the user's.

    Args:
        user_seq (np.ndarray): User VAD sequence, shape [T_u, 3].
        agent_seq (np.ndarray): Agent VAD sequence, shape [T_a, 3].
        K (float): A scaling constant.

    Returns:
        float: The ECS score.
    """
    V_u, A_u, _ = user_seq[:, 0], user_seq[:, 1], user_seq[:, 2]
    V_a, A_a, _ = agent_seq[:, 0], agent_seq[:, 1], agent_seq[:, 2]

    dtw_v = dtw_distance(V_u, V_a)
    dtw_a = dtw_distance(A_u, A_a)
    
    max_len = max(len(V_u), len(A_u), len(V_a), len(A_a))
    return 1 - (K * (dtw_v + dtw_a) / (4 * max_len)) if max_len > 0 else 1.0

def ebs(user_seq: np.ndarray, agent_seq: np.ndarray, delta_V: float = 0.211, delta_A: float = -0.105, delta_D: float = 0.098, K: float = 5.0) -> float:
    """
    Emotional Balancing Score (EBS) for a single turn.
    Evaluates the agent's ability to balance the user's extreme emotions.
    A score of 1.0 is returned if no extreme emotion is detected. Otherwise, it measures
    the distance to an ideal balanced response.

    Args:
        user_seq (np.ndarray): User VAD sequence.
        agent_seq (np.ndarray): Agent VAD sequence.
        delta_V, delta_A, delta_D (float): Target adjustment values for V, A, D.
        K (float): A scaling constant.

    Returns:
        float: The EBS score.
    """
    V_u, A_u, D_u = user_seq[:, 0], user_seq[:, 1], user_seq[:, 2]
    V_a, A_a, D_a = agent_seq[:, 0], agent_seq[:, 1], agent_seq[:, 2]
    
    thresholds = {'V_low': -0.07, 'A_high': 0.345, 'D_low': 0.210}
    user_avg = np.mean(user_seq, axis=0)
    
    is_extreme = [
        user_avg[0] < thresholds['V_low'],
        user_avg[1] > thresholds['A_high'],
        user_avg[2] < thresholds['D_low']
    ]

    if not any(is_extreme):
        return 1.0  # No balancing needed

    dtw_v = dtw_distance(V_u + delta_V, V_a)
    dtw_a = dtw_distance(A_u + delta_A, A_a)
    dtw_d = dtw_distance(D_u + delta_D, D_a)
    
    dtw_sum = dtw_v * is_extreme[0] + dtw_a * is_extreme[1] + dtw_d * is_extreme[2]
    max_len = max(len(V_u), len(A_u), len(D_u))
    
    return 1 - (K / sum(is_extreme)) * (dtw_sum / max_len) if max_len > 0 else 1.0

def ess(agent_seq: np.ndarray, threshold: float = 0.04, K: float = 5.0) -> float:
    """
    Emotional Stability Score (ESS) for a single turn.
    Measures the stability of the agent's emotion during its turn. Penalizes
    large, frequent fluctuations.

    Args:
        agent_seq (np.ndarray): Agent VAD sequence.
        threshold (float): Fluctuation threshold to consider significant.
        K (float): A scaling constant.

    Returns:
        float: The ESS score.
    """
    if len(agent_seq) < 2:
        return 1.0  # Cannot calculate change with less than 2 frames

    deltas = np.abs(np.diff(agent_seq, axis=0))
    big_deltas = deltas[deltas > threshold]
    
    if big_deltas.size == 0:
        return 1.0

    return 1 - K * np.sum(big_deltas) / big_deltas.size

def ers(user_seq: np.ndarray, agent_seq: np.ndarray) -> float:
    """
    Emotional Reasoning Score (ERS) for a single turn.
    A weighted combination of ECS, EBS, and ESS. The weights change
    depending on whether emotional balancing was required.
    """
    ecs_score = ecs(user_seq, agent_seq)
    ebs_score = ebs(user_seq, agent_seq)
    ess_score = ess(agent_seq)
    
    # Use different weights if emotional balancing was performed (EBS != 1.0)
    if ebs_score != 1.0:
        alpha, beta, gamma = 0.4, 0.4, 0.2
    else:
        alpha, beta, gamma = 0.7, 0.0, 0.3

    return alpha * ecs_score + beta * ebs_score + gamma * ess_score

# --- Evaluation Runners ---

def read_vad_from_csv(file_path: str) -> np.ndarray:
    """Reads a CSV and returns the VAD columns as a numpy array."""
    df = pd.read_csv(file_path)
    return df[['valence', 'arousal', 'dominance']].values

def evaluate_single_turn_folder(user_dir: str, agent_dir: str, output_path: str):
    """
    Calculates and saves single-turn scores (ECS, EBS, ESS, ERS) for all
    matching CSVs in two directories.
    """
    user_files = glob(os.path.join(user_dir, "*.csv"))
    results = []

    for user_file in user_files:
        filename = os.path.basename(user_file)
        agent_file = os.path.join(agent_dir, filename)

        if not os.path.exists(agent_file):
            print(f"Warning: Corresponding agent file not found for {filename}. Skipping.")
            continue

        user_seq = read_vad_from_csv(user_file)
        agent_seq = read_vad_from_csv(agent_file)

        results.append({
            'filename': filename,
            'ECS': ecs(user_seq, agent_seq),
            'EBS': ebs(user_seq, agent_seq),
            'ESS': ess(agent_seq),
            'ERS': ers(user_seq, agent_seq)
        })

    if not results:
        print("No matching files found to evaluate.")
        return

    result_df = pd.DataFrame(results)
    
    # Calculate averages. For EBS, only average the turns where balancing was needed.
    avg_ebs = result_df[result_df['EBS'] != 1.0]['EBS'].mean()
    avg_scores = {
        'filename': 'Average',
        'ECS': result_df['ECS'].mean(),
        'EBS': avg_ebs if not pd.isna(avg_ebs) else 1.0, # if no extreme turns, avg is 1.0
        'ESS': result_df['ESS'].mean(),
        'ERS': result_df['ERS'].mean()
    }
    
    final_df = pd.concat([result_df, pd.DataFrame([avg_scores])], ignore_index=True)
    final_df.round(4).to_csv(output_path, index=False)
    print(f"Single-turn evaluation results saved to {output_path}")

def get_dialogue_sequences(dialogue_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    A helper for multi-turn evaluation. This function is an example for the
    dGSLM dataset structure (speaker_0, speaker_1). You may need to adapt it
    for other dataset formats.
    """
    speaker_0_dir = os.path.join(dialogue_path, "speaker_0")
    speaker_1_dir = os.path.join(dialogue_path, "speaker_1")

    # This assumes speaker_0 is the user and speaker_1 is the agent
    user_files = sorted(glob(os.path.join(speaker_0_dir, "*.csv")))
    agent_files = sorted(glob(os.path.join(speaker_1_dir, "*.csv")))

    num_turns = min(len(user_files), len(agent_files))
    if num_turns < 2:
        return [], []

    user_seqs = [read_vad_from_csv(f) for f in user_files[:num_turns]]
    agent_seqs = [read_vad_from_csv(f) for f in agent_files[:num_turns]]
    
    return user_seqs, agent_seqs

def evaluate_multi_turn_folder(dialogues_dir: str, output_path: str):
    """
    Calculates and saves multi-turn (Cross-Turn, CT) scores for all
    dialogue subdirectories. This function assumes a specific data layout
    (e.g., dGSLM format) and uses get_dialogue_sequences helper.
    """
    dialogue_folders = [d.path for d in os.scandir(dialogues_dir) if d.is_dir()]
    results = []

    for dialogue_path in dialogue_folders:
        user_seqs, agent_seqs = get_dialogue_sequences(dialogue_path)
        if not user_seqs:
            print(f"Warning: Not enough turns in {os.path.basename(dialogue_path)}. Skipping.")
            continue
        
        # Calculate Cross-Turn (CT) metrics
        ct_ecs_scores = [ecs(u, a) for u, a in zip(user_seqs, agent_seqs)]
        ct_ebs_scores = [ebs(u, a) for u, a in zip(user_seqs, agent_seqs)]
        
        # Calculate CT-ESS (stability between agent turns)
        ct_ess_sum = 0
        for i in range(len(agent_seqs) - 1):
            s1, s2 = agent_seqs[i], agent_seqs[i+1]
            d_v = dtw_distance(s1[:, 0], s2[:, 0])
            d_a = dtw_distance(s1[:, 1], s2[:, 1])
            d_d = dtw_distance(s1[:, 2], s2[:, 2])
            max_len = max(len(s1), len(s2))
            ct_ess_sum += (d_v + d_a + d_d) / (6 * max_len) if max_len > 0 else 0
        
        ct_ess = 1 - 5 * (ct_ess_sum / (len(agent_seqs) - 1)) if len(agent_seqs) > 1 else 1.0

        # Aggregate CT scores
        ct_ecs = np.mean(ct_ecs_scores)
        valid_ebs_scores = [s for s in ct_ebs_scores if s != 1.0]
        ct_ebs = np.mean(valid_ebs_scores) if valid_ebs_scores else 1.0

        alpha, beta, gamma = (0.35, 0.35, 0.3) if valid_ebs_scores else (0.6, 0, 0.4)
        ct_ers = alpha * ct_ecs + beta * ct_ebs + gamma * ct_ess

        results.append({
            'dialogue': os.path.basename(dialogue_path),
            'CT-ECS': ct_ecs, 'CT-EBS': ct_ebs, 'CT-ESS': ct_ess, 'CT-ERS': ct_ers
        })
    
    if not results:
        print("No valid dialogues found for multi-turn evaluation.")
        return

    result_df = pd.DataFrame(results)
    avg_scores = {
        'dialogue': 'Average',
        'CT-ECS': result_df['CT-ECS'].mean(),
        'CT-EBS': result_df[result_df['CT-EBS'] != 1.0]['CT-EBS'].mean(),
        'CT-ESS': result_df['CT-ESS'].mean(),
        'CT-ERS': result_df['CT-ERS'].mean()
    }
    final_df = pd.concat([result_df, pd.DataFrame([avg_scores])], ignore_index=True)
    final_df.round(4).to_csv(output_path, index=False)
    print(f"Multi-turn evaluation results saved to {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate continuous emotional reasoning metrics.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Evaluation mode')

    # Sub-parser for single-turn evaluation
    parser_single = subparsers.add_parser('single-turn', help='Evaluate single-turn interactions.')
    parser_single.add_argument('--user_dir', type=str, required=True, help="Directory with user emotion CSVs.")
    parser_single.add_argument('--agent_dir', type=str, required=True, help="Directory with agent response CSVs.")
    parser_single.add_argument('--output_csv', type=str, required=True, help="Path to save the output CSV.")

    # Sub-parser for multi-turn evaluation
    parser_multi = subparsers.add_parser('multi-turn', help='Evaluate multi-turn dialogues.')
    parser_multi.add_argument('--dialogues_dir', type=str, required=True, help="Directory containing dialogue subfolders.")
    parser_multi.add_argument('--output_csv', type=str, required=True, help="Path to save the output CSV.")

    args = parser.parse_args()

    if args.command == 'single-turn':
        evaluate_single_turn_folder(args.user_dir, args.agent_dir, args.output_csv)
    elif args.command == 'multi-turn':
        evaluate_multi_turn_folder(args.dialogues_dir, args.output_csv)

# Example Usage:
#
# 1. Single-Turn Evaluation:
# python continuous_evaluate.py single-turn \
#   --user_dir /path/to/emotion/continuous/tts_result/single_turn/happy \
#   --agent_dir /path/to/emotion/continuous/LLaMA-Omni/single-turn/happy \
#   --output_csv /path/to/eval_res/continuous/LLaMA-Omni_happy_single_turn.csv
#
# 2. Multi-Turn Evaluation (e.g., for dGSLM dataset structure):
# python continuous_evaluate.py multi-turn \
#   --dialogues_dir /path/to/emotion/continuous/dGSLM \
#   --output_csv /path/to/eval_res/continuous/dGSLM_multi_turn.csv
# --- END OF REVISED FILE continuous_evaluate.py ---