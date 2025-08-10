import pandas as pd
import os
import argparse
from typing import Dict, Tuple

SCORE_TABLE: Dict[Tuple[str, str], float] = {
    ('NEUTRAL', 'NEUTRAL'): 0.9, ('NEUTRAL', 'HAPPY'): 0.6, ('NEUTRAL', 'ANGRY'): 0.3, ('NEUTRAL', 'SAD'): 0.4,
    ('HAPPY', 'NEUTRAL'): 0.5, ('HAPPY', 'HAPPY'): 1.0, ('HAPPY', 'ANGRY'): 0.2, ('HAPPY', 'SAD'): 0.2,
    ('ANGRY', 'NEUTRAL'): 0.8, ('ANGRY', 'HAPPY'): 0.1, ('ANGRY', 'ANGRY'): 0.4, ('ANGRY', 'SAD'): 0.5,
    ('SAD', 'NEUTRAL'): 0.6, ('SAD', 'HAPPY'): 0.2, ('SAD', 'ANGRY'): 0.4, ('SAD', 'SAD'): 0.9
}

def calculate_appropriateness_score(user_emotion_path: str, agent_response_path: str, output_csv_path: str):
    """
    Calculates the emotional appropriateness score for a set of conversations.

    It matches filenames between the user and agent CSVs, looks up the score
    for the (user_emotion, agent_emotion) pair in SCORE_TABLE, and saves the
    results, including an average score, to a new CSV file.

    Args:
        user_emotion_path (str): Path to the CSV file with user emotions.
        agent_response_path (str): Path to the CSV file with agent's response emotions.
        output_csv_path (str): Path to save the detailed scoring results.
    """
    try:
        user_df = pd.read_csv(user_emotion_path)
        agent_df = pd.read_csv(agent_response_path)
    except FileNotFoundError as e:
        print(f"Warning: Could not find file {e.filename}. Skipping this evaluation.")
        return

    scores = []
    total_score = 0.0
    valid_responses = 0

    for _, row in user_df.iterrows():
        filename = row['Filename']
        user_emotion = row['Emotion']

        # Find the corresponding agent response
        agent_row = agent_df[agent_df['Filename'] == filename]
        if not agent_row.empty:
            agent_emotion = agent_row['Emotion'].iloc[0]
            # Get score from the predefined table, default to 0 if pair not found
            score = SCORE_TABLE.get((user_emotion, agent_emotion), 0.0)
            scores.append({'Filename': filename, 'Score': score})
            total_score += score
            valid_responses += 1

    # Calculate average score, avoiding division by zero
    avg_score = total_score / valid_responses if valid_responses > 0 else 0.0

    # Create and save the results DataFrame
    result_df = pd.DataFrame(scores)
    # Add a row for the average score
    avg_row = pd.DataFrame([{'Filename': 'Average', 'Score': avg_score}])
    final_df = pd.concat([result_df, avg_row], ignore_index=True)
    final_df.to_csv(output_csv_path, index=False)


def main(args):
    """
    Main function to orchestrate the evaluation of all specified models.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    emotions = ['neutral', 'happy', 'angry', 'sad']
    all_models_summary = []

    # If no specific models are given, discover them from the response directory
    if not args.models:
        models_to_evaluate = [d for d in os.listdir(args.agent_response_dir) if os.path.isdir(os.path.join(args.agent_response_dir, d))]
        print(f"No models specified. Found and evaluating: {models_to_evaluate}")
    else:
        models_to_evaluate = args.models

    for model_name in models_to_evaluate:
        print(f"\n--- Evaluating Model: {model_name} ---")
        model_scores = {'model': model_name}
        
        # This assumes a structure like .../agent_response_dir/model_name/single-turn/
        agent_emotion_base_dir = os.path.join(args.agent_response_dir, model_name, 'single-turn')

        if not os.path.isdir(agent_emotion_base_dir):
            print(f"Warning: Directory not found for model '{model_name}': {agent_emotion_base_dir}. Skipping.")
            continue

        emotion_avg_scores = []
        for emotion in emotions:
            user_csv_path = os.path.join(args.user_emotion_dir, f'{emotion}.csv')
            agent_csv_path = os.path.join(agent_emotion_base_dir, f'{emotion}.csv')
            
            # Per-emotion detailed output path
            output_csv_path = os.path.join(args.output_dir, f'{model_name}_{emotion}_scores.csv')
            
            calculate_appropriateness_score(user_csv_path, agent_csv_path, output_csv_path)

            # Read the average score back from the generated file for summary
            if os.path.exists(output_csv_path):
                emotion_df = pd.read_csv(output_csv_path)
                avg_score = emotion_df.loc[emotion_df['Filename'] == 'Average', 'Score'].iloc[0]
                model_scores[emotion] = avg_score
                emotion_avg_scores.append(avg_score)

        # Calculate the overall average score for the model
        if emotion_avg_scores:
            model_scores['overall_average'] = sum(emotion_avg_scores) / len(emotion_avg_scores)
        
        all_models_summary.append(model_scores)

    # Save the final summary of all models
    summary_df = pd.DataFrame(all_models_summary)
    summary_output_path = os.path.join(args.output_dir, 'models_appropriateness_summary.csv')
    summary_df.round(3).to_csv(summary_output_path, index=False)
    print(f"\nEvaluation complete. Summary saved to: {summary_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate categorical emotional appropriateness of agent responses.")
    parser.add_argument('--user_emotion_dir', type=str, required=True,
                        help="Directory containing the ground truth user emotion CSVs (e.g., 'neutral.csv').")
    parser.add_argument('--agent_response_dir', type=str, required=True,
                        help="Parent directory containing subdirectories for each model's responses.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the detailed and summary evaluation results.")
    parser.add_argument('--models', nargs='*',
                        help="Optional: A list of specific model names (subdirectories) to evaluate. "
                             "If not provided, all subdirectories will be evaluated.")

    args = parser.parse_args()
    main(args)

# Example Usage:
# python categorical_evaluate.py \
#   --user_emotion_dir /path/to/emotion/categorical/tts_result/single-turn \
#   --agent_response_dir /path/to/emotion/categorical \
#   --output_dir /path/to/eval_res/categorical/ \
#   --models LLaMA-Omni mini-omni Freeze-omni
# --- END OF REVISED FILE categorical_evaluate.py ---