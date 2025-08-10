import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
import seaborn as sns
import argparse
from CSER import CSERM

# Set global font for plotting
plt.rcParams['font.family'] = 'Times New Roman'

def run_inference(model: torch.nn.Module, feature_path: str, device: torch.device) -> np.ndarray:
    """
    Runs inference on a single pre-extracted feature file.

    Args:
        model (torch.nn.Module): The trained CSERM model.
        feature_path (str): Path to the .pt feature file.
        device (torch.device): The device to run inference on.

    Returns:
        np.ndarray: The predicted VAD values.
    """
    features = torch.load(feature_path).unsqueeze(0) # Add batch dimension
    
    model.eval()
    with torch.no_grad():
        predicted_output = model(features.to(device)).cpu().squeeze(0)

    return predicted_output.numpy()

def plot_predictions_vs_ground_truth(ground_truth: np.ndarray, predictions: np.ndarray, save_path: str):
    """
    Plots the predicted VAD values against the ground truth and saves the figure.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 7))

    labels = ['Arousal', 'Valence', 'Dominance']
    colors = sns.color_palette("deep", 3)
    
    for i, (label_name, color) in enumerate(zip(labels, colors)):
        # Smooth curves for better visualization
        x_gt = np.arange(len(ground_truth))
        x_pred = np.arange(len(predictions))
        
        f_gt = interpolate.interp1d(x_gt, ground_truth[:, i], kind='cubic', fill_value="extrapolate")
        f_pred = interpolate.interp1d(x_pred, predictions[:, i], kind='cubic', fill_value="extrapolate")
        
        x_smooth = np.linspace(0, len(x_gt)-1, 300)
        
        # Plot ground truth and prediction curves
        plt.plot(x_smooth, f_gt(x_smooth), label=f"Ground-Truth ({label_name})", linestyle='--', color=color, linewidth=3)
        plt.plot(x_smooth, f_pred(x_smooth), label=f"Prediction ({label_name})", linestyle='-', color=color, linewidth=3)

    plt.xlabel("Time (seconds)", fontsize=20, labelpad=15)
    plt.ylabel("Value", fontsize=20, labelpad=15)
    plt.xticks(fontsize=16)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=16)
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper right', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    bilstm_params = {'hidden_size': 512, 'num_layers': 2}
    model = CSERM(bilstm_params).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Run inference
    predictions = run_inference(model, args.feature_path, device)
    
    # Load ground truth labels
    labels_df = pd.read_csv(args.label_path)
    labels_df[['arousal', 'valence', 'dominance']] /= 100.0
    ground_truth = labels_df[['arousal', 'valence', 'dominance']].values

    # Plot results
    plot_predictions_vs_ground_truth(ground_truth, predictions, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single audio file and plot the results.")
    parser.add_argument('--feature_path', type=str, required=True, help='Path to the pre-extracted feature file (.pt).')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the ground truth label file (.csv).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained CSERM checkpoint.')
    parser.add_argument('--output_path', type=str, default='prediction_comparison.pdf', help='Path to save the output plot.')
    
    args = parser.parse_args()
    main(args)