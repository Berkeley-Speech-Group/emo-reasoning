import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import warnings

from CSER import CSERM
from Dataset import MSPFeatureDataset

warnings.filterwarnings("ignore", category=UserWarning)

def ccc_loss(y_true: torch.Tensor, y_pred: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Concordance Correlation Coefficient (CCC) loss.
    This loss is suitable for evaluating agreement between two continuous variables.

    Args:
        y_true (torch.Tensor): Ground truth labels. Shape: [b, t, 3]
        y_pred (torch.Tensor): Predicted labels. Shape: [b, t, 3]
        padding_mask (torch.Tensor): Mask indicating padded elements. Shape: [b, t]

    Returns:
        torch.Tensor: The CCC loss value.
    """
    # Ensure shapes are compatible, allowing for a 1-step difference due to model architecture
    if y_true.shape[1] > y_pred.shape[1]:
        y_true = y_true[:, :y_pred.shape[1], :]
        padding_mask = padding_mask[:, :y_pred.shape[1]]
    elif y_pred.shape[1] > y_true.shape[1]:
        y_pred = y_pred[:, :y_true.shape[1], :]

    valid_mask = (~padding_mask).unsqueeze(-1)
    
    # Apply mask to handle padded sequences
    masked_y_true = y_true * valid_mask
    masked_y_pred = y_pred * valid_mask
    num_valid = torch.sum(valid_mask, dim=1)

    mean_true = torch.sum(masked_y_true, dim=1) / num_valid
    mean_pred = torch.sum(masked_y_pred, dim=1) / num_valid
    
    var_true = torch.sum((masked_y_true - mean_true.unsqueeze(1))**2, dim=1) / num_valid
    var_pred = torch.sum((masked_y_pred - mean_pred.unsqueeze(1))**2, dim=1) / num_valid
    
    covar = torch.sum((masked_y_true - mean_true.unsqueeze(1)) * (masked_y_pred - mean_pred.unsqueeze(1)), dim=1) / num_valid
    
    ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)
    
    # Loss is 1 - mean CCC over the batch and dimensions
    return 1 - torch.mean(ccc)

def collate_fn(batch: list) -> tuple:
    """
    Custom collate function to pad sequences within a batch.
    Features are assumed to have a fixed length after preprocessing.
    Labels need to be padded to the maximum length in the batch.
    """
    features, labels = zip(*batch)
    
    features = torch.stack(features)
    
    # Pad labels and create a corresponding padding mask
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)
    label_padding_mask = (padded_labels == 0.0).all(dim=-1) # Mask is True for padded steps

    return features, padded_labels, label_padding_mask

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", ncols=100)
    
    for features, labels, label_mask in progress_bar:
        features, labels, label_mask = features.to(device), labels.to(device), label_mask.to(device)
        
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(labels, output, label_mask)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for features, labels, label_mask in tqdm(dataloader, desc="Evaluating", ncols=100):
            features, labels, label_mask = features.to(device), labels.to(device), label_mask.to(device)
            output = model(features)
            loss = criterion(labels, output, label_mask)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets and DataLoaders
    train_dataset = MSPFeatureDataset(args.train_feature_dir, args.train_label_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_dataset = MSPFeatureDataset(args.val_feature_dir, args.val_label_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, Optimizer, Criterion
    bilstm_params = {'hidden_size': args.hidden_size, 'num_layers': args.num_layers}
    model = CSERM(bilstm_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = ccc_loss
    
    # Training Loop
    os.makedirs(args.ckpt_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save model checkpoint periodically
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSERM model on pre-extracted features.")
    parser.add_argument('--train_feature_dir', type=str, required=True, help='Directory of training features.')
    parser.add_argument('--train_label_dir', type=str, required=True, help='Directory of training labels.')
    parser.add_argument('--val_feature_dir', type=str, required=True, help='Directory of validation features.')
    parser.add_argument('--val_label_dir', type=str, required=True, help='Directory of validation labels.')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of the BiLSTM.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the BiLSTM.')
    parser.add_argument('--save_interval', type=int, default=20, help='Save checkpoint every N epochs.')
    
    args = parser.parse_args()
    main(args)