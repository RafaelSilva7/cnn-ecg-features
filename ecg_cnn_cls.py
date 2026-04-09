import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from cnn import ECGClassifier
from datautils import get_dataloaders
from utils import set_seed, timestamp, MetricsLogger, classification_metrics



# ==========================================
# TRAINING & VALIDATION ROUTINE
# ==========================================
def train_and_evaluate(train_loader, val_loader, model, num_epochs, lr, device, save_dir):
    
    # Setup Loss, and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # Initialize Metrics Logger
    logger = MetricsLogger(save_dir)
    best_val_loss = float("inf")
    counter = 0
    patience = 5
    
    # Start Training Loop
    print("\nStarting training loop...\n" + "-"*40)
    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        model.train() 
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * inputs.size(0) # "Un-averaging" to get the total batch loss

            predicted = torch.argmax(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        
        # --- VALIDATION PHASE ---
        model.eval() # Sets dropout/batchnorm to evaluation mode
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Metrics
                val_loss += loss.item() * inputs.size(0) # "Un-averaging" to get the total batch loss

                predicted = torch.argmax(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        # ---- BEST MODEL CHECK ----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, save_dir / "best_model.pt")

            print(f"[Epoch {epoch+1}] ✅ New best model: {val_loss:.4f}")
        else:
            counter += 1
        
        # --- LOGGING ---
        logger.log(epoch+1, epoch_train_loss, epoch_val_loss, train_acc=epoch_train_acc, val_acc=epoch_val_acc)
        logger.save_csv()
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] "
              f"| Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:5.2f}%")
        
        # --- EARLY STOPPING CHECK ---
        if counter >= patience:
            print("\n# Early stopping! No improvement in validation loss for 5 consecutive epochs.")
            break

    print("-" * 40 + "\nTraining Complete!\n")


def evaluate_model(test_loader, model, device, save_dir: Path):

    criterion = nn.CrossEntropyLoss()  # or MSELoss()
    total_loss = 0.0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # ---- classification ----
            y = y.to(device).long().view(-1)

            logits = model(x)
            # TODO: Save the extracted features 
            # features = model._feat.cpu().numpy()  # `_feat` is the extracted features from the model

            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0) # "Un-averaging" to get the total batch loss

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    model.to("cpu")

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    # ---- Compute the evaluation metrics ----
    all_probs = torch.softmax(all_logits, dim=1)
    all_preds = torch.argmax(all_probs, dim=1)


    metric = classification_metrics(all_preds, all_targets)

    prob_pos_class = all_probs[:, 1]  # Assuming class 1 is the positive class
    metric['auc'] = roc_auc_score(all_targets, prob_pos_class, multi_class='ovr')
    print(f"AUROC: {metric['auc']:.4f}")
    
    metric['avg_loss'] = total_loss / all_targets.size(0)
    print(f"Test avg_loss: {metric['avg_loss']:.4f}")
    
    report = classification_report(all_targets, all_preds, target_names=['male', 'female'])  # Adjust target names as needed
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # --- Save the report and confusion matrix to files ---
    pd.DataFrame(cm).to_csv(save_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame([metric]).to_csv(save_dir / "metrics.csv", index=False)

    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    np.savez_compressed(save_dir / "test_probs.npz", logits=all_logits.numpy(), probs=all_probs.numpy())


def get_args():
    parser = argparse.ArgumentParser(description="ECG Training Script")

    parser.add_argument("--exp", type=str, required=True, help="Experiment name for logging and saving models")
    # parser.add_argument("--data_path", type=str, required=True)
    # parser.add_argument("--task", type=str, default="multiclass",
    #                     choices=["multiclass", "multilabel", "regression"],
    #                     help="Task type")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0, cuda:1, or cpu")

    return parser.parse_args()


# ==========================================
# SCRIPT EXECUTION
# ==========================================
if __name__ == "__main__":

    # ---- Arguments and Setup ----
    args = get_args()
    print("Parameters: ", vars(args))

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    set_seed(args.seed, deterministic=True)

    # Setup Data and Model
    print("\nLoading data...", end="")
    train_loader, val_loader, test_loader = get_dataloaders(dataset_dir=Path("data/cinc/ptbxl"), batch_size=16)

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = ECGClassifier(task="multiclass").to(device)
    print(f"Initializing setup on device: {device}")

    # Create a unique directory for this experiment
    save_dir = Path(f"results/{args.exp}_{args.seed}_{timestamp()}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    train_and_evaluate(train_loader, val_loader, model, args.epochs, args.lr, device, save_dir)

    evaluate_model(test_loader, model, device, save_dir)

    model.to("cpu")