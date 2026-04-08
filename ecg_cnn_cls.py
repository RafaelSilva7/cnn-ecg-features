import torch
import torch.nn as nn
import torch.optim as optim

from cnn import ECGClassifier
from datautils import get_dataloaders



# ==========================================
# TRAINING & VALIDATION ROUTINE
# ==========================================
def train_and_evaluate(train_loader, val_loader, model, num_epochs, lr, device):
    
    # Setup Loss, and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    best_val_loss = float("inf")
    counter = 0
    patience = 5
    
    # The Loop
    print("Starting training loop...\n" + "-"*40)
    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        model.train() 
        # TODO: Save the epoch-level metrics (loss, acc) for both train and val to plot later
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
            train_correct += (predicted == torch.argmax(labels, 1)).sum().item()
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
                val_correct += (predicted == torch.argmax(labels, 1)).sum().item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        # ---- BEST MODEL CHECK ----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0

            torch.save(model.state_dict(), "results/best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, "best_model.pt")
            print(f"[Epoch {epoch+1}] ✅ New best model: {val_loss:.4f}")
        else:
            counter += 1
        
        # --- LOGGING ---
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] "
              f"| Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:5.2f}%")
        
        if counter >= patience:
            print("Early stopping")
            break

    print("-" * 40 + "\nTraining Complete!")



# ==========================================
# SCRIPT EXECUTION
# ==========================================
if __name__ == "__main__":

    # TODO: Read the args

    # Configuration
    n_epochs = 10
    lr = 1e-3 #  1e-3 to 1e-5

    # Setup Data and Model
    print("Loading data...")
    # TODO: Load the labels as class indices (0, 1, 2, ...) instead of one-hot encoding to use CrossEntropyLoss directly without argmax in the training loop
    train_loader, val_loader = get_dataloaders(train_path="data/cinc/ptbxl/train.npz", 
                                               val_path="data/cinc/ptbxl/val.npz", batch_size=16)
    
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = ECGClassifier(task="multiclass").to(device)
    print(f"Initializing setup on device: {device}")

    train_and_evaluate(train_loader, val_loader, model, n_epochs, lr, device)

    # TODO: Load the best model and evaluate on the test set

    model.to("cpu")