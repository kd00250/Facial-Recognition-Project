import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, train_dataset, val_dataset, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, num_epochs=10, early_stop_patience=5):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # ----- Training Phase -----
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            for images, labels in train_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_bar.set_postfix(loss=loss.item())
            
            epoch_loss = running_loss / len(self.train_dataset)
            epoch_accuracy = 100 * correct / total
            print(f"\nEpoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            
            # ----- Validation Phase -----
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            val_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
            with torch.no_grad():
                for images, labels in val_bar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    val_bar.set_postfix(loss=loss.item())
            
            val_loss = val_loss / len(self.val_dataset)
            val_accuracy = 100 * correct_val / total_val
            print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
            
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("Model checkpoint saved!")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch duration: {epoch_duration:.2f} seconds\n")
