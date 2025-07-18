import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def train_normal(self, model, train_loader, test_loader, lr=0.001, epochs=10):
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in tqdm(range(epochs)):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{train_loss/len(train_loader):.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # validation
            val_loss, val_acc = self.evaluate(model, test_loader)
            
            # history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(100. * train_correct / train_total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
        
        return history
    
    def train_with_kd(self, student_model, train_loader, test_loader, soft_targets, 
                      temperature=8, alpha=0.7, lr=0.001, epochs=10):
        """Knowledge Distillation 학습"""
        student_model.to(self.device)
        criterion_hard = nn.CrossEntropyLoss()
        criterion_soft = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(student_model.parameters(), lr=lr)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in tqdm(range(epochs)):
            student_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f'KD Epoch {epoch+1}/{epochs}')
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)
                
                # load soft target wrt current batch
                batch_start = batch_idx * train_loader.batch_size
                batch_end = min(batch_start + train_loader.batch_size, len(soft_targets))
                soft_target_batch = soft_targets[batch_start:batch_end].to(self.device)
                
                optimizer.zero_grad()
                student_output = student_model(data)
                
                # Hard target loss
                hard_loss = criterion_hard(student_output, target)
                
                # Soft target loss
                student_soft = F.log_softmax(student_output / temperature, dim=1)
                soft_loss = criterion_soft(student_soft, soft_target_batch)
                
                # Combined loss
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(student_output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # validation
            val_loss, val_acc = self.evaluate(student_model, test_loader)
            
            # history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(100. * train_correct / train_total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
        
        return history
    
    def evaluate(self, model, test_loader):
        """evaluation"""
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
    
    def evaluate_detailed(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # acc for each class
                c = (predicted == target).squeeze()
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        overall_accuracy = 100. * correct / total
        
        print(f'Overall Accuracy: {overall_accuracy:.2f}%')
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'Class {i}: {class_acc:.2f}%')
        
        return overall_accuracy, class_correct, class_total
    
    def plot_history(self, history, title="Training History"):
        """learning curve"""
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.title(f'{title} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title(f'{title} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, histories, model_names, accuracies):
        """comparision"""
        plt.figure(figsize=(15, 5))
        
        # loss
        plt.subplot(1, 3, 1)
        for history, name in zip(histories, model_names):
            epochs = range(1, len(history['train_loss']) + 1)
            plt.plot(epochs, history['train_loss'], label=name)
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # acc
        plt.subplot(1, 3, 2)
        for history, name in zip(histories, model_names):
            epochs = range(1, len(history['val_acc']) + 1)
            plt.plot(epochs, history['val_acc'], label=name)
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # acc
        plt.subplot(1, 3, 3)
        colors = ['blue', 'orange', 'green', 'red']
        bars = plt.bar(model_names, accuracies, color=colors[:len(model_names)])
        plt.title('Final Test Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        
        # labeling
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()