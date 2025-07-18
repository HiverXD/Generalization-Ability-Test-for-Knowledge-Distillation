import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


class MNISTDataManager:
    """Data management class"""
    
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # dataset
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=self.transform
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=self.transform
        )
        
        # dataloader
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def generate_soft_targets(self, teacher_model, temperature=8, device='cpu'):
        """generate soft target using teacher model"""
        teacher_model.eval()
        soft_targets = []
        
        with torch.no_grad():
            for data, _ in tqdm(self.train_loader, desc="Generating soft targets"):
                data = data.to(device)
                output = teacher_model(data)
                soft_target = F.softmax(output / temperature, dim=1)
                soft_targets.append(soft_target.cpu())
        
        soft_targets = torch.cat(soft_targets, dim=0)
        print(f"Soft targets shape: {soft_targets.shape}")
        return soft_targets
    
    def create_dataset_without_label(self, exclude_label=3):
        """dataset without exclusive label"""
        train_data_filtered = []
        train_labels_filtered = []
        
        for data, label in self.train_dataset:
            if label != exclude_label:
                train_data_filtered.append(data)
                train_labels_filtered.append(label)
        
        train_data_filtered = torch.stack(train_data_filtered)
        train_labels_filtered = torch.tensor(train_labels_filtered)
        
        print(f"Original dataset size: {len(self.train_dataset)}")
        print(f"Filtered dataset size (without label {exclude_label}): {len(train_data_filtered)}")
        
        # new dataset, data loader
        filtered_dataset = TensorDataset(train_data_filtered, train_labels_filtered)
        filtered_loader = DataLoader(filtered_dataset, batch_size=self.batch_size, shuffle=True)
        
        return filtered_loader, train_data_filtered, train_labels_filtered
    
    def filter_soft_targets(self, soft_targets, exclude_label=3):
        """filter soft target except exclusive label"""
        filtered_soft_targets = []
        
        for i, (_, label) in enumerate(self.train_dataset):
            if label != exclude_label:
                filtered_soft_targets.append(soft_targets[i])
        
        filtered_soft_targets = torch.stack(filtered_soft_targets)
        print(f"Filtered soft targets shape: {filtered_soft_targets.shape}")
        return filtered_soft_targets