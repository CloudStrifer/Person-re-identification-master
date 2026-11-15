import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import time
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class KnowledgeDistillationLoss(nn.Module):
    """
    Enhanced Knowledge Distillation Loss with multiple loss components
    """

    def __init__(self, alpha=0.7, temperature=4.0, feature_loss_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.feature_loss_weight = feature_loss_weight
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, labels,
                student_features=None, teacher_features=None):
        # Distillation loss (soft targets)
        distillation_loss = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # Student loss (hard targets)
        student_loss = self.ce_loss(student_logits, labels)

        # Feature matching loss (optional)
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
            feature_loss = self.mse_loss(student_features, teacher_features)

        # Combined loss
        total_loss = (self.alpha * distillation_loss +
                      (1 - self.alpha) * student_loss +
                      self.feature_loss_weight * feature_loss)

        return total_loss, distillation_loss, student_loss, feature_loss


class TeacherModel(nn.Module):
    """Enhanced Vision Transformer as Teacher Model with feature extraction"""

    def __init__(self, num_classes=1000, pretrained=True, model_name='vit_base_patch16_224' ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model_path = './model/pytorch_model.bin'
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        print(f"已从本地加载:{model_path}")

        self.feature_dim = self.model.embed_dim if hasattr(self.model, 'embed_dim') else 768

    def forward(self, x, return_features=False):
        if return_features:
            # Extract features before classification
            features = self.model.forward_features(x)

            # Handle different global pooling methods
            if hasattr(self.model, 'global_pool'):
                if callable(self.model.global_pool):
                    # If global_pool is a function/layer
                    features = self.model.global_pool(features)
                elif isinstance(self.model.global_pool, str):
                    # If global_pool is a string indicating pooling type
                    if self.model.global_pool == 'token':
                        features = features[:, 0]  # Use CLS token
                    elif self.model.global_pool == 'avg':
                        features = features.mean(dim=1)  # Average pooling
                    else:
                        features = features[:, 0]  # Default to CLS token
                else:
                    features = features[:, 0]  # Default to CLS token
            else:
                # Default for ViT models - use CLS token
                if len(features.shape) == 3:  # [batch, seq_len, embed_dim]
                    features = features[:, 0]  # Use CLS token
                else:
                    features = features  # Already pooled

            # Get final logits
            if hasattr(self.model, 'head'):
                logits = self.model.head(features)
            elif hasattr(self.model, 'classifier'):
                logits = self.model.classifier(features)
            else:
                # Create a simple linear layer if no head exists
                if not hasattr(self, 'temp_head'):
                    self.temp_head = nn.Linear(features.size(-1), num_classes).to(features.device)
                logits = self.temp_head(features)

            return logits, features
        else:
            return self.model(x)


class StudentModel(nn.Module):
    """Enhanced MobileNetV3 as Student Model with feature extraction"""

    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            self.backbone = mobilenet_v3_small(weights=None)

        # Extract feature extractor and classifier
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool

        # Original classifier dimensions
        original_classifier = self.backbone.classifier
        self.feature_dim = original_classifier[0].in_features

        # Custom classifier
        if num_classes != 1000:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 1024),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(1024, num_classes)
            )
        else:
            self.classifier = original_classifier

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)

        if return_features:
            logits = self.classifier(features)
            return logits, features
        else:
            logits = self.classifier(features)
            return logits


class KnowledgeDistillationTrainer:
    """Enhanced trainer class for knowledge distillation with comprehensive training pipeline"""

    def __init__(self, teacher_model, student_model, device='cuda',
                 save_dir='./checkpoints', log_dir='./logs'):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
        self.kd_loss = KnowledgeDistillationLoss()

        # Create directories
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # 添加投影层
        teacher_dim = teacher_model.feature_dim
        student_dim = student_model.feature_dim
        if teacher_dim != student_dim:
            self.projection = nn.Linear(student_dim, teacher_dim).to(device)
        else:
            self.projection = None

        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Training history
        self.history = {
            'train_loss': [], 'train_distill_loss': [], 'train_student_loss': [],
            'val_loss': [], 'val_accuracy': [], 'learning_rates': []
        }

        self.logger.info(f"Trainer initialized with device: {device}")
        self.logger.info(f"Teacher parameters: {self.count_parameters(self.teacher):,}")
        self.logger.info(f"Student parameters: {self.count_parameters(self.student):,}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def count_parameters(model):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_epoch(self, data_loader, optimizer, epoch, use_features=True):
        """Train for one epoch"""
        self.student.train()
        total_loss = 0.0
        total_distill_loss = 0.0
        total_student_loss = 0.0
        total_feature_loss = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()

            try:
                # Get teacher predictions (no gradients)
                with torch.no_grad():
                    if use_features:
                        teacher_logits, teacher_features = self.teacher(data, return_features=True)
                    else:
                        teacher_logits = self.teacher(data)
                        teacher_features = None

                # Get student predictions
                if use_features and teacher_features is not None:
                    student_logits, student_features = self.student(data, return_features=True)
                    # 投影学生特征到教师特征维度
                    if self.projection is not None:
                        student_features = self.projection(student_features)
                else:
                    student_logits = self.student(data)
                    student_features = None

                # Calculate loss
                loss, distill_loss, student_loss, feature_loss = self.kd_loss(
                    student_logits, teacher_logits, target, student_features, teacher_features
                )

                loss.backward()
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                total_distill_loss += distill_loss.item()
                total_student_loss += student_loss.item()
                total_feature_loss += feature_loss if isinstance(feature_loss, (int, float)) else feature_loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Distill': f'{distill_loss.item():.4f}',
                    'Student': f'{student_loss.item():.4f}'
                })

            except Exception as e:
                self.logger.error(f"Error in training step {batch_idx}: {str(e)}")
                # Try without features if feature extraction fails
                if use_features:
                    self.logger.warning("Retrying without feature matching...")
                    with torch.no_grad():
                        teacher_logits = self.teacher(data, return_features=False)
                    student_logits = self.student(data, return_features=False)

                    loss, distill_loss, student_loss, feature_loss = self.kd_loss(
                        student_logits, teacher_logits, target, None, None
                    )

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_distill_loss += distill_loss.item()
                    total_student_loss += student_loss.item()
                else:
                    raise e

        avg_loss = total_loss / len(data_loader)
        avg_distill = total_distill_loss / len(data_loader)
        avg_student = total_student_loss / len(data_loader)
        avg_feature = total_feature_loss / len(data_loader)

        return avg_loss, avg_distill, avg_student, avg_feature

    def validate(self, data_loader):
        """Validate the student model"""
        self.student.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)

                try:
                    # Student predictions
                    student_logits = self.student(data)

                    # Teacher predictions
                    teacher_logits = self.teacher(data)

                    # Calculate loss
                    loss, _, _, _ = self.kd_loss(student_logits, teacher_logits, target)
                    total_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(student_logits.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                except Exception as e:
                    self.logger.warning(f"Error in validation step: {str(e)}")
                    continue

        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, optimizer, scheduler=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': self.history
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.student.load_state_dict(checkpoint['student_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")

        return start_epoch

    def plot_training_history(self):
        """Plot training history"""
        if not self.history['train_loss']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss
        axes[0, 0].plot(self.history['train_loss'], label='Total Loss')
        axes[0, 0].plot(self.history['train_distill_loss'], label='Distillation Loss')
        axes[0, 0].plot(self.history['train_student_loss'], label='Student Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Validation metrics
        if self.history['val_loss']:
            axes[0, 1].plot(self.history['val_loss'], label='Validation Loss', color='red')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)

            axes[1, 0].plot(self.history['val_accuracy'], label='Validation Accuracy', color='green')
            axes[1, 0].set_title('Validation Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].grid(True)

        # Learning rate
        if self.history['learning_rates']:
            axes[1, 1].plot(self.history['learning_rates'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train(self, train_loader, val_loader, num_epochs, optimizer, scheduler=None,
              save_every=5, use_features=True, resume_from=None):
        """Complete training loop"""
        start_epoch = 0
        best_val_acc = 0.0

        # Resume from checkpoint if specified
        if resume_from is not None and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from, optimizer, scheduler)

        self.logger.info(f"Starting training from epoch {start_epoch} to {num_epochs}")

        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()

            # Training
            train_loss, train_distill, train_student, train_feature = self.train_epoch(
                train_loader, optimizer, epoch + 1, use_features
            )

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_distill_loss'].append(train_distill)
            self.history['train_student_loss'].append(train_student)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # Logging
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s"
            )

            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, optimizer, scheduler, is_best)

        self.logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

        # Save final history
        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        # Plot training history
        self.plot_training_history()


class RandomDataset(torch.utils.data.Dataset):
    """Generate random dataset for training/testing"""

    def __init__(self, num_samples=1000, input_size=224, num_classes=10):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes

        # Pre-generate all data to ensure consistency
        torch.manual_seed(42)  # For reproducibility
        self.data = torch.randn(num_samples, 3, input_size, input_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_data_loaders(num_classes=10, batch_size=32, num_workers=4, input_size=224,
                        train_samples=2000, val_samples=500):
    """Create training and validation data loaders with random data"""

    print(f"Creating random datasets:")
    print(f"  - Training samples: {train_samples}")
    print(f"  - Validation samples: {val_samples}")
    print(f"  - Input size: {input_size}x{input_size}")
    print(f"  - Number of classes: {num_classes}")

    # Create random datasets
    train_dataset = RandomDataset(
        num_samples=train_samples,
        input_size=input_size,
        num_classes=num_classes
    )

    val_dataset = RandomDataset(
        num_samples=val_samples,
        input_size=input_size,
        num_classes=num_classes
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, num_classes


def main():
    """Main training function with random data"""
    # Configuration
    config = {
        'num_classes': 1000,  # Number of classification classes
        'batch_size': 32,  # Batch size for training
        'num_epochs': 10,  # Reduced epochs for demo with random data
        'learning_rate': 0.001,  # Learning rate
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints',
        'log_dir': './logs',
        'use_features': True,  # Use feature matching loss
        'resume_from': None,  # Path to checkpoint to resume from
        'train_samples': 2000,  # Number of random training samples
        'val_samples': 500,  # Number of random validation samples
        'input_size': 224,  # Input image size
        'save_every': 10  # Save checkpoint every N epochs
    }

    print("Starting Knowledge Distillation Training with Random Data")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Create data loaders with random data
    print("\nCreating random data loaders...")
    train_loader, val_loader, num_classes = create_data_loaders(
        num_classes=config['num_classes'],
        batch_size=config['batch_size'],
        train_samples=config['train_samples'],
        val_samples=config['val_samples'],
        input_size=config['input_size']
    )

    # Initialize models
    print("\nInitializing models...")
    teacher = TeacherModel(num_classes=num_classes, pretrained=True)  # Set to False for faster init
    student = StudentModel(num_classes=num_classes, pretrained=False)

    print(f"Teacher model: Vision Transformer")
    print(f"Student model: MobileNetV3")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = KnowledgeDistillationTrainer(
        teacher, student,
        device=config['device'],
        save_dir=config['save_dir'],
        log_dir=config['log_dir']
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * 0.01
    )

    print(f"\nOptimizer: AdamW (lr={config['learning_rate']}, weight_decay=0.01)")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Device: {config['device']}")

    # Start training
    print("\nStarting training...")
    print("=" * 60)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        optimizer=optimizer,
        scheduler=scheduler,
        save_every=config['save_every'],
        use_features=config['use_features'],
        resume_from=config['resume_from']
    )

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Checkpoints saved in: {config['save_dir']}")
    print(f"Logs saved in: {config['log_dir']}")


def demo_quick_test():
    """Quick demonstration with minimal data"""
    print("Running Quick Demo with Minimal Random Data")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Minimal configuration for quick test
    num_classes = 5
    batch_size = 8
    num_epochs = 3

    # Create minimal random data
    train_loader, val_loader, _ = create_data_loaders(
        num_classes=num_classes,
        batch_size=batch_size,
        train_samples=40,  # Very small for quick demo
        val_samples=20,
        input_size=224
    )

    # Initialize models (no pretrained weights for speed)
    teacher = TeacherModel(num_classes=num_classes, pretrained=False)
    student = StudentModel(num_classes=num_classes, pretrained=False)

    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(
        teacher, student,
        device=device,
        save_dir='./demo_checkpoints',
        log_dir='./demo_logs'
    )

    # Simple optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

    print(f"\nDemo training for {num_epochs} epochs...")

    # Run training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        save_every=1,
        use_features=False
    )

    print("Demo completed successfully!")
    return trainer


if __name__ == "__main__":
    # Choose which function to run:

    # Option 1: Quick demo with minimal data (recommended for testing)
    print("Choose training mode:")
    print("1. Quick Demo (minimal data, 3 epochs)")
    print("2. Full Training (more data, 50 epochs)")

    choice = input("Enter choice (1 or 2), or press Enter for quick demo: ").strip()

    if choice == "2":
        main()
    else:
        # Run quick demo by default
        trainer = demo_quick_test()

        # Optionally show a sample prediction
        print("\n" + "=" * 50)
        print("Sample Model Output:")

        device = trainer.device
        sample_input = torch.randn(1, 3, 224, 224).to(device)

        trainer.student.eval()
        trainer.teacher.eval()

        with torch.no_grad():
            student_output = trainer.student(sample_input)
            teacher_output = trainer.teacher(sample_input)

            print(f"Student predictions: {F.softmax(student_output, dim=1)[0][:5]}")
            print(f"Teacher predictions: {F.softmax(teacher_output, dim=1)[0][:5]}")
            print(f"Student top class: {torch.argmax(student_output, dim=1).item()}")
            print(f"Teacher top class: {torch.argmax(teacher_output, dim=1).item()}")

        print("Demo completed! Check './demo_logs/training_history.png' for training plots.")