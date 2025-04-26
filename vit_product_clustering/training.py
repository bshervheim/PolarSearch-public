import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import json

from .model import ViTContrastive
from .loss import SupConLoss
from .utils import create_dataloader

def train_model(
    dataset,
    output_dir=None,
    epochs=None,
    batch_size=None,
    learning_rate=2e-5,
    temperature=0.1,
    weight_decay=0.01,
    use_amp=None,
    num_workers=None
):
    """
    Train a ViT model with supervised contrastive learning
    
    Args:
        dataset: Dataset object with images and set_ids
        output_dir (str): Directory to save models and logs
        epochs (int): Number of training epochs (auto-calculated if None)
        batch_size (int): Batch size (auto-calculated if None)
        learning_rate (float): Initial learning rate
        temperature (float): Temperature for contrastive loss
        weight_decay (float): Weight decay for optimizer
        use_amp (bool): Whether to use automatic mixed precision
        num_workers (int): Number of data loading workers
        
    Returns:
        dict: Training metadata including best model path and loss
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory with timestamp if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"vit_supcon_model_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-determine batch size based on GPU if not provided
    if batch_size is None:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem > 16:
                batch_size = 64
            elif gpu_mem > 8:
                batch_size = 32
            else:
                batch_size = 16
        else:
            batch_size = 8
    
    # Auto-determine number of workers if not provided
    if num_workers is None:
        num_workers = 4 if torch.cuda.is_available() else 2
    
    # Create data loader
    train_loader = create_dataloader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Auto-determine number of epochs based on dataset size if not provided
    if epochs is None:
        num_samples = len(dataset)
        if num_samples < 1000:
            epochs = 30
        elif num_samples < 5000:
            epochs = 20
        else:
            epochs = 10
    
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # Create model
    model = ViTContrastive().to(device)
    
    # Determine whether to use mixed precision
    if use_amp is None:
        use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate/20
    )
    
    # Loss function
    criterion = SupConLoss(temperature=temperature)
    
    # Training loop
    best_loss = float('inf')
    best_model_path = None
    training_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        # Progress bar
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Mixed precision training
            with torch.autocast(device.type, enabled=use_amp):
                embeddings = model(images)
                loss = criterion(embeddings, labels)
            
            # Optimization step
            optimizer.zero_grad()
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update running loss and progress bar
            running_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{running_loss/(batch_idx+1):.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
            })
            
            # Clear cache every few batches
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s. Loss: {epoch_loss:.6f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(output_dir, 'vit_supcon_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.6f}")
        
        # Save checkpoint at final epoch
        if epoch == epochs - 1:
            final_model_path = os.path.join(output_dir, 'vit_supcon_final.pth')
            torch.save(model.state_dict(), final_model_path)
            print(f"Final model saved to {final_model_path}")
    
    # Training summary
    training_time = time.time() - training_start
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best loss: {best_loss:.6f}")
    
    # Save training metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(dataset),
        'num_sets': len(dataset.sets),
        'epochs': epochs,
        'batch_size': batch_size,
        'final_loss': epoch_loss,
        'best_loss': best_loss,
        'training_time_seconds': training_time,
        'model': 'ViT with supervised contrastive learning',
        'embedding_dim': 128,
        'use_amp': use_amp,
        'best_model_path': best_model_path
    }
    
    metadata_path = os.path.join(output_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to {metadata_path}")
    
    return {
        'model': model,
        'best_loss': best_loss,
        'best_model_path': best_model_path,
        'metadata': metadata
    }
