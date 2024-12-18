import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig
from torch.utils.data import random_split, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import os
import time


# Define the Swin-VAE
class Swin_VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(Swin_VAE, self).__init__()

        # Load pre-trained Swin Transformer (SwinModel for feature extraction)
        swin_config = SwinConfig.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        self.swin_encoder = SwinModel(swin_config)

        # Latent space layers (mean and log variance)
        self.fc_mu = nn.Linear(swin_config.hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(swin_config.hidden_size, latent_dim)

        # Map latent vector to a feature map
        self.fc_to_feature = nn.Linear(latent_dim, 256 * 7 * 7)  # Match dimensions for 256 channels

        # Decoder network with transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample to 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Upsample to 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Upsample to 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # Upsample to 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),     # Upsample to 224x224
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def encode(self, x):
        # Get features from Swin (SwinModel for feature extraction)
        swin_out = self.swin_encoder(x).last_hidden_state  # Access the last hidden states

        # Use the [CLS] token's representation
        cls_token = swin_out[:, 0, :]  # Extract the feature corresponding to the [CLS] token

        # Pass through the fully connected layers to get mean (mu) and log variance (logvar)
        mu = self.fc_mu(cls_token)
        logvar = self.fc_logvar(cls_token)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from the latent space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Map latent vector to feature map
        features = self.fc_to_feature(z).view(-1, 256, 7, 7)  # Reshape to match decoder input with 256 channels
        # Decode to reconstruct the image
        return self.decoder(features)

    def forward(self, x):
        # Perform encoding, reparameterization, and decoding
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def beta_schedule(epoch, start_beta=0.1, max_beta=1, warmup_epochs=10):
    """
    Linearly increases beta over the first `warmup_epochs`.

    Parameters:
    - epoch: Current epoch.
    - start_beta: Initial beta value.
    - max_beta: Maximum beta value.
    - warmup_epochs: Number of epochs over which beta increases.

    Returns:
    - Beta value for the current epoch.
    """
    if epoch < warmup_epochs:
        return start_beta + (max_beta - start_beta) * (epoch / warmup_epochs)
    else:
        return max_beta

def vae_loss(recon_x, x, mu, logvar, beta=1):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

class HealthyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory containing the .npy files
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        # List all .npy files in the directory
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        # The length of the dataset is the number of .npy files
        return len(self.files)

    def __getitem__(self, idx):
        # Get the file path for the given index
        slice_file = self.files[idx]
        slice_path = os.path.join(self.data_dir, slice_file)

        # Load the .npy file (the brain slice)
        slice_image = np.load(slice_path)
        slice_image = np.transpose(slice_image, (2, 0, 1))

        # Convert to a tensor
        slice_image = torch.tensor(slice_image, dtype=torch.float32)

        # The filename contains the slice number and patient ID, but we can ignore labels for now
        return slice_image


# Create the DataLoader
healthy_features_dir = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020_preprocessed/BraTS2020_TrHealthy"  # Specify the path to your extracted healthy features
healthy_dataset = HealthyDataset(healthy_features_dir)
split_ratio = 0.8
batch_size = 32

# Split dataset into train and validation
total_samples = len(healthy_dataset)
train_size = int(split_ratio * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(healthy_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and device
latent_dim = 256
lr=1e-5
wd=1e-5
step_size=5
gamma=0.15
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

vae = Swin_VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training loop
epochs = 25
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

vae.train()
for epoch in range(epochs):
    start_time = time.time()  # Start time for the epoch

    total_train_loss = 0
    total_train_accuracy = 0
    total_val_loss = 0
    total_val_accuracy = 0
    beta = beta_schedule(epoch)

    # Training phase
    vae.train()
    for batch in train_loader:
        batch = batch.to(device)

        # Forward pass
        reconstructed, mu, logvar = vae(batch)

        # Compute custom VAE loss
        loss = vae_loss(reconstructed, batch, mu, logvar, beta)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Compute accuracy-like metric (based on MSE between reconstructed and original batch)
        mse_loss = nn.MSELoss()(reconstructed, batch).item()
        accuracy = 1 / (1 + mse_loss)  # Higher is better
        total_train_accuracy += accuracy

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_accuracy = total_train_accuracy / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    # Validation phase
    vae.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            reconstructed, mu, logvar = vae(batch)

            # Compute custom VAE loss
            loss = vae_loss(reconstructed, batch, mu, logvar, beta)
            total_val_loss += loss.item()

            # Compute accuracy-like metric (based on MSE between reconstructed and original batch)
            mse_loss = nn.MSELoss()(reconstructed, batch).item()
            accuracy = 1 / (1 + mse_loss)  # Higher is better
            total_val_accuracy += accuracy

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_accuracy = total_val_accuracy / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)

    # Step the scheduler
    scheduler.step()

    # Calculate the time for this epoch
    epoch_time = time.time() - start_time  # Time in seconds

    # Print losses, accuracies, and time for the epoch
    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}, Time: {epoch_time:.2f} sec")

torch.save(vae.state_dict(), 'D:/robor/Documents/ViT-VAE BRATS anomaly map/results/swinvae_weights2.pth')
np.save('D:/robor/Documents/ViT-VAE BRATS anomaly map/results/train_losses.npy', np.array(train_losses))
np.save('D:/robor/Documents/ViT-VAE BRATS anomaly map/results/train_accuracies.npy', np.array(train_accuracies))
np.save('D:/robor/Documents/ViT-VAE BRATS anomaly map/results/val_losses.npy', np.array(val_losses))
np.save('D:/robor/Documents/ViT-VAE BRATS anomaly map/results/val_accuracies.npy', np.array(val_accuracies))

# Optionally, plot the losses and accuracies
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss, LR = {lr}')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Training and Validation Accuracy, LR = {lr}')
plt.legend()

plt.tight_layout()
plt.show()
