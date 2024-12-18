import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import SwinModel, SwinConfig
import os
import re
import random

# Define the VAE
class StandardVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(StandardVAE, self).__init__()

        # Encoder: Convolutional layers to extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Input: [3, 224, 224]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: [64, 56, 56]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: [128, 28, 28]
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened dimension for fully connected layers
        self.flattened_dim = 128 * (224 // (2 ** 3)) * (224 // (2 ** 3))  # Adjust for 3 conv layers

        # Latent space layers (mean and log variance)
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

        # Decoder: Fully connected and convolutional layers to reconstruct the image
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [64, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [32, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [3, 224, 224]
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def encode(self, x):
        # Pass through the encoder and calculate latent space parameters
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Reconstruct the image from the latent space
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 28, 28)  # Reshape for transposed conv layers
        x = self.decoder(x)
        return x

    def forward(self, x):
        # Perform encoding, reparameterization, and decoding
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def compute_dice(pred, target, threshold=0.5, smooth=1e-6):
    """
    Computes the Dice Similarity Coefficient (DSC) for NumPy arrays.

    Args:
        pred (np.ndarray): Anomaly map with continuous values.
        target (np.ndarray): Ground truth binary mask.
        threshold (float): Threshold to binarize the prediction.
        smooth (float): Smoothing term to avoid division by zero.

    Returns:
        float: Dice score.
    """
    # Binarize prediction using the threshold
    threshold = np.percentile(pred, 90)
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)

    # Flatten arrays
    pred_binary = pred_binary.flatten()
    target_binary = target_binary.flatten()

    # Compute the intersection and Dice score
    intersection = np.sum(pred_binary * target_binary)
    dice = (2. * intersection + smooth) / (np.sum(pred_binary) + np.sum(target_binary) + smooth)
    return dice

def compute_psnr(img1, img2, max_value=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1 (numpy.ndarray): First image (e.g., anomaly map).
        img2 (numpy.ndarray): Second image (e.g., segmentation mask).
        max_value (float): Maximum possible pixel value of the images.
    Returns:
        float: PSNR value.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr = 10 * np.log10(max_value**2 / mse)
    return psnr

def compute_reconstruction_error(original, reconstruction):
    """
    Compute the reconstruction error between the original image and its reconstruction.
    Args:
        original (numpy.ndarray): Original image.
        reconstruction (numpy.ndarray): Reconstructed image.
    Returns:
        float: Mean squared error (reconstruction error).
    """
    mse = np.mean((original - reconstruction) ** 2)
    return mse

def visualize_inference(reconstruction, original, anomaly, segmentation_mask, patient_num, slice_num, dice_score=0, output_dir='', idx=0, top=True):
    """
    Visualize original image, reconstruction, anomaly map, and optional segmentation mask and reconstruction error.

    Args:
        original (np.ndarray): The original input slice.
        reconstruction (np.ndarray): The reconstructed slice.
        segmentation_mask (np.ndarray, optional): The ground truth segmentation mask.
    """
    plt.figure(figsize=(20, 12))


    # Original slice
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Slice")

    # Reconstructed slice
    plt.subplot(2, 3, 2)
    plt.imshow(reconstruction, cmap='gray')
    plt.title("Reconstructed Slice")

    # Anomaly map
    plt.subplot(2, 3, 3)
    plt.imshow(anomaly, cmap='plasma')
    plt.title("Anomaly Map")

    # Segmentation mask
    if segmentation_mask is not None:
        plt.subplot(2, 3, 4)
        plt.imshow(segmentation_mask, cmap='gray')
        plt.title("Segmentation Mask")

    if not (output_dir=='' and dice_score==0 and idx==0):
        if (top):
            plt.savefig(os.path.join(output_dir, f"top_dice_{idx + 1}_score_{dice_score:.4f}.png"))
        else:
            plt.savefig(os.path.join(output_dir, f"bottom_dice_{idx + 1}_score_{dice_score:.4f}.png"))
    plt.suptitle(f"Patient {patient_num}, slice {slice_num}")
    plt.show()


def inference(model, test_path, label_dir, n=2000, top_k=5, visualize=False):
    """
    Perform inference on test samples, match them with segmentation labels, and optionally visualize results.

    Args:
        model (torch.nn.Module): Trained VAE model.
        test_samples (list of str): List of paths to test .npy files.
        label_dir (str): Path to the directory containing segmentation labels.
        threshold (float): Threshold for anomaly detection.
        visualize (bool): Whether to visualize the results.
        device (str): Device to run inference on ('cpu' or 'cuda').
    """
    model.eval()
    dice_scores = []
    psnr_scores = []
    mses = []
    top_dice_scores = []
    top_dice_images = []
    bottom_dice_scores = []
    bottom_dice_images = []

    random_n_samples = random.sample(os.listdir(test_path), n)
    for filename in random_n_samples:
        # Extract patient and slice info from the file name
        sample_path = os.path.join(test_path, filename)
        parts = filename.split('_')
        patient_num = parts[2]
        slice = parts[3].replace('.npy', '')
        slice_num = int(re.search(r'\d+', slice).group())

        # Load input features
        tests = np.load(sample_path)
        tests = np.transpose(tests, (2, 0, 1))  # Ensure correct shape [C, H, W]
        tests = torch.tensor(tests, dtype=torch.float32).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            recon, _, _ = model(tests)

        # Prepare for visualization
        original = tests.squeeze()[0].cpu().numpy().reshape(224, 224)
        #replace = np.percentile(original, 95)
        #original[original > np.percentile(original, 95)] = replace
        #original = (original - original.min()) / (original.max() - original.min())
        reconstruction = recon.squeeze()[0].cpu().numpy().reshape(224, 224)
        #reconstruction = (reconstruction - original.min())/(original.max() - original.min())

        # Match and load corresponding segmentation label
        label_path = os.path.join(label_dir, f"BraTS20_Training_{patient_num}",
                                  f"BraTS20_Training_{patient_num}_seg.nii")
        segmentation_mask = None
        if os.path.exists(label_path):
            seg = nib.load(label_path).get_fdata()
            slice_index = slice_num - 1  # Convert to 0-based index
            segmentation_mask = seg[:, :, slice_index]
            segmentation_mask[segmentation_mask > 0] = 1  # Binary segmentation mask
        if segmentation_mask.sum()==0:
            continue

        anomaly = original - reconstruction
        anomaly[anomaly < 0] = 0
        #anomaly = (anomaly - anomaly.min())/(anomaly.max()-anomaly.min())
        anomaly[anomaly < np.percentile(anomaly, 90)] = 0

        mse = compute_reconstruction_error(anomaly, segmentation_mask)
        mses.append(mse)
        psnr = compute_psnr(anomaly, segmentation_mask)
        psnr_scores.append(psnr)
        dice = compute_dice(anomaly, segmentation_mask)
        dice_scores.append(dice)

        #if (dice < 0.2):
            #visualize_inference(reconstruction, original, anomaly, segmentation_mask, patient_num, slice_num)

        # Save the top k images based on Dice score
        if len(top_dice_scores) < top_k:
            top_dice_scores.append(dice)
            top_dice_images.append((original, reconstruction, anomaly, segmentation_mask, patient_num, slice_num))
        else:
            min_dice_idx = np.argmin(top_dice_scores)
            if dice > top_dice_scores[min_dice_idx]:
                top_dice_scores[min_dice_idx] = dice
                top_dice_images[min_dice_idx] = (original, reconstruction, anomaly, segmentation_mask, patient_num, slice_num)

        # Save the bottom k images based on Dice score
        if len(bottom_dice_scores) < top_k:
            bottom_dice_scores.append(dice)
            bottom_dice_images.append((original, reconstruction, anomaly, segmentation_mask, patient_num, slice_num))
        else:
            max_dice_idx = np.argmax(bottom_dice_scores)
            if dice < bottom_dice_scores[max_dice_idx]:
                bottom_dice_scores[max_dice_idx] = dice
                bottom_dice_images[max_dice_idx] = (original, reconstruction, anomaly, segmentation_mask, patient_num, slice_num)


    # Save top k images based on Dice score
    output_dir = "D:/robor/Documents/ViT-VAE BRATS anomaly map"
    sorted_indices = np.argsort(top_dice_scores)[::-1]
    for idx, i in enumerate(sorted_indices):
        original, reconstruction, anomaly, segmentation_mask, patient_num, slice_num = top_dice_images[i]
        visualize_inference(reconstruction, original, anomaly, segmentation_mask, patient_num, slice_num, top_dice_scores[i], output_dir, idx)

    sorted_indices = np.argsort(bottom_dice_scores)[::-1]
    for idx, i in enumerate(sorted_indices):
        original, reconstruction, anomaly, segmentation_mask, patient_num, slice_num  = bottom_dice_images[i]
        visualize_inference(reconstruction, original, anomaly, segmentation_mask, patient_num, slice_num, bottom_dice_scores[i], output_dir, idx, top=False)


    return np.array(dice_scores), np.array(psnr_scores), np.array(mses)

# Define your VAE architecture
latent_dim = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
model = StandardVAE(latent_dim=latent_dim).to(device)

# Load the trained model weights
model.load_state_dict(torch.load('D:/robor/Documents/ViT-VAE BRATS anomaly map/results/standardvae_weights.pth', weights_only=True))

test_path = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020_preprocessed/BraTS2020_TrDiseased/"
labels_path = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020_preprocessed/BraTS2020_TrainingData"
dice_scores, psnr_scores, mses = inference(model, test_path, labels_path, n=500)

print(f"Mean Dice Score: {np.mean(dice_scores):.2f}")
print(f"Median Dice Score: {np.median(dice_scores):.2f}")
print(f"Dice Standard Deviation: {np.std(dice_scores):.2f}\n")

print(f"Mean PSNR Score: {np.mean(psnr_scores):.2f}")
print(f"Median PSNR Score: {np.median(psnr_scores):.2f}")
print(f"PSNR Standard Deviation: {np.std(psnr_scores):.2f}\n")

print(f"Mean MSE Score: {np.mean(mses):.2f}")
print(f"Median MSE Score: {np.median(mses):.2f}")
print(f"MSE Standard Deviation: {np.std(mses):.2f}")

