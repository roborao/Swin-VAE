import os
import nibabel as nib
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

def visualize_patient_slice(data_dir, patient_id, modality, slice_idx):
    """
    Visualize a specific slice of a specific patient's MRI scan.

    Args:
        data_dir (str): Directory containing patient data.
        patient_id (str): Identifier for the patient (e.g., "patient_1").
        modality (str): Modality to visualize (e.g., "flair", "t1", "t2", "t1ce").
        slice_idx (int): Slice index to visualize.
    """
    # Construct the path to the patient's modality file
    file_path = os.path.join(data_dir, patient_id, f"{patient_id}_{modality}.nii")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    else:
        print(f"Found the file: {file_path}")

    # Load the MRI scan
    img = nib.load(file_path)
    data = np.array(img.get_fdata())

    # Ensure the slice index is valid
    if slice_idx < 0 or slice_idx >= data.shape[-1]:
        print(f"Invalid slice index. Must be between 0 and {data.shape[-1] - 1}.")
        return

    # Visualize the slice
    pic = data[:, :, slice_idx]

    try:
        plt.figure(figsize=(15, 8))
        plt.imshow(pic, cmap='gray')
        plt.title(f"{patient_id} - {modality} - Slice {slice_idx}")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error occurred during visualization: {e}")


# Example usage
data_dir = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
patient_id = "BraTS20_Training_067"
modality = "t1"  # Options: "flair", "t1", "t2", "t1ce"
slice_idx = 88  # Specify the slice number

visualize_patient_slice(data_dir, patient_id, modality, slice_idx)

