import os
import nibabel as nib
import numpy as np
import slicevisualizer


# Paths
data_dir = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020_preprocessed/BraTS2020_TrainingData"  # Directory containing patient folders
healthy_dir = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020_preprocessed/BraTS2020_TrHealthy"  # Directory to save healthy slices
diseased_dir = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020_preprocessed/BraTS2020_TrDiseased"  # Directory to save diseased slices

# Ensure output directories exist
os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(diseased_dir, exist_ok=True)

# Process each patient folder
for patient_folder in os.listdir(data_dir):
    patient_path = os.path.join(data_dir, patient_folder)

    try:
        if os.path.isdir(patient_path):
            # Load segmentation mask
            seg_path = os.path.join(patient_path, f"{patient_folder}_seg.nii")
            if not os.path.exists(seg_path):
                print(f"Segmentation file not found for {patient_folder}, skipping.")
                continue

            seg = nib.load(seg_path).get_fdata()  # Shape: (H, W, Slices)

            # Load modalities
            modalities = ["flair", "t1", "t2"]  # Ignore t1ce
            modality_data = {}
            for modality in modalities:
                modality_path = os.path.join(patient_path, f"{patient_folder}_{modality}.nii")
                if not os.path.exists(modality_path):
                    print(f"Modality file {modality} not found for {patient_folder}, skipping.")
                    continue
                modality_data[modality] = nib.load(modality_path).get_fdata()  # Shape: (H, W, Slices)

            # Ensure all modalities have the same shape
            modality_shapes = [modality_data[mod].shape for mod in modalities]
            if len(set(modality_shapes)) != 1:
                print(f"Shape mismatch in modalities for {patient_folder}, skipping.")
                continue

            # Iterate through each slice
            for slice_idx in range(seg.shape[-1]):  # Last dimension is the slice index
                slice_mask = seg[:, :, slice_idx]  # Extract 2D slice

                # Combine modalities into a 3-channel slice
                combined_slice = np.stack([modality_data[mod][:, :, slice_idx] for mod in modalities], axis=2)  # Shape: (H, W, 3)
                slice_mask[slice_mask > 0 ] = 1
                # Check if the slice is healthy (no tumor regions) and save the slice
                if slice_mask.sum()==0:  # No tumor regions
                    #slicevisualizer.visualize_patient_slice(data_dir,patient_folder,"seg",slice_idx)
                    save_path = os.path.join(healthy_dir, f"{patient_folder}_slice{slice_idx}.npy")
                    np.save(save_path, combined_slice)
                elif slice_mask.sum()>=int(.05*224*224): # Diseased slice
                    save_path = os.path.join(diseased_dir, f"{patient_folder}_slice{slice_idx}.npy")
                    np.save(save_path, combined_slice)



        print(f"Saved {patient_folder}")
    except Exception as e:
        print(f"Error processing {patient_path}: {e}")

print(f"Healthy slices saved to {healthy_dir}")
print(f"Diseased slices saved to {diseased_dir}")