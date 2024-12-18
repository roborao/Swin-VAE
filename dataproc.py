import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import nibabel as nib
import os

def preprocess_image(image, target_shape, crop_values, slice_range=None, is_label=False):
    """
    Preprocess a 3D image by resizing slices and cropping to a specified range.

    Parameters:
        image (numpy.ndarray): Input 3D image.
        target_shape (tuple): Desired output shape for each slice (e.g., (224, 224)).
        slice_range (tuple): Range of slices to retain (e.g., (40, 125)).
        crop_values (dict): Dictionary specifying crop values {top, bottom, left, right}.
        is_label (bool): If True, use nearest-neighbor interpolation for labels.

    Returns:
        numpy.ndarray: Preprocessed 3D image with resized slices and cropped range.
    """
    # Crop the image
    top, bottom, left, right = crop_values["top"], crop_values["bottom"], crop_values["left"], crop_values["right"]
    cropped_image = image[top:image.shape[0] - bottom, left:image.shape[1] - right, :]

    # Retain only the specified slice range
    if slice_range:
        cropped_image = cropped_image[:, :, slice_range[0]:slice_range[1]]

    # Resize each slice to the target shape
    zoom_factors = (target_shape[0] / cropped_image.shape[0], target_shape[1] / cropped_image.shape[1], 1)
    order = 0 if is_label else 1  # Nearest-neighbor for labels, linear for images
    resized_image = zoom(cropped_image, zoom_factors, order=order)

    # Normalize intensity values for images (not for labels)
    if not is_label:
        resized_image = (resized_image - np.min(resized_image)) / (np.max(resized_image) - np.min(resized_image))

    return resized_image

def preprocess_and_save(data_dir, save_dir, target_shape=(224, 224), slice_range=(40, 125), crop_values=None):
    """
    Preprocess BraTS .nii.gz files for ViT and save preprocessed images as .nii.gz.

    Parameters:
        data_dir (str): Path to the dataset directory containing patient folders.
        save_dir (str): Path to save the preprocessed .nii.gz files.
        target_shape (tuple): Desired shape for each 2D slice (e.g., (224, 224)).
        slice_range (tuple): Range of slices to retain (e.g., (40, 125)).
        crop_values (dict): Dictionary specifying crop values {top, bottom, left, right}.
    """
    if crop_values is None:
        crop_values = {
            "top": 20,
            "bottom": 20,
            "left": 20,
            "right": 20
        }
    os.makedirs(save_dir, exist_ok=True)
    patients = os.listdir(data_dir)

    for patient in patients:
        patient_path = os.path.join(data_dir, patient)
        if os.path.isdir(patient_path):
            save_patient_path = os.path.join(save_dir, patient)
            os.makedirs(save_patient_path, exist_ok=True)

            # Process modalities
            modalities = ["flair", "t1", "t1ce", "t2", "seg"]
            for modality in modalities:
                file_path = os.path.join(patient_path, patient+f"_{modality}.nii")
                save_path = os.path.join(save_patient_path, patient+f"_{modality}.nii")

                # Load image
                nii = nib.load(file_path)
                image_data = nii.get_fdata()

                # Preprocess image
                is_label = (modality == "seg")
                preprocessed_data = preprocess_image(
                    image_data, target_shape, crop_values, slice_range=slice_range, is_label=is_label
                )

                # Save as new .nii.gz
                new_nii = nib.Nifti1Image(preprocessed_data, affine=nii.affine, header=nii.header)
                nib.save(new_nii, save_path)

                print(f"Processed and saved: {save_path}")

# Example usage
original_data_dir = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
preprocessed_data_dir = "D:/robor/Documents/ViT-VAE BRATS anomaly map/brats2020_preprocessed/BraTS2020_TrainingData"
cropval = {
    "top": 20,
    "bottom": 20,
    "left": 20,
    "right": 20
}
preprocess_and_save(
    original_data_dir,
    preprocessed_data_dir,
    target_shape=(224, 224),
    slice_range=(50, 125),
    crop_values=cropval
)
