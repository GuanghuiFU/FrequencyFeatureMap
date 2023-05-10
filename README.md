# Frequency Feature Map Visualization for 3D and 2D Models

This code is designed to visualize and save the feature maps of 3D and 2D models such as UNet and VGG. The feature maps can be viewed in the image domain and frequency domain, and saved as `.npy` files. We provide examples of feature map visualization for 3D segmentation network UNet and 2D classification network VGG, and plot feature maps save as `.png`.

## Dependencies

- numpy
- nibabel
- torch
- torchvision
- PIL (Pillow)
- matplotlib

## Functions

### Data Processing

- `image_path2tensor(path)`: Preprocesses an image, converting it to a tensor
- `path2nib(path)`: Loads a NIfTI file using nibabel
- `path2np(path)`: Converts a NIfTI file to a numpy array and returns its affine transformation
- `path2tensor(path, type)`: Converts a NIfTI file to a PyTorch tensor
- `save_nii(feature_map, affine, save_path)`: Saves a feature map as a NIfTI file
- `normalize_np(mri_np)`: Normalizes a numpy array
- `feature_map_np_fourier_transform(feature_map_np)`: Computes the Fourier transform of a feature map in numpy format

### Feature Map Extraction

- `extract_all_layers(model)`: Extracts all layers from a PyTorch model
- `extract_feature_maps(layer_outputs)`: Extracts feature maps from a list of layer outputs
- `visualize_feature_maps_3d(model, input_tensor, device)`: Visualizes feature maps for a 3D segmentation model
- `visualize_feature_maps_2d(model, input_tensor, device)`: Visualizes feature maps for a 2D classification model

### Saving and Plotting Feature Maps

- `save_feature_maps_to_npy(feature_map_dict, save_base_path)`: Saves feature maps to `.npy` files
- `get_min_max_from_npy_files(path)`: Calculates the minimum and maximum values from a set of `.npy` files
- `adjust_range(min_val, max_val, percentage)`: Adjusts the range of an array based on a percentage
- `plot_feature_map_3d(base_path, clim_ranges, percen_ranges, slice_no)`: Plots feature maps for a 3D segmentation model
- `plot_feature_map_2d(base_path, clim_ranges, percen_ranges, slice_no)`: Plots feature maps for a 2D classification model

## Main Functions (Usage Example)

- `main_segmentation()`: Applies feature map visualization for a 3D segmentation model (e.g., UNet)
- `main_classification()`: Applies feature map visualization for a 2D classification model (e.g., VGG)

## Usage

To run the feature map visualization for a 2D classification model such as VGG, simply run the `main_2d()` function. For a 3D segmentation model like UNet, use the `main_3d()` function.

```python
# Import necessary libraries
import numpy as np
import nibabel as nib
import torch
import torchvision.transforms as transforms
from lib.medzoo.Unet3D_3 import UNet3D_3
from torchvision.models import vgg16
from PIL import Image
import feature_visualization_frequency.feature_map_visualization_open as fv


# Function to convert image path to a preprocessed tensor
def image_path2tensor(path):
    # Load image using PIL.Image
    image = Image.open(path)
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Preprocess the image
    input_tensor = preprocess(image)
    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


# Function to convert 3D nifti volume file path to a tensor
def mri_path2tensor(path):
    # Load MRI volume using nibabel
    mri_nii = nib.load(path)
    # Convert to numpy array
    mri_np = np.asarray(mri_nii.get_fdata(dtype=np.float32))
    # Convert numpy array to PyTorch tensor
    mri_tensor = torch.from_numpy(mri_np)
    return mri_tensor


# Function for visualizing feature maps of a 3D MRI volume
def main_3d():
    # Define input and model paths
    input_path = 'path/to/your/mri.nii.gz'
    pth_path = 'path/to/your/model.pth'
    feature_map_img_save_path = 'path/to/your/save_path'

    # Load and process MRI tensor
    mri_tensor = mri_path2tensor(input_path)
    mri_tensor = mri_tensor.view(1, 1, *mri_tensor.size())

    # Load the model and move to GPU
    model = UNet3D_3(in_channels=1, n_classes=2, base_n_filter=8)
    model.restore_checkpoint(pth_path)
    model = model.cuda()

    # Visualize and save feature maps
    feature_map_dict = fv.visualize_feature_maps_3d(model, mri_tensor, device=torch.device('cuda'))
    fv.save_feature_maps_to_npy(feature_map_dict, feature_map_img_save_path)

    # Define color limits for visualization and visualize feature maps
    # If you don't define the color limits, the function will automatically calculate it across all the files.
    clim_ranges = [0.4, 0.6]
    fv.plot_feature_map_3d(feature_map_img_save_path, clim_ranges=clim_ranges, percen_ranges=0.3, slice_no='avg')


# Function for visualizing feature maps of a 2D image
def main_2d():
    # Define input path and feature map save path
    input_path = 'path/to/your/img.png'
    feature_map_img_save_path = 'path/to/your/save_path'

    # Load the pretrained VGG16 model and move to GPU
    model = vgg16(pretrained=True).cuda()

    # Load and process input image tensor
    img_tensor = image_path2tensor(input_path).cuda()

    # Visualize and save feature maps
    feature_map_dict = fv.visualize_feature_maps_2d(model, img_tensor, device=torch.device('cuda'))
    fv.save_feature_maps_to_npy(feature_map_dict, feature_map_img_save_path)
    # Define color limits for visualization and visualize feature maps. 
    # If you don't define the color limits, the function will automatically calculate it across all the files.
    clim_ranges = [0.4, 0.6]
    fv.plot_feature_map_2d(feature_map_img_save_path, clim_ranges=clim_ranges, percen_ranges=0.3)


if __name__ == 'main':
    # Run 2D feature map visualization
    main_2d()
    # Run 3D feature map visualization
    main_3d()
```

## Function Summary

Here is the table summarizing the functions and their input types:

| Function                                   | Input Type                                            | Summary                                                                      |
| ------------------------------------------ | ----------------------------------------------------- | ---------------------------------------------------------------------------- |
| image_path2tensor(path)                    | path: str                                             | Converts an image at the specified path into a PyTorch tensor.               |
| path2nib(path)                             | path: str                                             | Loads a NIfTI image from the specified path using the nibabel library.       |
| path2np(path)                              | path: str                                             | Returns the NumPy array representation of the image and its affine matrix.   |
| path2tensor(path, type)                    | path: str, type: str                                  | Converts a NIfTI image into a PyTorch tensor.                               |
| save_nii(feature_map, affine, save_path)   | feature_map: np.ndarray, affine: np.ndarray, save_path: str | Saves the feature map as a NIfTI file with the given affine and save_path.  |
| extract_all_layers(model)                  | model: nn.Module                                      | Extracts all the layers in the given PyTorch model.                         |
| normalize_np(mri_np)                       | mri_np: np.ndarray                                    | Normalizes the given MRI NumPy array.                                       |
| feature_map_np_fourier_transform(feature_map_np) | feature_map_np: np.ndarray                        | Computes the normalized Fourier transform of the given feature map.          |
| extract_feature_maps(layer_outputs)        | layer_outputs: List[Tuple[torch.Tensor, str]]         | Extracts feature maps and their names from the given layer outputs.         |
| visualize_feature_maps_3d(model, input_tensor, device) | model: nn.Module, input_tensor: torch.Tensor, device: torch.device | Generates a dictionary containing the feature maps and their names for a 3D model. |
| visualize_feature_maps_2d(model, input_tensor, device) | model: nn.Module, input_tensor: torch.Tensor, device: torch.device | Generates a dictionary containing the feature maps and their names for a 2D model. |
| save_feature_maps_to_npy(feature_map_dict, save_base_path) | feature_map_dict: Dict[int, Dict[str, np.ndarray]], save_base_path: str | Saves the feature maps to npy files. |
| get_min_max_from_npy_files(path)           | path: str                                             | Returns the minimum and maximum values found in all npy files within the given path. |
| adjust_range(min_val, max_val, percentage) | min_val: float, max_val: float, percentage: float    | Adjusts the range of values by the given percentage.                         |
| plot_feature_map_3d(base_path, clim_ranges, percen_ranges, slice_no) | base_path: str, clim_ranges: List[float], percen_ranges: float, slice_no: Union[str, int] | Plots the 3D feature maps, saving them as images. |
| plot_feature_map_2d(base_path, clim_ranges, percen_ranges, slice_no) | base_path: str, clim_ranges: List[float], percen_ranges: float, slice_no: Union[str, int] | Plots the 2D feature maps, saving them as images. |      
