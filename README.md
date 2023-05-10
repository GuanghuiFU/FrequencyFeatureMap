# Import necessary libraries
import numpy as np
import nibabel as nib
import torch
import torchvision.transforms as transforms
from lib.medzoo.Unet3D import UNet3D
from torchvision.models import vgg16
from PIL import Image
import feature_map_visualization as fv


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
    model = UNet3D(in_channels=1, n_classes=2, base_n_filter=8)
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
