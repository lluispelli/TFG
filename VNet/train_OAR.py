    # -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:23:24 2023

@author: Adria
"""
#Setup imports
import pdb

import numpy as np
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,          # need to search
    AsDiscreted,         # version dictionary of the above
    EnsureChannelFirstd, # ensures the original data to construct "channel first" shape.
    Compose,             # to apply several transformations at once
    CropForegroundd,     # removes all zero borders to focus on the valid body area of the images and labels.
    LoadImaged,          # loads the spleen CT images and labels from NIfTI format files.
    Orientationd,        # the data orientation based on the affine matrix.
    RandCropByPosNegLabeld, # randomly crop patch samples from big image based on pos / neg ratio.
    SaveImaged,             # saves image
    ScaleIntensityRanged,   # extracts intensity range [-57, 164] and scales to [0, 1].
    Spacingd,               # adjusts the spacing by pixdim=(1.5, 1.5, 2.) based on the affine matrix.
    Invertd,
    RandGaussianNoised,
    RandAffined,
    RandSmoothDeformd,
    RandSpatialCropd,
    Resized
)
from monai.handlers.utils import from_engine
from monai.networks.nets import VNet
from monai.networks.layers import Norm
from monai.metrics import (
    DiceMetric,
    VarianceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    MSEMetric,
    MAEMetric,
    RMSEMetric,
    ConfusionMatrixMetric)
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch #to accelerate training and validation process, it's 10x faster than the regular Dataset.
                                                                          #To achieve best performance, set cache_rate=1.0 to cache all the data
                                                                          # , if memory is not enough,set lower value.
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
from os import listdir, makedirs
from os.path import join, exists, dirname
import glob
import gc
from sklearn.model_selection import train_test_split
import nibabel as nib

from src.training import Segmentation, Segmentation2Chan

print_config()
#%%
#Setup data directory
"""
You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
This allows you to save results and reuse downloads.
If not specified a temporary directory will be used.
"""

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
results_dir = "/home/biof01/lluis_tfg/results/oar_aligned_v0"
if not exists(results_dir):
    makedirs(results_dir)
training_file = join(results_dir, 'train_logs.csv')
val_file = join(results_dir, 'val_logs.csv')

# Setting dataset directory
data_dir = "/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned"
max_epochs = 5000
val_interval = 5
num_classes = 5

#%%
# Get all subjects
subjects_excluded = ['sub-215']
# sub-215: wrong intensity values range=[8000-9000]
subject_list = [sbj for sbj in listdir(data_dir) if int(sbj[4:]) < 220 and sbj not in subjects_excluded]

# Filter only those with all classes available
subject_label_list = {sbj: join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-oar_dseg.nii.gz') for sbj in subject_list}
subject_list_nlabels = {sbj: np.unique(np.array(nib.load(f).dataobj)) for sbj, f in subject_label_list.items()}
subject_list = list(filter(lambda sbj: len(subject_list_nlabels[sbj]) == num_classes, subject_list))


# Train/Val split. Set random_state=14 for reproducibility
subject_list_train_val, subject_list_test = train_test_split(subject_list, test_size=0.2, train_size=0.8, random_state=14)
subject_list_train, subject_list_val = train_test_split(subject_list_train_val, test_size=0.2, train_size=0.8, random_state=14)

# import subprocess
# images = {sbj: join('/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/', 'nifti', sbj, 'ses-0', 'anat', sbj + '_ses-0_CT.nii.gz') for sbj in subject_list}
# wrong_subjects = []# ['sub-179', 'sub-180', 'sub-197', 'sub-210', 'sub-193', 'sub-186', 'sub-214', 'sub-208', 'sub-184', 'sub-198', 'sub-202', 'sub-187', 'sub-206', 'sub-189', 'sub-195', 'sub-194', 'sub-190', 'sub-183', 'sub-218', 'sub-188', 'sub-196', 'sub-199', 'sub-203', 'sub-200', 'sub-205', 'sub-204', 'sub-192', 'sub-182', 'sub-181', 'sub-191']
#                    #'/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-179/ses-0/anat/sub-179_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-180/ses-0/anat/sub-180_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-197/ses-0/anat/sub-197_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-210/ses-0/anat/sub-210_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-193/ses-0/anat/sub-193_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-186/ses-0/anat/sub-186_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-214/ses-0/anat/sub-214_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-208/ses-0/anat/sub-208_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-184/ses-0/anat/sub-184_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-202/ses-0/anat/sub-202_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-187/ses-0/anat/sub-187_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-206/ses-0/anat/sub-206_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-189/ses-0/anat/sub-189_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-195/ses-0/anat/sub-195_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-194/ses-0/anat/sub-194_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-183/ses-0/anat/sub-183_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-218/ses-0/anat/sub-218_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-188/ses-0/anat/sub-188_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-196/ses-0/anat/sub-196_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-199/ses-0/anat/sub-199_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-203/ses-0/anat/sub-203_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-200/ses-0/anat/sub-200_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-205/ses-0/anat/sub-205_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-192/ses-0/anat/sub-192_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-182/ses-0/anat/sub-182_ses-0_space-TEMPLATE_CT.nii.gz', '/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned/sub-191/ses-0/anat/sub-191_ses-0_space-TEMPLATE_CT.nii.gz']
# for sbj, im_path in images.items():
#     proxy = nib.load(im_path)
#     data = np.array(proxy.dataobj)
#     if np.min(data > 0):
#         path = join('/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/dicom', sbj, 'nifti', 'image.nii.gz')
#         subprocess.call(['rm', '-rf', im_path])
#         subprocess.call(['cp', path, im_path])
#         wrong_subjects.append(sbj)
# pdb.set_trace()

# Get filepath to train/val volumes.
train_images = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_CT.nii.gz') for sbj in subject_list_train]
train_labels = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-oar_dseg.nii.gz') for sbj in subject_list_train]
train_vagina = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-vagina_dseg.nii.gz') for sbj in subject_list_train]
val_images = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_CT.nii.gz') for sbj in subject_list_val]
val_labels = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-oar_dseg.nii.gz') for sbj in subject_list_val]
val_vagina = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-vagina_dseg.nii.gz') for sbj in subject_list_val]

# Build dictionaries as accepted by MONAI loaders.
train_files = [{"image": image_name, "label": label_name, "vagina": vagina_name, "id": image_name} for image_name, label_name, vagina_name in zip(train_images, train_labels, train_vagina)]
val_files = [{"image": image_name, "label": label_name,  "vagina": vagina_name, "id": image_name} for image_name, label_name, vagina_name in zip(val_images, val_labels, val_vagina)]

# Set deterministic training for reproducibility
set_determinism(seed=0)

################
## Transforms ##
################
# Setup transforms for training and validation
spatial_size = (256, 256, 64)
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label", "vagina"]),
        EnsureChannelFirstd(keys=["image", "label", "vagina"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-800,
            a_max=500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "label", "vagina"], axcodes="RAS"), #Change the input image’s orientation into the specified based on axcodes.
        RandSpatialCropd(
            keys=["image", "label", "vagina"],
            roi_size=spatial_size,
            random_size=False
        ),
        # RandAffined(
        #     keys=["image", "label", "vagina"], spatial_size=None, prob=0.1, rotate_range=[15/180*np.pi]*3, shear_range=[0.01]*6,
        #     translate_range=[3]*3, scale_range=[0.1]*3, mode=["bilinear", 'nearest']
        # ),
        # RandSmoothDeformd(
        #     keys=["image", "label", "vagina"], spatial_size=spatial_size, def_range=0.1,
        #     rand_size=tuple([int(s*0.04) for s in spatial_size]),
        # ),
        # RandGaussianNoised(
        #     keys=["image", "vagina"], prob=0.4, mean=0.0, std=0.1
        # )
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label", "vagina"]),
        EnsureChannelFirstd(keys=["image", "label", "vagina"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-800,
            a_max=500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "label", "vagina"], axcodes="RAS"),
        RandSpatialCropd(
            keys=["image", "label", "vagina"],
            roi_size=spatial_size,
            random_size=False
        ),
    ]
)


##################
## Data loaders ##
##################
# Define CacheDataset and DataLoader for training and validation
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1., num_workers=4)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)


###########
## Model ##
###########
# Create Model, Loss, Optimizer and metrics
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=num_classes,
    ).to(device)

loss_function = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
metrics = {
    'dice': DiceMetric(include_background=False, reduction="mean"),
    'hd': HausdorffDistanceMetric(include_background=False, reduction="mean")
}

##############
## Training ##
##############


training_scheme = Segmentation2Chan(max_epochs, results_dir, device)
training_scheme.build_train(train_loader, model, optimizer, loss_function, val_loader, val_interval, metrics, training_file, val_file)
log_dict = training_scheme.train()
