# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:23:24 2023

@author: Adria
"""
#Setup imports
import pdb

import pandas as pd
import numpy as np
from skimage import measure
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
    DivisiblePadd,             # makes image divisible by L
    ScaleIntensityRanged,   # extracts intensity range [-57, 164] and scales to [0, 1].
    Spacingd,               # adjusts the spacing by pixdim=(1.5, 1.5, 2.) based on the affine matrix.
    Invertd,
    RandSpatialCropd,
    CenterSpatialCropd,
    DivisiblePadd
)
from monai.handlers.utils import from_engine
from monai.networks.nets import VNet
from monai.networks.layers import Norm
from monai.metrics import (
    SurfaceDistanceMetric,
    DiceMetric,
    VarianceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    MSEMetric,
    MAEMetric,
    RMSEMetric,
    ConfusionMatrixMetric)
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference, SimpleInferer
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
from os.path import exists, dirname, join
import nibabel as nib
from sklearn.model_selection import train_test_split
from utils.io import plot_results



# Setting dataset directory
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
results_dir = "/home/biofisica/ADRIA/Results/V_OAR_3"
if not exists(results_dir):
    makedirs(results_dir)
train_logs_file = os.path.join(results_dir, 'train_logs.csv')
val_logs_file = os.path.join(results_dir, 'val_logs.csv')

# Setting dataset directory
data_dir = "/mnt/3997f083-96a1-40ec-a4eb-e1d6e8088a28/Data/RadioTH/aligned"

seg2chan = True
evaluation_flag = 'test'
max_epochs = 5000
val_interval = 5
num_classes = 5

subject_list = [sbj for sbj in listdir(data_dir) if int(sbj[4:]) < 220]
subject_label_list = {sbj: join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-oar_dseg.nii.gz') for sbj in subject_list}
subject_list_nlabels = {sbj: np.unique(np.array(nib.load(f).dataobj)) for sbj, f in subject_label_list.items()}
subject_list = list(filter(lambda sbj: len(subject_list_nlabels[sbj]) == num_classes, subject_list))

subject_list_train_val, subject_list_test = train_test_split(subject_list, test_size=0.2, train_size=0.8, random_state=14)
subject_list_train, subject_list_val = train_test_split(subject_list_train_val, test_size=0.2, train_size=0.8, random_state=14)

train_images = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_CT.nii.gz') for sbj in subject_list_train]
train_labels = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-oar_dseg.nii.gz') for sbj in subject_list_train]
train_vagina = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-vagina_dseg.nii.gz') for sbj in subject_list_train]
val_images = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_CT.nii.gz') for sbj in subject_list_val]
val_labels = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-oar_dseg.nii.gz') for sbj in subject_list_val]
val_vagina = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-vagina_dseg.nii.gz') for sbj in subject_list_val]
test_images = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_CT.nii.gz') for sbj in subject_list_test]
test_labels = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-oar_dseg.nii.gz') for sbj in subject_list_test]
test_vagina = [join(data_dir, sbj, 'ses-0', 'anat', sbj + '_ses-0_space-TEMPLATE_desc-vagina_dseg.nii.gz') for sbj in subject_list_test]

# Build dictionaries as accepted by MONAI loaders.
train_files = [{"image": image_name, "label": label_name, "vagina": vagina_name, "id": image_name} for image_name, label_name, vagina_name in zip(train_images, train_labels, train_vagina)]
val_files = [{"image": image_name, "label": label_name,  "vagina": vagina_name, "id": image_name} for image_name, label_name, vagina_name in zip(val_images, val_labels, val_vagina)]
test_files = [{"image": image_name, "label": label_name,  "vagina": vagina_name, "id": image_name} for image_name, label_name, vagina_name in zip(test_images, test_labels, val_vagina)]

# Set deterministic training for reproducibility
set_determinism(seed=0)

################
## Transforms ##
################
spatial_size = (256, 256, 64)
data_tf = Compose(
    [
        LoadImaged(keys=["image", "label", "vagina"]),
        EnsureChannelFirstd(keys=["image", "label", "vagina"]),
        Orientationd(keys=["image", "label", "vagina"], axcodes="RAS"),
        DivisiblePadd(keys=["image", "label", "vagina"], k=16),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-800,
            a_max=500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # RandCropByPosNegLabeld(
        #         keys=["image", "label"],
        #         label_key="vagina",
        #         spatial_size=spatial_size,
        #         pos=1,
        #         neg=0,
        #         num_samples=1,
        #     ),
        # RandSpatialCropd(
        #     keys=["image", "label"],
        #     roi_size=spatial_size,
        #     random_size=False
        # )
    ]
)

data_inv_evaluate_tf = Compose(
    [
        AsDiscreted(keys=["pred"], argmax=True, dim=0, to_onehot=num_classes),
        AsDiscreted(keys=["label"], to_onehot=num_classes),
        # Invertd(
        #     keys=["pred", "label"],
        #     transform=data_tf,
        #     orig_keys=["label", "label"],
        #     orig_meta_keys=["label_meta_dict", "label_meta_dict"],
        #     nearest_interp=False,
        #     to_tensor=True,
        #     device="cpu",
        # ),
    ]
)

data_inv_pred_tf = Compose(
    [
        AsDiscreted(keys=["pred"], argmax=True, dim=0),
        # Invertd(
        #     keys=["pred", "label", "image"],
        #     transform=data_tf,
        #     orig_keys=["label", "label", "image"],
        #     orig_meta_keys=["label_meta_dict", "label_meta_dict", "image_meta_dict"],
        #     nearest_interp=[True, True, False],
        #     to_tensor=True,
        #     device="cpu",
        # ),
    ]
)

##################
## Data loaders ##
##################
# Define CacheDataset and DataLoader for training and validation
batch_size = 1
train_ds = Dataset(data=train_files, transform=data_tf)
train_loader = DataLoader(train_ds, batch_size=1, num_workers=2)
val_ds = Dataset(data=val_files, transform=data_tf)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
test_ds = Dataset(data=test_files, transform=data_tf)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)

###########
## Model ##
###########
# Create Model, Loss, Optimizer and metrics
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
model = VNet(
    spatial_dims=3,
    in_channels=2 if seg2chan else 1,
    out_channels=num_classes,
    ).to(device)

checkpoint = torch.load(os.path.join(results_dir, "last_model.pth"), map_location=device)
model.load_state_dict(checkpoint)
model.eval()

metrics = {
    'dice': DiceMetric(include_background=True, reduction="mean"),
    'surface': SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="mean"),
    '95hd': HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95),
    'hd': HausdorffDistanceMetric(include_background=True, reduction="mean")
}

#############
## Testing ##
#############
plot_results(train_logs_file, val_logs_file)

# Copmute metrics on full-size VAL images
val_logs = {metric_str: [] for metric_str in metrics.keys()}
test_logs = {**{metric_str + '_1': [] for metric_str in metrics.keys()},
             **{metric_str + '_2': [] for metric_str in metrics.keys()},
             **{metric_str + '_3': [] for metric_str in metrics.keys()},
             **{metric_str + '_4': [] for metric_str in metrics.keys()},
             **{metric_str + '_5': [] for metric_str in metrics.keys()},
             **{'id': []}}


if evaluation_flag == 'train':
    loader = train_loader
else:
    loader = test_loader


with torch.no_grad():
    for it_test, test_data in enumerate(loader):
        print(str(it_test+1) + '/' + str(len(loader)))
        if seg2chan:
            test_inputs = torch.cat([test_data["image"], test_data["vagina"]], axis=1).to(device)

        else:
            test_inputs = test_data["image"].to(device)

        try:
            test_data["pred"] = model(test_inputs)
            posteriors = np.transpose(np.squeeze(test_data["pred"].numpy()), axes=(1, 2, 3, 0))
            # img = nib.Nifti1Image(posteriors, np.squeeze(test_data['image_meta_dict']['affine'].numpy()))
            # nib.save(img, os.path.join(results_dir, str(it_test) + '_0', 'val_posteriors.nii.gz'))
        except:
            pdb.set_trace()

        pred_data = [data_inv_pred_tf(i) for i in decollate_batch(test_data)]
        for it_td, td in enumerate(pred_data):
            import nibabel as nib
            if not os.path.exists(os.path.join(results_dir, str(it_test) + '_' + str(it_td),)):
                os.makedirs(os.path.join(results_dir, str(it_test) + '_' + str(it_td),))

            img = nib.Nifti1Image(np.squeeze(td['label'].detach().cpu().numpy()).astype('uint8'), np.squeeze(test_data['image_meta_dict']['affine'].detach().cpu().numpy()))
            nib.save(img, os.path.join(results_dir, str(it_test) + '_' + str(it_td), 'val_labels.nii.gz'))
            img = nib.Nifti1Image(np.squeeze(td["pred"].detach().cpu().numpy()).astype('uint8'),  np.squeeze(test_data['image_meta_dict']['affine'].detach().cpu().numpy()))
            nib.save(img, os.path.join(results_dir, str(it_test) + '_' + str(it_td), 'val_output.nii.gz'))
            img = nib.Nifti1Image(np.squeeze(td['image'].detach().cpu().numpy()), np.squeeze(test_data['image_meta_dict']['affine'].detach().cpu().numpy()))
            nib.save(img, os.path.join(results_dir, str(it_test) + '_' + str(it_td), 'val_inputs.nii.gz'))

        eval_data = [data_inv_evaluate_tf(i) for i in decollate_batch(test_data)]
        for it_td, td in enumerate(eval_data):
            print(str(it_test) + '_' + str(it_td) + ' --- ', end='', flush=True)
            test_logs['id'] += [os.path.basename(test_images[it_test*batch_size + it_td])[6:13]]

            for metric_str, metric_f in metrics.items():
                m = metric_f(y_pred=torch.unsqueeze(td["pred"], 0), y=torch.unsqueeze(td["label"], 0))
                m = np.squeeze(m.numpy())
                print(metric_str + ': ' + str(m), end=' ', flush=True)
                for it_mf, mf in enumerate(m):
                    test_logs[metric_str + '_' + str(it_mf+1)] += [float(mf)]

            print('')

test_logs_df = pd.DataFrame.from_dict(test_logs)
test_logs_df.to_csv(os.path.join(results_dir, 'statistics.csv'))
for metric_str, metric_list in test_logs.items():
    print()
    if metric_str == 'id': continue
    print(metric_str + ': ' + str(sum(metric_list)/len(metric_list)))
