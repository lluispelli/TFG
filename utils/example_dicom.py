from os.path import join, exists
from os import makedirs

import nibabel as nib
import numpy as np
from rt_utils import RTStructBuilder
from rt_utils.image_helper import get_pixel_to_patient_transformation_matrix

lps2ras = np.eye(4)
lps2ras[0, 0] = -1
lps2ras[1, 1] = -1

################################
#### Exemple NIFTI to DICOM ####
################################
INPUT_DIR = '/home/acasamitjana/Data/RadioTH/sub-152/DICOM-RT_STRUCT'
OUTPUT_FILE = '/home/acasamitjana/Data/RadioTH/sub-152/DICOM-RT_STRUCT/new_RT_STRUCT.dcm'
FILE_MASK = '/home/acasamitjana/Data/RadioTH/sub-4107902/ses-0/seg/sub-4107902_ses-0_OAR.nii.gz'

proxymask = nib.load(FILE_MASK)
mask = np.array(proxymask.dataobj) > 0
mask = np.stack([mask[..., it_d].T for it_d in range(mask.shape[-1])], -1)

rtstruct = RTStructBuilder.create_new(dicom_series_path=INPUT_DIR)
rtstruct.add_roi(
    mask=mask,
    color=[255, 0, 255],
    name="Replicate OAR"
)
rtstruct.save(OUTPUT_FILE)


################################
#### Exemple DICOM to NIFTI ####
################################


INPUT_DIR = '/home/acasamitjana/Data/RadioTH/sub-152/DICOM-RT_STRUCT'
OUTPUT_DIR = '/home/acasamitjana/Data/RadioTH/'

rtstruct = RTStructBuilder.create_from(
    dicom_series_path=INPUT_DIR, rt_struct_path=join(INPUT_DIR, 'RS_sub-152_4107902.dcm')
)

vox2lps = get_pixel_to_patient_transformation_matrix(rtstruct.series_data)
vox2ras = lps2ras @ vox2lps
sid = rtstruct.ds.PatientID

image = np.stack([sd.pixel_array.T for sd in rtstruct.series_data], -1)
mask_3d = rtstruct.get_roi_mask_by_name("VAGINA OAR")
mask_3d = np.stack([mask_3d[..., it_d].T for it_d in range(mask_3d.shape[-1])], -1)

if not exists(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'anat')): makedirs(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'anat'))
if not exists(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'seg')): makedirs(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'seg'))
img = nib.Nifti1Image(mask_3d.astype('uint8'), vox2ras)
nib.save(img, join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'seg', 'sub-' + str(sid) + '_ses-0_OAR.nii.gz'))
img = nib.Nifti1Image(image, vox2ras)
nib.save(img, join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'anat', 'sub-' + str(sid) + '_ses-0_CT.nii.gz'))
