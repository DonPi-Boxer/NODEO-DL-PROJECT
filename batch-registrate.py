import os
import glob
import Registration as rg
import argparse
import subprocess

#get and store all file paths for the MRI images and the segmentations in arrays
data_dir = './oasis-data'

file_paths_mri = []
for dirpath, dirnames, filenames in os.walk('.'):
    for filename in [f for f in filenames if f.endswith("norm.nii.gz")]:
        file_paths_mri.append(os.path.join(dirpath,filename))
        
file_paths_seg = []
for dirpath, dirnames, filenames in os.walk('.'):
    for filename in [f for f in filenames if f.endswith("seg35.nii.gz")]:
        file_paths_seg.append(os.path.join(dirpath,filename))
        



subprocess.run(['python', 'Registration.py', '--moving', './oasis-data/OASIS_OAS1_0049_MR1/aligned_norm.nii.gz', '--fixed', './oasis-data/OASIS_OAS1_0134_MR1/aligned_norm.nii.gz', '--moving_seg', './oasis-data/OASIS_OAS1_0049_MR1/aligned_seg35.nii.gz', '--fixed_seg', './oasis-data/OASIS_OAS1_0134_MR1/aligned_seg35.nii.gz'])

#subprocess.run(['python', 'Registration.py', '--moving', './data/OAS1_0002_MR1/brain.nii.gz', '--fixed', './data/OAS1_0001_MR1/brain.nii.gz', '--moving_seg', './data/OAS1_0001_MR1/brain_aseg.nii.gz', '--fixed_seg', './data/OAS1_0002_MR1/brain_aseg.nii.gz'])
