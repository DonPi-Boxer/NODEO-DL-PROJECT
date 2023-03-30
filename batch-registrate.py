import os
import glob
import Registration as rg
import argparse
import subprocess
import numpy as np

#get and store all file paths for the MRI images and the segmentations in arrays
moving_data_dir = './oasis-data/fixed'
moving_data_labels = np.arange(1,49,1)

fixed_data_dir = './oasis-data/fixed'
fixed_data_labels = [10,20,30,40,50]

moving_set_name = []
moving_file_paths_mri = []
moving_file_paths_seg = []
for dirpath, dirnames, filenames in os.walk(moving_data_dir):
    for filename in [f for f in filenames if f.endswith("norm.nii.gz")]:
        moving_set_name.append(os.path.basename(os.path.normpath(dirpath)))
        moving_file_paths_mri.append(os.path.join(dirpath,filename))
    for filename in [f for f in filenames if f.endswith("seg35.nii.gz")]:    
        moving_file_paths_seg.append(os.path.join(dirpath,filename))
   
fixed_set_name =[]
fixed_files_paths_mri = []
fixed_file_paths_seg = []     
for dirpath, dirnames, filenames in os.walk('.'):
    for filename in [f for f in filenames if f.endswith("norm.nii.gz")]:
        fixed_set_name.append(os.path.basename(os.path.normpath(dirpath)))
        fixed_files_paths_mri.append(os.path.join(dirpath,filename))
    for filename in [f for f in filenames if f.endswith("seg35.nii.gz")]:
        fixed_file_paths_seg.append(os.path.join(dirpath,filename))

#print(moving_set_name)
#print(fixed_set_name)
numruns = 0
mean_avg_dice = []
for moving_set_name,moving_mri,moving_seg in zip(moving_set_name,moving_file_paths_mri,moving_file_paths_seg):
    for fixed_set_name,fixed_mri,fixed_seg in zip(fixed_set_name,fixed_files_paths_mri,fixed_file_paths_seg):

            if moving_mri != fixed_mri:
                numruns = numruns +1
                savepath_run = './result/' + moving_set_name +'/' + fixed_set_name
            #print(savepath_run)
                avg_dice =  subprocess.run(['python', 'Registration.py', '--moving', moving_mri, '--fixed', fixed_mri, '--moving_seg', moving_seg, '--fixed_seg', fixed_seg, '--savepath', savepath_run])
                mean_avg_dice.append(avg_dice)
                print(avg_dice)
                print("Mean average dice after " , numruns , " runs is " , np.mean(mean_avg_dice))
print("all done ! Mean avg dice is ", np.mean(mean_avg_dice))
#moving_mri = moving_file_paths_mri[2]
#print(moving_mri)
#moving_seg = moving_file_paths_seg[2]
#print(moving_seg)
#fixed_mri = fixed_files_paths_mri[3]
#print(fixed_mri)
#fixed_seg = fixed_file_paths_seg[3]
#print(fixed_seg)
#savepath_run = "./bla"
#subprocess.run(['python', 'Registration.py', '--moving', moving_mri, '--fixed', fixed_mri, '--moving_seg', moving_seg, '--fixed_seg', fixed_seg])
#subprocess.run(['python', 'Registration.py', '--moving', './oasis-data/moving/OASIS_OAS1_0018_MR1/aligned_norm.nii.gz', '--fixed','./oasis-data/fixed/OASIS_OAS1_0030_MR1/aligned_norm.nii.gz',  '--moving_seg', './oasis-data/moving/OASIS_OAS1_0018_MR1/aligned_seg35.nii.gz', '--fixed_seg', './oasis-data/fixed/OASIS_OAS1_0030_MR1/aligned_seg35.nii.gz', '--savepath', savepath_run], capture_output=True)

#m_avg_dice = np.mean(mean_avg_dice)
#print("All done ! Mean average dice is " + m_avg_dice)
#subprocess.run(['python', 'Registration.py', '--moving', './data/OAS1_0002_MR1/brain.nii.gz', '--fixed', './data/OAS1_0001_MR1/brain.nii.gz', '--moving_seg', './data/OAS1_0001_MR1/brain_aseg.nii.gz', '--fixed_seg', './data/OAS1_0002_MR1/brain_aseg.nii.gz'])
