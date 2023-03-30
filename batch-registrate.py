import os
import glob
import Registration
import argparse
import subprocess
import numpy as np

#get and store all file paths for the MRI images and the segmentations in arrays

def main(config):
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
            fixed_set_name.append(os.path.basename(os.path.join(dirpath))) 
            fixed_files_paths_mri.append(os.path.join(dirpath,filename))
        for filename in [f for f in filenames if f.endswith("seg35.nii.gz")]:
            fixed_file_paths_seg.append(os.path.join(dirpath,filename))
    print(fixed_set_name)
    print(moving_set_name)
            #print(moving_set_name)
            #print(fixed_set_name)
    numruns = 0
    runtime = 0
    mean_avg_dice = []
    for moving_set_name,moving_mri,moving_seg in zip(moving_set_name,moving_file_paths_mri,moving_file_paths_seg):
        for fixed_set_name,fixed_mri,fixed_seg in zip(fixed_set_name,fixed_files_paths_mri,fixed_file_paths_seg):
            print("hi")
            if moving_mri != fixed_mri:
                numruns = numruns +1
                savedir = './result/' + moving_set_name +'/' + fixed_set_name
                print(savedir)
                if not os.path.isdir(savedir):
                    os.makedirs(savedir)
                print("hi")
                avg_dice, runtime = Registration.main(config = config, moving_mri = moving_mri, fixed_mri = fixed_mri,savedir=savedir, fixed_seg_in = fixed_seg, moving_seg_in=moving_seg)
                        #Registration.parser.set_defaults(moving = moving_mri)
                        #avg_dice =  Registration.main('moving:', moving_mri, '--fixed', fixed_mri, '--moving_seg', moving_seg, '--fixed_seg', fixed_seg, '--savepath', savepath_run)
                runtime = runtime + runtime
                mean_avg_dice.append(avg_dice)
                #print(avg_dice)
                print("Mean average dice after " , numruns , " registrations is " , np.mean(mean_avg_dice))
    print("all done ! Mean avg dice is ", np.mean(mean_avg_dice))
    print("total runtime was ", runtime, " for in total ", numruns, " Registrations")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # File path
    #parser.add_argument("--savepath", type=str,
    #                    dest="savepath", default='./result',
    #                    help="path for saving results")
    #parser.add_argument("--fixed", type=str,
    #                    dest="fixed", default='./data/OAS1_0001_MR1/brain.nii.gz',
    #                    help="fixed image data path")
    #parser.add_argument("--moving", type=str,
    #                    dest="moving", default='./data/OAS1_0002_MR1/brain.nii.gz',
    #                    help="moving image data path")
    #parser.add_argument("--fixed_seg", type=str,
    ##                    dest="fixed_seg", default='./data/OAS1_0001_MR1/brain_aseg.nii.gz',
    #                    help="fixed image segmentation data path")
    #parser.add_argument("--moving_seg", type=str,
    #                    dest="moving_seg", default='./data/OAS1_0002_MR1/brain_aseg.nii.gz',
    #                    help="moving image segmentation data path")
    # Model configuration
    parser.add_argument("--ds", type=int,
                        dest="ds", default=2,
                        help="specify output downsample times.")
    parser.add_argument("--bs", type=int,
                        dest="bs", default=16,
                        help="bottleneck size.")
    parser.add_argument("--smoothing_kernel", type=str,
                        dest="smoothing_kernel", default='AK',
                        help="AK: Averaging kernel; GK: Gaussian Kernel")
    parser.add_argument("--smoothing_win", type=int,
                        dest="smoothing_win", default=15,
                        help="Smoothing Kernel size")
    parser.add_argument("--smoothing_pass", type=int,
                        dest="smoothing_pass", default=1,
                        help="Number of Smoothing pass")
    # Training configuration
    parser.add_argument("--time_steps", type=int,
                        dest="time_steps", default=2,
                        help="number of time steps between the two images, >=2.")
    parser.add_argument("--optimizer", type=str,
                        dest="optimizer", default='Euler',
                        help="Euler or RK.")
    parser.add_argument("--STEP_SIZE", type=float,
                        dest="STEP_SIZE", default=0.001,
                        help="step size for numerical integration.")
    parser.add_argument("--epoches", type=int,
                        dest="epoches", default=300,
                        help="No. of epochs to train.")
    parser.add_argument("--NCC_win", type=int,
                        dest="NCC_win", default=21,
                        help="NCC window size")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.005,
                        help="Learning rate.")
    parser.add_argument("--lambda_J", type=int,
                        dest="lambda_J", default=2.5,
                        help="Loss weight for neg J")
    parser.add_argument("--lambda_df", type=int,
                        dest="lambda_df", default=0.05,
                        help="Loss weight for dphi/dx")
    parser.add_argument("--lambda_v", type=int,
                        dest="lambda_v", default=0.00005,
                        help="Loss weight for neg J")
    parser.add_argument("--loss_sim", type=str,
                        dest="loss_sim", default='NCC',
                        help="Similarity measurement")
    # Debug
    parser.add_argument("--debug", type=bool,
                        dest="debug", default=False,
                        help="debug mode")
    #Device run on GPU
    parser.add_argument("--device", type=str,
                        dest="device", default='cuda:0',
                        help="gpu: cuda:0; cpu: cpu")
    
    # Device run on CPU    
    #parser.add_argument("--device", type=str,
    #                    dest="device", default='cpu')
    config = parser.parse_args()
    
main(config)