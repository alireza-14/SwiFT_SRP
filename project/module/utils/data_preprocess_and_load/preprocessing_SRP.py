import nibabel as nib
import torch
import os
import time
from multiprocessing import Process, Queue
from glob import glob

from tqdm import tqdm

def read_data(filename, mask_filename,save_root,subj_name,scaling_method=None, fill_zeroback=False):
    print("processing: " + filename, flush=True)
    subj_id = subj_name.split("_")[0]

    bold_path = filename
    mask_path = mask_filename
    try:
        # load each nifti file
        data = nib.load(bold_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        mask = mask == 0
    except Exception as e:
        return print(e)
    
    # fill masks with zero
    for t in range(data.shape[3]):
        data[:, :, :, t][mask] = 0
    
    #change this line according to your file names
    save_dir = os.path.join(save_root,subj_name)
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    
    # change this line according to your dataset
    # width, height, depth, time
    # Inspect the fMRI file first using your visualization tool. 
    # Limit the ranges of width, height, and depth to be under 96. Crop the background, not the brain regions. 
    # Each dimension of fMRI registered to MNI space (2mm) is expected to be around 100.
    # You can do this when you load each volume at the Dataset class, including padding backgrounds to fill dimensions under 96.
    data = torch.from_numpy(data).float()
    background = data==0
    
    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp[~background].min() if not fill_zeroback else 0 
    # data_temp[~background].min() is expected to be 0 for scaling_method == 'minmax', and minimum z-value for scaling_method == 'z-norm'
    data_global[~background] = data_temp[~background]

    # save volumes one-by-one in fp16 format.
    data_global = data_global.type(torch.float16)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in tqdm(enumerate(data_global_split), desc=f"processing {subj_id}:"):
        torch.save(TR.clone(), os.path.join(save_dir,"frame_"+str(i)+".pt"))


def main():
    # change two lines below according to your dataset
    dataset_name = 'SRP'
    file_formats = '/content/ds003745/*/*_desc-preproc_bold.nii.gz'
    save_root = f'/content/{dataset_name}_MNI_to_TRs_minmax'
    scaling_method = 'z-norm' # choose either 'z-norm'(default) or 'minmax'.

    # make result folders
    filenames = glob(file_formats)
    
    os.makedirs(os.path.join(save_root,'img'), exist_ok = True)
    os.makedirs(os.path.join(save_root,'metadata'), exist_ok = True) # locate your metadata file at this folder 
    save_root = os.path.join(save_root,'img')
    
    finished_samples = os.listdir(save_root) 
    count = 0
    for filename in sorted(filenames):
        mask_filename = filename.replace(
            "_desc-preproc_bold.nii.gz",
            "_desc-brain_mask.nii.gz"
        )
        if not os.path.exists(mask_filename):
            continue

        subj_name = os.path.basename(filename[:-25])
        # extract subject name from nifti file. [:-7] rules out '.nii.gz'
        # we recommend you use subj_name that aligns with the subject key in a metadata file.

        expected_seq_length = 1000 # Specify the expected sequence length of fMRI for the case your preprocessing stopped unexpectedly and you try to resume the preprocessing.
        
        # change the line below according to your folder structure
        if (subj_name not in finished_samples) or (len(os.listdir(os.path.join(save_root,subj_name))) < expected_seq_length): # preprocess if the subject folder does not exist, or the number of pth files is lower than expected sequence length. 
            try:
                count+=1
                read_data(filename, mask_filename,save_root,subj_name,scaling_method)
            except Exception:
                print('encountered problem with'+filename)
                print(Exception)

if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')    
