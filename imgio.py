# TODO:
# * functino to read CT, PET and mask in parallel (must sort mask:images).
# * function to save extracted features to file


import os
#import nrrd
import SimpleITK as sitk


import utils


def relative_paths(path_to_dir, target_format=None):
    """Return a list of relative paths to all files in directory with target
    extension."""

    # Collect all file names with target extension from dir.
    file_names = os.listdir(path_to_dir)

    rel_paths = []
    for file_name in file_names:

        rel_path = os.path.join(path_to_dir, file_name)
        if os.path.isfile(rel_path) and rel_path.endswith(target_format):
            rel_paths.append(rel_path)

    return rel_paths


# NB: Does not sort by specific key. Assumes a standardized format.
def relative_paths_pairwise(path_to_a, path_to_b, target_format=None):

    rel_paths_a = relative_paths(path_to_a, target_format=target_format)
    rel_paths_b = relative_paths(path_to_b, target_format=target_format)

    # Sort path to obtain correct pairwise matching of rel paths.
    rel_paths_a.sort(), rel_paths_b.sort()

    return rel_paths_a, rel_paths_b


def read_nrrd(path_to_files, traverse=True):

    # Collect all NRRD files in dir.
    if traverse:

        rel_paths = relative_paths(path_to_files, target_format='nrrd')

        data, metadata = [], []
        for path_to_file in rel_paths:

            _data = sitk.ReadImage(path_to_file)
            utils.check_image(sitk.GetArrayFromImage(_data))
            data.append(_data)

            #_data, _metadata = nrrd.read(path_to_file)
            #data.append(utils.check_image(_data)), metadata.append(_metadata)

    # Read single file.
    else:
        if os.path.isfile(path_to_files) and path_to_files.endswith('nrrd'):

            data = sitk.ReadImage(path_to_files)
            utils.check_image(utils.sitk_to_ndarray(data))

            #_data, metadata = nrrd.read(path_to_files)
            #data = utils.check_image(_data)

        else:
            raise RuntimeError('Invalid path: `{}`'.format(path_to_files))

    return data


def write_nrrd(path_to_file):

    pass



if __name__ == '__main__':


    from datetime import datetime

    path_ct_dir = './../data/imgs/prep_ct_scans/'
    path_pet_dir = './../data/imgs/prep_pet_scans/'
    path_masks_dir = './../data/imgs/prep_masks/'

    #start = datetime.now()
    #read_nrrd(path_ct_dir)
    #read_nrrd(path_pet_dir)
    #read_nrrd(path_masks_dir)
    #print('Execution time: {}'.format(datetime.now() - start))

    #print(relative_paths(path_ct_dir, target_format='.nrrd'))
