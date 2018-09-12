import numpy as np
import SimpleITK as sitk


def sitk_to_ndarray(image):

    return sitk.GetArrayFromImage(image)


def check_image(image):
    # Perform type check and dimensionality check of image.
    if not isinstance(image, np.ndarray):
        try:
            _image = np.copy(sitk_to_ndarray(image))
        except:
            raise TypeError('Image should be <numpy.ndarray> or SimpleITK, '
                            'not {}'.format(type(image)))
    else:
        _image = np.copy(image)

    if _image.ndim == 3:
        return None
    else:
        raise RuntimeError('Invalid image dimensionality {}D. Should be 3D.'
                           ''.format(_image.ndim))


def check_mask_image(image, mask):
    # Perform type check and dimensionality check of image and corresponding
    # mask.

    if not isinstance(image, np.ndarray):
        try:
            _image = np.copy(sitk_to_ndarray(image))
        except:
            raise TypeError('Image should be <numpy.ndarray> or SimpleITK, '
                            'not {}'.format(type(image)))
    else:
        _image = np.copy(image)

    if not isinstance(mask, np.ndarray):
        try:
            _mask = np.copy(sitk_to_ndarray(mask))
        except:
            raise TypeError('Mask should be <numpy.ndarray> or SimpleITK, '
                            'not {}'.format(type(mask)))
    else:
        _mask = np.copy(mask)

    if _image.ndim == 3:
        if not np.shape(_image) == np.shape(_mask):
            raise RuntimeError('Invalid mask-image pair.')
        else:
            return None
    else:
        raise RuntimeError('Invalid image dimensionality {}D. Should be 3D.'
                           ''.format(_image.ndim))


if __name__ == '__main__':

    from readimg import read_nrrd

    from datetime import datetime

    path_ct_dir = './../data/imgs/prep_ct_scans/'
    path_pet_dir = './../data/imgs/prep_pet_scans/'
    path_masks_dir = './../data/imgs/prep_masks/'

    # QUESTION: How to quantize filtered images is only pyradiomic object are
    # returned?

    #start = datetime.now()
    #read_nrrd(path_ct_dir)
    #read_nrrd(path_pet_dir)
    #read_nrrd(path_masks_dir)
    #print('Execution time: {}'.format(datetime.now() - start))

    #print(relative_paths(path_ct_dir, target_format='.nrrd'))
