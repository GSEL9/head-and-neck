import ioutil

import numpy as np
from joblib import Parallel, delayed


def binarize_mask(raw_mask):

    mask = np.zeros(np.shape(raw_mask), dtype=int)
    mask[raw_mask != 0] = 1

    return mask


def process_masks(raw_masks, n_jobs=1, verbose=0):

    masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
        binarize_mask(raw_mask) for raw_mask in raw_masks
    )
    ioutil.matlab_to_nrrd(masks)

    return None

if __name__ == '__main__':
    # TODO: Convert patient 58 (P58) PET, CT and mask .mat data to nrrd.

    path_raw = ''
    path_prep = ''
