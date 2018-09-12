"""Pipieline:
--------------
1. read image
2. filter image
    - All possible filters
        - All possible filter param combinations
3. save image
4. convert to numpy.
5. quantize image
6. save image.
"""

import os
import imgio
import filters

from datetime import datetime


def quantize_image(image, nbins=8):
    """
    Args:
        bins (int): The number of distinct intensity values.

    Returns:
        ():

    """

    min_pixel, max_pixel = np.min(image), np.max(image)
    bins = np.linspace(min_pixel, max_pixel, nbins)

    quantized = np.digitize(np.copy(image), bins)

    return quantized.astype(float)


def filter_image(image, mask, filters, params, to_disk=True, **kwargs):

    filtered_imgs = []
    for name, func in filters.keys():
        # Regulate arguments to filter function.
        if name in params:
            filtered = func(image, mask, params[name])
        else:
            filtered = func(image, mask)

        # Save filtered image to disk???


def main(path_to_imgs, path_to_masks, verbose=True, **kwargs):

    if verbose:
        start_time = datetime.now()

    rel_img_paths, rel_mask_paths = imgio.relative_paths_pairwise(
        path_to_imgs, path_to_masks, target_format='nrrd'
    )
    for num, img_path in enumerate(rel_img_paths):

        image = imgio.read_nrrd(img_path, traverse=False)
        mask = imgio.read_nrrd(rel_mask_paths[num], traverse=False)

        filtered_imgs = filter_image(image, mask, **kwargs)
        for filtered_img in filtered_imgs:
            quantized = quantize_image(filtered_img)

    if verbose:
        print('Execution time: {}'.format(datetime.now() - start))

if __name__ == '__main__':

    # Setup:
    path_ct_dir = './../data/imgs/ct_stacks/cropped_ct/'
    path_pet_dir = './../data/imgs/pet_stacks/cropped_per/'
    path_masks_dir = './../data/imgs/masks/prep_masks/'

    main(path_ct_dir, path_masks_dir)
