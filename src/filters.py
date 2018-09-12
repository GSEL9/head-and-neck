import utils
import imgio

import numpy as np

from radiomics import imageoperations


# TODO:
# * Perform type chekcing and comparison of iamge + mask inside filter funcs.
# * Check if correct to return only pyradiomics object as filtered images.


def log(image, mask, sigma=[1.0]):
    """Applies a Laplacian of Gaussian filter to the input image and yields a
    derived image for each sigma value specified."""

    utils.check_mask_image(image, mask)

    return imageoperations.getLoGImage(image, mask, sigma=sigma)

# NB ERROR:
def wavelet(image, mask, start_level=None, level=None):

    utils.check_mask_image(image, mask)

    return imageoperations.getWaveletImage(
        image, mask, start_level=start_level, level=level
    )


def square(image, mask):
    """Computes the square of the image intensities."""

    utils.check_mask_image(image, mask)

    return imageoperations.getSquareImage(image, mask)


def square_root(image, mask):
    """Computes the square root of the absolute value of image intensities."""

    utils.check_mask_image(image, mask)

    return imageoperations.getSquareRootImage(image, mask)


def logarithm(image, mask):
    """Computes the logarithm of the absolute value of the original image + 1.
    """

    utils.check_mask_image(image, mask)

    return imageoperations.getLogarithmImage(image, mask)


def exponential(image, mask):
    """Computes the exponential of the original image."""

    #utils.check_mask_image(image, mask)

    return imageoperations.getExponentialImage(image, mask)


def lbp(mask, image, levels=None, radius=None, divisions=None):
    """Compute the Local Binary Pattern image in 3D using spherical harmonics.

    Args:

    Returns:


    """

    utils.check_mask_image(image, mask)

    return imageoperations.getLBP3DImage(
        image, mask, lbp3DLevels=levels, lbp3DIcosphereRadius=radius,
        lbp3DIcosphereSubdivision=divisions
    )


if __name__ == '__main__':
    # NB: Error in importing read nrrd from io.py module because of name
    # conflict with python geenric. Make package ans inastall to use io.py.

    # TODO: Checkout python multiprocessing with imgaes

    # TEMP:
    import numpy as np
    from datetime import datetime
    import SimpleITK as sitk

    # Setup:
    path_ct_dir = './../data/imgs/ct_stacks/cropped_ct/'
    path_pet_dir = './../data/imgs/pet_stacks/cropped_per/'
    path_masks_dir = './../data/imgs/masks/prep_masks/'

    # Demo run:
    rel_ct_paths, rel_mask_paths = imgio.relative_paths_pairwise(
        path_ct_dir, path_masks_dir, target_format='nrrd'
    )

    #image = imgio.read_nrrd(rel_ct_paths[0], traverse=False)
    #mask = imgio.read_nrrd(rel_mask_paths[0], traverse=False)

    #img = exponential(image, mask)

    #for num, items in enumerate(img):
    #    sitk.WriteImage(items[0], 'test.nrrd')

    # Sequential filtering of images.
    start = datetime.now()
    for num, path_to_file in enumerate(rel_ct_paths[:2]):

        img = imgio.read_nrrd(path_to_file, traverse=False)
        mask = imgio.read_nrrd(rel_mask_paths[num], traverse=False)
        print(type(img))
        #filtered = exponential(img, mask)

        #for img in filtered:
        #    sitk.WriteImage(img[0], 'test.nrrd')
    print('Execution time: {}'.format(datetime.now() - start))

    # Execution time: 0:00:13.836056
