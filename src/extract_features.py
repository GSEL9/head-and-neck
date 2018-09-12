"""
Steps taken inside pyradiomics feature extractor .execute():
-----------------------------------------------------------------------
1. Image and mask are loaded and normalized/resampled if necessary.
2. Validity of ROI is checked using :py:func:`~imageoperations.checkMask`,
    which also computes and returns the bounding box.
3. If enabled, provenance information is calculated and stored as part of the
    result. (Not available in voxel-based extraction)
4. Shape features are calculated on a cropped (no padding) version of the
    original image. (Not available in voxel-based extraction)
5. If enabled, resegment the mask based upon the range specified in
    ``resegmentRange`` (default None: resegmentation disabled).
6. Other enabled feature classes are calculated using all specified image
    types in ``_enabledImageTypes``. Images are cropped to tumor mask
    (no padding) after application of any filter and before being passed to the
    feature class.
7. The calculated features are returned as ``collections.OrderedDict``.

"""

# Pipeline:
# 1. Read image
# 2. Filter image
# 3. Quantie image
# 4. Extract features (different extractor settings sheets)
# 5. Save features to file.


# NOTE: In parallel:
for image in images:
    # In parallel
    for filter in filters:
        filtered_img = filter(image)


        for quantize in bins:
            binned = quantize(filtered_img)

            features = extractor(binned)


# https://stackoverflow.com/questions/33778155/python-parallelized-image-reading-and-preprocessing-using-multiprocessing
# https://stackoverflow.com/questions/19080792/run-separate-processes-in-parallel-python
