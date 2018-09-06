# -*- coding: utf-8 -*-
#
# io.py
#
# This module is part of radiomics.
#

"""
Read PET and CT scans in MATLAB format and write to NRRD format.
"""


import os
import nrrd

import numpy as np
import pandas as pd
import scipy.io as io


# TODO:
# Do this in parallel for quick conversion.
# Read PET and CT images in MATLAB format.
# Write image and mask to NRRD format.

# QUESTION:
# Difference prep/mask PET/CT?
# Only need to convert images once.
