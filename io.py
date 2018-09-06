# -*- coding: utf-8 -*-
#
# io.py
#
# This module is part of radiomics.
#

"""
Convert PET and CT scans from MATLAB to NRRD format including the ROI mask.
"""


import os
import nrrd

import numpy as np
import pandas as pd
import scipy.io as io


# TODO:
# Read PET and CT images in MATLAB format.
# Write image and mask to NRRD format.

# QUESTION:
# Difference prep PET/CT?
# Do this process in parallel with feat extract?
# Same mask to PET and CT?
