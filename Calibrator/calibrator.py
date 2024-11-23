#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Title        : calibrator.py
Description  : <Brief description of what the script does.>
Author       : Dr. Xu Hao
Email        : xuhao3e8@buaa.edu.cn
Affiliation  : Institute of Unmanned Systems, Beihang University
Created Date : 2024-11-22
Last Updated : 2024-11-22

===============================================================================
Copyright (C) <Year> Dr. Xu Hao

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2.1 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with this program; if not, see <https://www.gnu.org/licenses/lgpl-2.1.html>.

===============================================================================
"""

import cv2
import numpy as np

class IntrinsicCalibrator:
    def __init__(self, calibration_method="OPENCV") -> None:
        self.calibration_method = calibration_method

    def calibrate_mono(self, detector, image_size, K_init = np.array([2., 1117., 1117., 651., 384.]),
                                                    D_init = np.array([-0.2, 0.4, 0., 0.])):
        pts_3d, pts_2d = detector.gather_information()
        if pts_3d is None or pts_2d is None or len(pts_3d) == 0 or len(pts_2d) == 0:
            print("No enough information for calibration.")
            return None, None, None, None, None
        if self.calibration_method == "OPENCV":
            flags = 0
            if K_init is not None:
                flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            K = np.array([[K_init[1], 0, K_init[3]], [0, K_init[2], K_init[4]], [0, 0, 1]])
            retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(pts_3d, pts_2d, image_size=image_size, K=K, D=D_init, flags=flags)
            print("Intrinsic calibration done: ", retval)
            print("K: ", K)
            print("D: ", D)
        elif self.calibration_method == "GTSAM":
            pass
        else:
            raise ValueError("The calibration method is not supported.")
        return retval, K, D, rvecs, tvecs
