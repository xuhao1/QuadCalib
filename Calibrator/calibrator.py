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

    def calibrate_mono(self, detector, image_size, intrinsic_init = np.array([1.24, 813, 812, 640, 360]),
                                                    D_init = np.array([-0.2, 0.4, 0., 0.]), is_omnidir=False):
        pts_3d, pts_2d = detector.gather_information()
        if pts_3d is None or pts_2d is None or len(pts_3d) == 0 or len(pts_2d) == 0:
            print("No enough information for calibration.")
            return None, None, None, None, None
        if self.calibration_method == "OPENCV":
            if is_omnidir:
                flags = cv2.omnidir.CALIB_FIX_SKEW
                if intrinsic_init is not None:
                    flags |= cv2.omnidir.CALIB_USE_GUESS
                D_init = D_init.reshape((1, 4))
                K = np.array([[intrinsic_init[1], 0, intrinsic_init[3]], [0, intrinsic_init[2], intrinsic_init[4]], [0, 0, 1]], dtype=np.float64)
                xi = np.array([intrinsic_init[0]], dtype=np.float64)
                rvecs = None
                tvecs = None
                retval, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(pts_3d, pts_2d, size=image_size, K=K,
                                                                xi=xi, D=D_init, flags=flags,
                                                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
            else: # Use opencv pinhole to calibrate
                D_init = D_init.reshape((1, 4))
                rvecs = None
                tvecs = None
                retval, K, D, rvecs, tvecs, idx = cv2.calibrateCamera(pts_3d, pts_2d, image_size, cameraMatrix=None, distCoeffs=None,
                                                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
            print("Intrinsic calibration done: RMSE:", retval)
            print(f"Intrinsic: [{xi[0]}, {K[0, 0]}, {K[1, 1]}, {K[0, 2]}, {K[1, 2]}]")
            print("K:\n", K)
            print("D: ", D[0])
        elif self.calibration_method == "GTSAM":
            pass
        else:
            raise ValueError("The calibration method is not supported.")
        return retval, K, D, rvecs, tvecs
