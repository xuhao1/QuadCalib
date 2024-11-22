#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Title        : main.py
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

class Detector:               
    def __init__(self, camera_id = 0, tag_config="tag36h11", calibration_method="OPENCV", minimum_tag_num=4):
        self.camera_id = camera_id
        self.tag_config = tag_config
        self.detector_opencv = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.results = {}
        self.calibration_method = calibration_method
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.markerBorderBits = 2
        arucoParams.adaptiveThreshWinSizeStep = 1
        arucoParams.adaptiveThreshWinSizeMin = 3
        self.arucoParams = arucoParams
        self.minimum_tag_num = minimum_tag_num
    
    def detect(self, image, image_t, image_idx, show=False):
        # Defaultly detect use apriltag
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #And then use opencv
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, self.detector_opencv,
                                                                parameters=self.arucoParams)
        if len(corners) >= self.minimum_tag_num:
            print(f"Detect {len(corners)} on camera {self.camera_id} at I{image_idx}")
            self.results[image_idx] = (image_t, corners, ids)

        if show:
            # Draw corners and ids on the image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))
            # cv2.imshow(f"Image {self.camera_id}", image)
        return image
    
    def calibrate_mono(self, image_size, initial_K=None, initial_D=None):
        pts_3d, pts_2d = self.gather_information()
        if self.calibration_method == "OPENCV":
            flags = 0
            if initial_K is not None:
                flags |= cv2.fisheye.CALIB_FIX_INTRINSIC
            retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(pts_3d, pts_2d, image_size=image_size, K=initial_K, D=initial_D, flags=flags)
        elif self.calibration_method == "GTSAM":
            pass
        else:
            raise ValueError("The calibration method is not supported.")
    
    def gather_information(self):
        pass