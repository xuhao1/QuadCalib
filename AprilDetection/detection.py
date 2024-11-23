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
import numpy as np

import yaml
import numpy as np
from AprilDetection.aprilgrid import generate_aprilgrid_3d_points, load_aprilgrid_config

class Detector:               
    def __init__(self, camera_id = 0, tag_config="tag36h11", minimum_tag_num=4, yaml_file=None) -> None:
        self.camera_id = camera_id
        self.tag_config = tag_config
        self.detector_opencv = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.results = {}
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.markerBorderBits = 2
        arucoParams.adaptiveThreshWinSizeStep = 1
        arucoParams.adaptiveThreshWinSizeMin = 3
        self.arucoParams = arucoParams
        self.minimum_tag_num = minimum_tag_num
        self.image_accumulate_corners = None
        aprilgrid_config = load_aprilgrid_config(yaml_file)  
        self.aprilgrid_3d_points = generate_aprilgrid_3d_points(aprilgrid_config['tagCols'],
                                                                aprilgrid_config['tagRows'],
                                                                aprilgrid_config['tagSize'],
                                                                aprilgrid_config['tagSpacing'])
        
    def detect(self, image, image_t, image_idx, show=False):
        # Defaultly detect use apriltag
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #And then use opencv
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, self.detector_opencv,
                                                                parameters=self.arucoParams)
        # corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        if len(corners) >= self.minimum_tag_num:
            self.results[image_idx] = (image_t, corners, ids)
        if show:
            # Draw corners and ids on the image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if self.image_accumulate_corners is None:
                self.image_accumulate_corners = np.zeros(image.shape)
            
            cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))
            
            # Draw corners in image_accumulate_corners
            if len(corners) >= self.minimum_tag_num:
                for corner in corners:
                    for c in corner[0]:
                        cv2.circle(self.image_accumulate_corners, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)
            
        return image, self.image_accumulate_corners

    def gather_information(self):
        # Output 3D points and 2D points
        # objectPoints	vector of vectors of calibration pattern points in the calibration pattern coordinate space.
        # imagePoints	vector of vectors of the projections of calibration pattern points. imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to objectPoints[i].size() for each i.
        objectPoints = []
        imagePoints = []
        for image_idx, (image_t, corners, ids) in self.results.items():
            objectPointsInFrame = []
            imagePointsInFrame = []
            if len(corners) < self.minimum_tag_num:
                continue
            # Get the 3D points of the tags
            for corner, id in zip(corners, ids):
                apriltag_pts = self.aprilgrid_3d_points[id[0]].reshape(1, -1, 3)
                corners = corner.reshape(1, -1, 2)
                objectPointsInFrame.extend(apriltag_pts)
                imagePointsInFrame.extend(corners)
            objectPointsInFrame = np.array(objectPointsInFrame, dtype=np.float32).reshape(-1, 1, 3)
            imagePointsInFrame = np.array(imagePointsInFrame, dtype=np.float32).reshape(-1, 1, 2)
            objectPoints.append(objectPointsInFrame)
            imagePoints.append(imagePointsInFrame)
        return objectPoints, imagePoints
