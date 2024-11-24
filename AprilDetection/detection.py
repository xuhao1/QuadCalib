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
from QuadUtils import K_xi_from_Intrinsic

class Detector:               
    def __init__(self, camera_id = 0, tag_config="tag36h11", minimum_tag_num=4, 
                    yaml_file=None,
                    intrinsic_init=None,
                    D_init=None,
                    undist_before_detection=False) -> None:
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
        self.undist_before_detection = undist_before_detection
        self.intrinsic_init = intrinsic_init
        self.D_init = D_init
        
        K, xi = K_xi_from_Intrinsic(self.intrinsic_init)
        new_size = (2000, 1000)
        Knew = np.array([[new_size[0]/5, 0.0, 0.0],
                        [0.0, new_size[0]/5, 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
        self.map1, self.map2 = cv2.omnidir.initUndistortRectifyMap(K, self.D_init, xi, np.eye(3),
                                                                   Knew, new_size, cv2.CV_16SC2, cv2.omnidir.RECTIFY_CYLINDRICAL)
    
    def detect_subpix_corner(self, image_gray, marker_corners):
        markers_corners_subpixes = []
        if len(marker_corners) >= self.minimum_tag_num:
            for corners in marker_corners:
                criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 40, 0.001 )
                corners_subpixes = cv2.cornerSubPix(image_gray, corners, (11,11), (-1,-1), criteria)
                markers_corners_subpixes.append(corners_subpixes)
        return markers_corners_subpixes
    def detect(self, image, image_t, image_idx, show=False, enable_subpix=False):
        # Defaultly detect use apriltag
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"data/image_{image_idx}_{self.camera_id}.png", image_gray)
        #And then use opencv
        if self.undist_before_detection:
            image_gray = cv2.remap(image_gray, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        marker_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image_gray, self.detector_opencv,
                                                                parameters=self.arucoParams)
        if enable_subpix:
            marker_corners = self.detect_subpix_corner(image_gray, marker_corners)

        self.results[image_idx] = (image_t, marker_corners, ids)
        if show:
            # Draw corners and ids on the image
            image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
            if self.image_accumulate_corners is None:
                self.image_accumulate_corners = np.zeros(image.shape)
            if len(marker_corners) >= self.minimum_tag_num:
                cv2.aruco.drawDetectedMarkers(image, marker_corners, ids, (0, 255, 0))
            
            # Draw corners in image_accumulate_corners
            if len(marker_corners) >= self.minimum_tag_num:
                for corner in marker_corners:
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
                apriltag_pts = self.aprilgrid_3d_points[id[0]].reshape(-1, 3)
                corners = corner.reshape(-1, 2)
                objectPointsInFrame.extend(apriltag_pts)
                imagePointsInFrame.extend(corners)
            objectPointsInFrame = np.array(objectPointsInFrame, dtype=np.float32).reshape(-1, 1, 3)
            imagePointsInFrame = np.array(imagePointsInFrame, dtype=np.float32).reshape(-1, 1, 2)
            objectPoints.append(objectPointsInFrame)
            imagePoints.append(imagePointsInFrame)
        return objectPoints, imagePoints
