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

from AprilDetection.detection import Detector

def parse_bag_and_calibrate_quad_in_single_thread(rosbag_path, show=False):
    import rosbag
    import cv2 as cv
    from Utils import splitImage
    import numpy as np
    
    detectors = [Detector(camera_id=i) for i in range(4)]
    
    bag = rosbag.Bag(rosbag_path)
    frame_id = 0
    for topic, msg, t in bag.read_messages():
        if msg._type == "sensor_msgs/Image":
            print("Raw image not supported yet")
            continue
        elif msg._type == "sensor_msgs/CompressedImage":
            img = msg.data
        else:
            continue
        if msg.format == "jpeg" or msg.format == "jpg":
            # Decode the image data
            img = cv.imdecode(np.frombuffer(img, np.uint8), cv.IMREAD_ANYCOLOR)
            imgs = splitImage(img)
            for i in range(4):
                imgs[i] = detectors[i].detect(imgs[i], t, frame_id, show=show)
            # Concatenate the images to 2x2
            img = np.concatenate((np.concatenate((imgs[0], imgs[1]), axis=1),
                                    np.concatenate((imgs[2], imgs[3]), axis=1)), axis=0)
            frame_id += 1
            if show:
                cv.imshow("Image", img)
                cv.waitKey(1)
                
    