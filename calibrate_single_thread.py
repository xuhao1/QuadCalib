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
from Calibrator.calibrator import IntrinsicCalibrator
import cv2 as cv
import numpy as np
from tqdm import tqdm
from BagUtils import CountBagImageNumber
from QuadUtils import split_quad_image

def concate_quad_images(imgs):
    img = np.concatenate((np.concatenate((imgs[0], imgs[1]), axis=1),
                                    np.concatenate((imgs[2], imgs[3]), axis=1)), axis=0)
    cv.line(img, (0, img.shape[0]//2), (img.shape[1], img.shape[0]//2), (255, 255, 0), 2)
    cv.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (255, 255, 0), 2)
    return img

def parse_bag_and_calibrate_quad_in_single_thread(rosbag_path, show=False, step=3, intrinsic_init=None, D_init=None, undist_before_detection=False):
    import rosbag
    detectors = [Detector(camera_id=i, intrinsic_init=intrinsic_init, D_init=D_init,
                          undist_before_detection=undist_before_detection) for i in range(4)]
    bag = rosbag.Bag(rosbag_path)
    total_image_num = CountBagImageNumber(rosbag_path)
    pbar = tqdm(total=total_image_num)

    frame_id = 0
    calibrator = IntrinsicCalibrator()
    print(f"Total image number: {total_image_num}")
    shape = None
    for topic, msg, t in bag.read_messages():
        img = None
        if msg._type == "sensor_msgs/Image":
            print("Raw image not supported yet")
            continue
        elif msg._type == "sensor_msgs/CompressedImage":
            img = msg.data
            if msg.format == "jpeg" or msg.format == "jpg":
                # Decode the image data
                img = cv.imdecode(np.frombuffer(img, np.uint8), cv.IMREAD_ANYCOLOR)
            else:
                print("Unsupported image format.")
                continue
        else:
            continue
        if frame_id % step != 0:
            frame_id += 1
            continue
        imgs = split_quad_image(img)
        cul_img = [None] * 4
        for i in range(4):
            imgs[i], cul_img[i] = detectors[i].detect(imgs[i], t, frame_id, show=show)
            if shape is None:
                shape = imgs[i].shape[:2]
        # Use tqdm to show the progress
        pbar.update(step)
        if show:
            show_image = concate_quad_images(imgs)
            cv.putText(show_image, f"Frame {frame_id}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv.imshow("Image", show_image)
            cv.imshow("CulImage", concate_quad_images(cul_img))
            # Draw the frame_id
            key = cv.waitKey(1)
            if key == ord('q') or key == 27:
                break
        frame_id += 1
    pbar.close()
    for i in range(4):
        print(f"Start to calibrate camera {i}")
        retval, K, D, rvecs, tvecs = calibrator.calibrate_mono(detectors[i], shape)
    bag.close()
    cv.destroyAllWindows()
