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

import argparse
import os
import sys
from calibrate_single_thread import parse_bag_and_calibrate_quad_in_single_thread
import cv2
import numpy as np
intrinsic_init = np.array([1.24, 813, 812, 640, 360])
D_init = np.array([-0.38871409,  0.14562629, -0.00313268, -0.0010537])

if __name__ == "__main__":
    # Create the parser for argparse. User should input the file path of bag
    # file and the output directory.
    parser = argparse.ArgumentParser(description="This script is used to "
                                                 "extract the image data from "
                                                 "the rosbag file.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="The file path of the rosbag file.")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="The output directory.")
    parser.add_argument("-s", "--step", type=int, default=3, help="The step to "
                        "extract the image data. Default is 3.")
    parser.add_argument("--show", action="store_true",
                        help="Show the image.")
    # Parse the arguments
    args = parser.parse_args()
    
    cv2.setNumThreads(10)
    
    parse_bag_and_calibrate_quad_in_single_thread(args.input, show=args.show, step=args.step,
                                                  intrinsic_init=intrinsic_init, D_init=D_init, undist_before_detection=False)
