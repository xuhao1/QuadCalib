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
    # If --show is set, the image will be shown.
    parser.add_argument("--show", action="store_true",
                        help="Show the image.")
    # Parse the arguments
    args = parser.parse_args()
    parse_bag_and_calibrate_quad_in_single_thread(args.input, show=args.show)
    