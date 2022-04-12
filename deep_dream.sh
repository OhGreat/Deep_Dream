#!/bin/bash

python deep_dream.py \
-img readme_files/input/park_reshaped.jpeg \
-img_out park_6 \
-img_max_dim 1500 \
-ts 500 \
-ss 0.01 \
-or -4 2 \
-os 30 \
-osc 1.2 \
-m inceptionResNet \
-ml  0 8