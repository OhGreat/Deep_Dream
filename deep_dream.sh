#!/bin/bash

python deep_dream.py \
-img readme_files/input/park_reshaped.jpeg \
-img_out park_6 \
-img_max_dim 1500 \
-ts 512 \
-ss 0.01 \
-or -4 2 \
-os 50 \
-osc 1.1 \
-m inceptionResNet \
-ml 20 35