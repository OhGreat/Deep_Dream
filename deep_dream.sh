#!/bin/bash

python deep_dream.py \
-img input/input_reshaped/field_reshaped.jpg \
-img_out field_2 \
-img_max_dim 1024 \
-ts 600 \
-ss 0.01 \
-or -3 3 \
-os 50 \
-osc 1.25 \
-m inceptionResNet \
-ml 12 35