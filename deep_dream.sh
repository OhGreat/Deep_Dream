#!/bin/bash

python deep_dream.py \
-img input/snow.jpg \
-img_out snow \
-img_max_dim 1024 \
-ts 600 \
-ss 0.01 \
-or -4 3 \
-os 50 \
-osc 1.25 \
-m inceptionResNet \
-ml 10 30