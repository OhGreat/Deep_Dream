#!/bin/bash

python deep_dream.py \
-img input/park.jpg \
-img_out park_6 \
-img_max_dim 1500 \
-ts 512 \
-ss 0.01 \
-or -40 2 \
-os 5 \
-osc 1.02 \
-m inceptionResNet \
-ml 0 30