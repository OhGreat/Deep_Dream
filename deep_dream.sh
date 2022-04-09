#!/bin/bash

python main.py \
-img input/lighthouse.jpg \
-img_out lighthouse \
-img_max_dim 1024 \
-ts 512 \
-ss 0.01 \
-or -3 3 \
-os 60 \
-osc 1.3 \
-m inceptionV3 \
-ml 0 2