#!/bin/bash

python main.py \
-img input/selfie.jpg \
-img_max_dim 1024 \
-ts 512 \
-ss 0.01 \
-or -3 1 \
-os 70 \
-osc 1.25 \
-m inceptionResNet \
-ml 0 6