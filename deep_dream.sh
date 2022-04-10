#!/bin/bash

python deep_dream.py \
-img random_noise \
-img_out random_noise_3 \
-img_max_dim 1024 \
-ts 256 \
-ss 0.01 \
-or -4 2 \
-os 80 \
-osc 1.2 \
-m inceptionResNet \
-ml 25 30