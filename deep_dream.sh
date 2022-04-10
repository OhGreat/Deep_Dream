#!/bin/bash

python deep_dream.py \
-img random_noise \
-img_out random_noise_2 \
-img_max_dim 1024 \
-ts 512 \
-ss 0.01 \
-or -4 4 \
-os 120 \
-osc 1.2 \
-m inceptionResNet \
-ml 20 40