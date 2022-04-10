#!/bin/bash

python deep_dream.py \
-img random_noise \
-img_out random_noise_3 \
-img_max_dim 1024 \
-ts 512 \
-ss 0.01 \
-or -4 2 \
-os 100 \
-osc 1.3 \
-m inceptionResNet \
-ml 25 30