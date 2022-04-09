#!/bin/bash

python main.py \
-img input/leiden_night.jpg \
-img_max_dim 1024 \
-img_out leiden_night \
-ts 512 \
-ss 0.01 \
-or -3 2 \
-os 60 \
-osc 1.25 \
-m inceptionResNet \
-ml 20 25