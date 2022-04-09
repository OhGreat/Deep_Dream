#!/bin/bash

python main.py \
-img input/in.jpg \
-img_out out \
-img_max_dim 1024 \
-ts 512 \
-ss 0.01 \
-or -4 1 \
-os 60 \
-osc 1.2 \
-m inceptionResNet \
-ml 20 30