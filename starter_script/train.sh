#!/bin/bash

source activate webbrain

NCCL_DEBUG=info python train_t5.py