#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main_dpe.py   --config configs \
                                            --wandb-log \
                                            --datasets caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 \
                                            --backbone RN50 \
                                            # --coop