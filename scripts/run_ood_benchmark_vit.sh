#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main_dpe.py   --config configs \
                                            --wandb-log \
                                            --datasets I/A/V/R/S \
                                            --backbone ViT-B/16 \
                                            # --coop