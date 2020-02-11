#!/bin/bash
./baseline.py --pretrain --train --sardir /nfs/data/cosmiq/spacenet/competitions/SN6_buildings/Distributable/train/SAR-Intensity --labeldir /nfs/data/cosmiq/spacenet/competitions/SN6_buildings/Distributable/train/Buildings --maskdir ~/wdata/rotterdam/baseline_data/masks --orient --mintrainsize 20
