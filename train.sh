#!/bin/bash

srcdir=/nfs/data/cosmiq/spacenet/competitions/SN6_buildings/Distributable
dstdir=~/wdata/rotterdam/baseline_data

./baseline.py --pretrain --train --sardir $srcdir/train/SAR-Intensity --opticaldir $srcdir/train/PS-RGB --labeldir $srcdir/train/Buildings --maskdir $dstdir/masks --sarprocdir $dstdir/sartrain --opticalprocdir $dstdir/optical --rotationfile $srcdir/train/SummaryData/SAR_orientations.txt --rotate --mintrainsize 20
