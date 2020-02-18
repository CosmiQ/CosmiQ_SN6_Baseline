#!/bin/bash

srcdir=/nfs/data/cosmiq/spacenet/competitions/SN6_buildings/Distributable
dstdir=~/wdata/rotterdam/baseline_data

./baseline.py --pretest --sardir $srcdir/train/SAR-Intensity --opticaldir $srcdir/train/PS-RGB --labeldir $srcdir/train/Buildings --rotationfile $srcdir/train/SummaryData/SAR_orientations.txt --maskdir $dstdir/masks --sarprocdir $dstdir/sartrain --opticalprocdir $dstdir/optical --traincsv $dstdir/train.csv --validcsv $dstdir/valid.csv --testcsv $dstdir/test.csv --yamlpath $dstdir/sar.yaml --modeldir $dstdir/models --testdir $dstdir/test_public/SAR-Intensity --testprocdir $dstdir/sartest --testoutdir $dstdir/inference_output --rotate --mintrainsize 20
