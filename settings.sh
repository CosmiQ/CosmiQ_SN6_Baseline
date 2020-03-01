#!/bin/bash

srcdir=/nfs/data/cosmiq/spacenet/competitions/SN6_buildings/Distributable
dstdir=~/wdata/rotterdam/baseline_data

settings="--sardir $srcdir/train/SAR-Intensity --opticaldir $srcdir/train/PS-RGB --labeldir $srcdir/train/Buildings --rotationfile $srcdir/train/SummaryData/SAR_orientations.txt --maskdir $dstdir/masks --sarprocdir $dstdir/sartrain --opticalprocdir $dstdir/optical --traincsv $dstdir/train.csv --validcsv $dstdir/valid.csv --opticaltraincsv $dstdir/opticaltrain.csv --opticalvalidcsv $dstdir/opticalvalid.csv --testcsv $dstdir/test.csv --yamlpath $dstdir/sar.yaml --opticalyamlpath $dstdir/optical.yaml --modeldir $dstdir/weights --testdir $srcdir/test_public_georeferenced/SAR-Intensity --testprocdir $dstdir/sartest --testoutdir $dstdir/inference_continuous --testbinarydir $dstdir/inference_binary --testvectordir $dstdir/inference_vectors --outputcsv $dstdir/proposal.csv --rotate --transferoptical --mintrainsize 20 --mintestsize 80"
