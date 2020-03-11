#!/bin/bash

source activate solaris

traindatapath=$1
traindataargs="\
--sardir $traindatapath/SAR-Intensity \
--opticaldir $traindatapath/PS-RGB \
--labeldir $traindatapath/geojson_buildings \
--rotationfile $traindatapath/SummaryData/SAR_orientations.txt \
"

source settings.sh

./baseline.py --pretrain --train $traindataargs $settings
