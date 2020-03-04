#!/bin/bash

testdatapath=$1
outputpath=$2
testdataargs="\
--testdir $testdatapath/SAR-Intensity \
--outputcsv $outputpath \
"

source settings.sh

./baseline.py --test --eval $testdataargs $settings
