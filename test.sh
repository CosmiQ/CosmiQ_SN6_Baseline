#!/bin/bash

source activate solaris

testdatapath=$1
outputpath=$2
testdataargs="\
--testdir $testdatapath/SAR-Intensity \
--outputcsv $outputpath \
"

source settings.sh

./baseline.py --pretest --test $testdataargs $settings
