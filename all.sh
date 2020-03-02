#!/bin/bash

source settings.sh

./baseline.py --pretrain --train --pretest --test --eval $settings
