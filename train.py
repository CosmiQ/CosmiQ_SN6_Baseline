#!/usr/bin/env python

import sys
import argparse

import solaris as sol

def train(inputargs):
    parser = argparse.ArgumentParser(description='SpaceNet 6: Baseline Algorithm Training')
    parser.add_argument('--sardir', help='SAR_Intensity folder')
    parser.add_argument('--labeldir', help='Vector building footprint folder')
    parser.add_argument('--maskdir', help='Where to save building footprint masks')
    


if __name__ == '__main__':
    train(sys.argv[1:])
