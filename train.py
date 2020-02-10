#!/usr/bin/env python

import sys
import argparse

def train(inputargs):
    parser = argparse.ArgumentParser(description='SpaceNet 6: Baseline Algorithm Training')
    print('hello world')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpaceNet 6 : Multi-Sensor All-Weather Mapping : Baseline Algorithm')
    parser.add_argument('-r', help='Train model')
    parser.add_argument('-s', help='Test model')
    main(sys.argv[1:])
