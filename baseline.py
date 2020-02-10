#!/usr/bin/env python

import sys
import argparse

import solaris as sol

def pretrain(args):
    print('pretrain')

def train(args):
    print('train')

def pretest(args):
    print('pretest')
    
def test(args):
    print('test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpaceNet 6: Baseline Algorithm Training')
    #Which operations to carry out
    parser.add_argument('--pretrain', action='store_true',
                        help='Whether to format training data')
    parser.add_argument('--train', action='store_true',
                        help='Whether to train model')
    parser.add_argument('--pretest', action='store_true',
                        help='Whether to format testing data')
    parser.add_argument('--test', action='store_true',
                        help='Whether to test model')
    #File paths
    parser.add_argument('--sardir',
                        help='SAR_Intensity folder')
    parser.add_argument('--labeldir',
                        help='Vector building footprint folder')
    parser.add_argument('--maskdir',
                        help='Where to save building footprint masks')
    args = parser.parse_args(sys.argv[1:])

    if args.pretrain:
        pretrain(args)
    if args.train:
        train(args)
    if args.pretest:
        pretest(args)
    if args.test:
        test(args)
