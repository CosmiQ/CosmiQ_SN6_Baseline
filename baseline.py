#!/usr/bin/env python

import os
import sys
import glob
import argparse
import tqdm

import solaris as sol


def pretrain(args):
    """
    Creates formatted versions of data used for training,
    including raster label masks.
    """
    print('Pretrain')
    assert(args.sardir is not None and args.labeldir is not None and args.maskdir is not None)

    #Get paths to relevant files
    sarpaths = glob.glob(os.path.join(args.sardir, '*.tif'))
    tilepaths.sort()
    labelpaths = glob.glob(os.path.join(args.labeldir, '*.geojson'))
    labelpaths.sort()
    maskpaths = [os.path.join(args.maskdir, os.path.basename(tilepath)) for tilepath in tilepaths]

    #Create empty folder to hold masks
    pathlib.Path(args.maskdir).mkdir(exist_ok=True)
    oldmasks = glob.glob(os.path.join(args.maskdir, '*.tif'))
    for oldmask in oldmasks:
        os.remove(oldmask)

    #Create and save masks
    for i, (sarpath, labelpath, maskpath) in tqdm.tqdm(enumerate(zip(sarpaths, labelpaths, maskpaths)), total=len(sarpaths)):
        gdf = gpd.read_file(labelpath)
        maskdata = sol.vector.mask.footprint_mask(
            df=gdf,
            reference_im=sarpath,
            out_file=maskpath
        )

def train(args):
    print('Train')

def pretest(args):
    print('Pretest')
    
def test(args):
    print('Test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpaceNet 6 Baseline Algorithm')
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
