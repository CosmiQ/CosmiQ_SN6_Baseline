#!/usr/bin/env python

import os
import sys
import glob
import pathlib
import argparse
import geopandas as gpd
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
    sarpaths.sort()
    labelpaths = glob.glob(os.path.join(args.labeldir, '*.geojson'))
    labelpaths.sort()
    maskpaths = [os.path.join(args.maskdir, os.path.basename(sarpath)) for sarpath in sarpaths]

    #Create empty folders to hold masks
    pathlib.Path(args.maskdir).mkdir(exist_ok=True)
    oldmasks = glob.glob(os.path.join(args.maskdir, '*.tif'))
    for oldmask in oldmasks:
        os.remove(oldmask)

    #Create masks, with optional size threshold
    for i, (sarpath, labelpath, maskpath) in tqdm.tqdm(enumerate(zip(sarpaths, labelpaths, maskpaths)), total=len(sarpaths)):
        gdf = gpd.read_file(labelpath)
        if args.mintrainsize is not None:
            cut = gdf.area > float(args.mintrainsize)
            gdf = gdf.loc[cut]
        maskdata = sol.vector.mask.footprint_mask(
            df=gdf,
            reference_im=sarpath,
            out_file=maskpath
        )
        if i>10:
            break

    #Rotate masks and images, if enabled
    if args.orient:
        assert(args.orientfile is not None)
        

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
                        help='Folder of SAR imagery files')
    parser.add_argument('--opticaldir',
                        help='Folder of optical imagery files')
    parser.add_argument('--labeldir',
                        help='Folder of building footprint vector files')
    parser.add_argument('--maskdir',
                        help='Where to save building footprint masks')
    parser.add_argument('--sarprocdir',
                        help='Where to save preprocessed SAR imagery files')
    parser.add_argument('--opticalprocdir',
                        help='Where to save preprocessed optical image files')
    parser.add_argument('--orientfile',
                        help='File of data acquisition directions')
    #Algorithm settings
    parser.add_argument('--orient', action='store_true',
                        help='Rotate tiles to align imaging direction')
    parser.add_argument('--mintrainsize',
                        help='Minimum building size (m^2) for training')
    args = parser.parse_args(sys.argv[1:])

    if args.pretrain:
        pretrain(args)
    if args.train:
        train(args)
    if args.pretest:
        pretest(args)
    if args.test:
        test(args)
