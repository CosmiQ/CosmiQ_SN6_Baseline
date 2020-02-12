#!/usr/bin/env python

import os
import sys
import glob
import shutil
import pathlib
import argparse
import pandas as pd
import geopandas as gpd
import tqdm

import solaris as sol


def makeemptyfolder(path):
    """
    Create an empty folder, deleting anything there already
    """
    shutil.rmtree(path, ignore_errors=True)
    pathlib.Path(path).mkdir(exist_ok=True)


def readrotationfile(path):
    """
    Reads SAR_orientations file, which lists whether each strip was imaged
    from the north (denoted by 0) or from the south (denoted by 1).

    Example of using the resulting dataframe:
    rotationdf.loc['20190823161251_20190823161546'].squeeze()
    """
    rotationdf = pd.read_csv(args.rotationfile,
                             sep=' ',
                             index_col=0,
                             names=['strip', 'direction'],
                             header=None)
    rotationdf['direction'] = rotationdf['direction'].astype(int)
    return rotationdf

def lookuprotation(tilepath, rotationdf):
    """
    Looks up the SAR_orientations value for a tile based on its filename
    """
    pass


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

    #Create empty folders to hold masks, processed SAR, & processed optical
    folders = [args.maskdir, args.sarprocdir]
    if args.opticalprocdir is not None:
        folders.append(args.opticalprocdir)
    for folder in folders:
        makeemptyfolder(folder)

    #Rotate masks and images, if enabled
    if args.rotate:
        assert(args.rotationfile is not None)
        rotationdf = readrotationfile(args.rotationfile)

    #Copy SAR imagery to local folder, with optional rotation

    #Create masks, with optional rotation and optional size threshold
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
        if i>5:
            break


        

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
    parser.add_argument('--rotationfile',
                        help='File of data acquisition directions')
    #Algorithm settings
    parser.add_argument('--rotate', action='store_true',
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
