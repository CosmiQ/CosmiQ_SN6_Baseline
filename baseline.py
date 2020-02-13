#!/usr/bin/env python

import os
import sys
import glob
import uuid
import shutil
import pathlib
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import gdal
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
    tilename = os.path.splitext(os.path.basename(tilepath))[0]
    stripname = '_'.join(tilename.split('_')[-4:-2])
    rotation = rotationdf.loc[stripname].squeeze()
    return rotation


def copyrotateimage(srcpath, dstpath, rotate=False, deletesource=False):
    """
    Copying with rotation:  Copies a TIFF image from srcpath to dstpath,
    rotating the image by 180 degrees if specified.
    """
    #Handles special case where source path and destination path are the same
    if srcpath==dstpath:
        if not rotate:
            #Then there's nothing to do
            return
        else:
            #Move file to temporary location before continuing
            srcpath = srcpath + str(uuid.uuid4())
            shutil.move(dstpath, srcpath)
            deletesource = True

    if not rotate:
        shutil.copy(srcpath, dstpath, follow_symlinks=True)
    else:
        driver = gdal.GetDriverByName('GTiff')
        tilefile = gdal.Open(srcpath)
        copyfile = driver.CreateCopy(dstpath, tilefile, strict=0)
        numbands = copyfile.RasterCount
        for bandnum in range(1, numbands+1):
            banddata = tilefile.GetRasterBand(bandnum).ReadAsArray()
            banddata = np.fliplr(np.flipud(banddata)) #180 deg rotation
            copyfile.GetRasterBand(bandnum).WriteArray(banddata)
        copyfile.FlushCache()
        copyfile = None
        tilefile = None

    if deletesource:
        pass#os.remove(srcpath)


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
    sarprocpaths = [os.path.join(args.sarprocdir, os.path.basename(sarpath)) for sarpath in sarpaths]
    if args.opticaldir is not None:
        opticalpaths = glob.glob(os.path.join(args.opticaldir, '*.tif'))
        opticalpaths.sort()
        opticalprocpaths = [os.path.join(args.opticalprocdir, os.path.basename(opticalpath)) for opticalpath in opticalpaths]
    else:
        opticalpaths = [''] * len(sarpaths)
        opticalprocpaths = [''] * len(sarpaths)

    #Create empty folders to hold masks, processed SAR, & processed optical
    folders = [args.maskdir, args.sarprocdir]
    if args.opticalprocdir is not None:
        folders.append(args.opticalprocdir)
    for folder in folders:
        makeemptyfolder(folder)

    #Look up how to rotate masks and images, if enabled
    if args.rotate:
        assert(args.rotationfile is not None)
        rotationdf = readrotationfile(args.rotationfile)

    #Create masks, with optional rotation and optional size threshold
    #Also copy SAR and optical imagery to local folder, with optional rotation
    for i, (sarpath, opticalpath, labelpath, maskpath, sarprocpath, opticalprocpath) in tqdm.tqdm(enumerate(zip(sarpaths, opticalpaths, labelpaths, maskpaths, sarprocpaths, opticalprocpaths)), total=len(sarpaths)):
        #Generate mask
        gdf = gpd.read_file(labelpath)
        if args.mintrainsize is not None:
            cut = gdf.area > float(args.mintrainsize)
            gdf = gdf.loc[cut]
        maskdata = sol.vector.mask.footprint_mask(
            df=gdf,
            reference_im=sarpath,
            out_file=maskpath
        )
        #Optionally rotate mask
        if args.rotate:
            rotationflag = lookuprotation(sarpath, rotationdf)
        else:
            rotationflag = 0
        if rotationflag==1:
            copyrotateimage(maskpath, maskpath, rotate=True)
        #Copy SAR and optical imagery, with optional rotation
        rotationflagbool = rotationflag == 1
        copyrotateimage(sarpath, sarprocpath, rotate=rotationflagbool)
        if args.opticaldir is not None:
            copyrotateimage(opticalpath, opticalprocpath, rotate=rotationflagbool)

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
