#!/usr/bin/env python

import os
import sys
import glob
import math
import uuid
import shutil
import pathlib
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import gdal
import tqdm

import solaris as sol

import model
import loss

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
        #print('Deleting %s' % (srcpath))
        os.remove(srcpath)


def pretrain(args):
    """
    Creates rotated versions of imagery used for training
    as well as raster label masks.
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
    pathlib.Path(args.modeldir).mkdir(exist_ok=True)

    #Look up how to rotate masks and images, if enabled
    if args.rotate:
        assert(args.rotationfile is not None)
        rotationdf = readrotationfile(args.rotationfile)

    #Create masks, with optional rotation and optional size threshold
    #Also copy SAR and optical imagery to local folder, with optional rotation
    #Also create Pandas dataframe of training data
    combodf = pd.DataFrame(columns=['opticalimage',
                                    'sarimage',
                                    'label',
                                    'group'])
    ledge = 592000-450
    redge = 596700-450
    numgroups = 5
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

        #Assign the tile to one of a small number of groups, for setting
        #aside validation data (or for k-fold cross-validation, not used here).
        #Caveats: These groups slightly overlap each other.  Also, they are
        #not of equal size.
        sarfile = gdal.Open(sarpath)
        sartransform = sarfile.GetGeoTransform()
        sarx = sartransform[0]
        groupnum = min(numgroups-1, max(0, math.floor((sarx-ledge) / (redge-ledge) * numgroups)))
        combodf = combodf.append({
            'sarimage': sarpath,
            'opticalimage': opticalpath,
            'label': maskpath,
            'group': groupnum}, ignore_index=True)

    #Write reference CSVs for training
    for i in range(numgroups+1):
        print( '%i: %i' % (i, len(combodf[combodf['group']==i])))
    validationgroup = numgroups - 1
    traindf = combodf[combodf['group'] != validationgroup]
    validdf = combodf[combodf['group'] == validationgroup]
    traindf = traindf.loc[:, ['sarimage', 'label']].rename(columns={'sarimage':'image'})
    validdf = validdf.loc[:, ['sarimage', 'label']].rename(columns={'sarimage':'image'})
    traindf.to_csv(args.traincsv, index=False)
    validdf.to_csv(args.validcsv, index=False)


#Small wrapper class to apply sigmoid and mask to output of a Module class.
class Sigmoid_and_Mask(torch.nn.Module):
    def __init__(self, WrappedClass=model.SeResNext50_9ch_Unet):
        super(Sigmoid_and_Mask, self).__init__()
        self.innermodel = WrappedClass()
    def forward(self, x):
        logits = self.innermodel.forward(x)
        sigout = torch.sigmoid(logits)
        maskout = torch.where(x.view(sigout.size()) > -4, sigout, torch.zeros(sigout.size()).cuda()) #The magic number in this line is -mean/std for image normalization
        return maskout


#Custom model dictionary, defined globally
seresnext50_dict = {
    'model_name': 'SeResNext50_9ch_Unet',#'Sigmoid_and_Mask',
    'weight_path': None,
    'weight_url': None,
    'arch': model.SeResNext50_9ch_Unet#Sigmoid_and_Mask
}


def defineyaml():
    #YAML
    yamlcontents = """
model_name: SeResNext50_9ch_Unet #xdxd_spacenet4 or SeResNext50_9ch_Unet or Sigmoid_and_Mask

model_path:
train: true
infer: true

pretrained: false
nn_framework:  torch
batch_size: 8

data_specs:
  width: 512
  height: 512
  dtype:
  image_type: 32bit
  rescale: false
  rescale_minima: auto
  rescale_maxima: auto
  channels: 4
  label_type: mask
  is_categorical: false
  mask_channels: 1
  val_holdout_frac:
  data_workers:

training_data_csv: '$TRAINCSV'
validation_data_csv: '$VALIDCSV'
inference_data_csv: '$TESTCSV'

training_augmentation:
  augmentations:
    HorizontalFlip:
      p: 0.5
    #RandomRotate90:
    #  p: 0.5
    #Rotate:
    #  limit: 5
    #  border_mode: constant
    #  cval: 0
    #  p: 0.5
    RandomCrop:
      height: 512
      width: 512
      p: 1.0
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0 #255.0 or 65535.0
      p: 1.0
  p: 1.0
  shuffle: true
validation_augmentation:
  augmentations:
    CenterCrop:
      height: 512
      width: 512
      p: 1.0
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0 #255.0 or 65535.0
      p: 1.0
  p: 1.0
inference_augmentation:
  augmentations:
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0 #255.0 or 65535.0
      p: 1.0
  p: 1.0
training:
  epochs:  100000
  steps_per_epoch:
  optimizer: AdamW #Adam or AdamW
  lr: .5e-4
  opt_args:
  loss:
    #bcewithlogits:
    #jaccard:
    dice:
        logits: true
    focal:
        logits: true
    #ScaledTorchDiceLoss:
    #    scale: false
    #    logits: true
    #ScaledTorchFocalLoss:
    #    scale: false
    #    logits: true
    #bcewithlogits:
  loss_weights:
    #bcewithlogits: 10
    #jaccard: 2.5
    dice: 1.0
    focal: 10.0
    #ScaledTorchDiceLoss: 1.0
    #ScaledTorchFocalLoss: 10.0
    #bcewithlogits: 1.0
  metrics:
    training:
    validation:
  checkpoint_frequency: 10
  callbacks:
    model_checkpoint:
      filepath: '$MODELDIR/best.model'
      monitor: val_loss
  model_dest_path: '$MODELDIR/last.model'
  verbose: true

inference:
  window_step_size_x: 512
  window_step_size_y: 512
  output_dir: '$TESTOUTDIR'
"""
    if args.traincsv is not None:
        yamlcontents = yamlcontents.replace('$TRAINCSV', args.traincsv)
    if args.validcsv is not None:
        yamlcontents = yamlcontents.replace('$VALIDCSV', args.validcsv)
    if args.testcsv is not None:
        yamlcontents = yamlcontents.replace('$TESTCSV', args.testcsv)
    if args.modeldir is not None:
        yamlcontents = yamlcontents.replace('$MODELDIR', args.modeldir)
    if args.testoutdir is not None:
        yamlcontents = yamlcontents.replace('$TESTOUTDIR', args.testoutdir)
    yamlfile = open(args.yamlpath, 'w')
    yamlfile.write(yamlcontents)
    yamlfile.close()


def train(args):
    """
    Trains the model.
    """
    print('Train')
    
    #Create YAML file
    defineyaml()

    #Instantiate trainer and train
    config = sol.utils.config.parse(args.yamlpath)
    custom_losses = {'ScaledTorchDiceLoss' : loss.ScaledTorchDiceLoss,
                     'ScaledTorchFocalLoss' : loss.ScaledTorchFocalLoss}
    trainer = sol.nets.train.Trainer(config, custom_model_dict=seresnext50_dict, custom_losses=custom_losses)
    trainer.train()


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
    #Input file paths
    parser.add_argument('--sardir',
                        help='Folder of SAR imagery files')
    parser.add_argument('--opticaldir',
                        help='Folder of optical imagery files')
    parser.add_argument('--labeldir',
                        help='Folder of building footprint vector files')
    parser.add_argument('--rotationfile',
                        help='File of data acquisition directions')
    #Preprocessed file paths
    parser.add_argument('--maskdir',
                        help='Where to save building footprint masks')
    parser.add_argument('--sarprocdir',
                        help='Where to save preprocessed SAR training files')
    parser.add_argument('--opticalprocdir',
                        help='Where to save preprocessed optical image files')
    parser.add_argument('--sartestdir',
                        help='Where to save preprocessed SAR testing files')
    #Reference CSV file paths
    parser.add_argument('--traincsv',
                        help='Where to save reference CSV of training data')
    parser.add_argument('--validcsv',
                        help='Where to save reference CSV of validation data')
    parser.add_argument('--testcsv',
                        help='Where to save reference CSV of testing data')
    #YAML file path
    parser.add_argument('--yamlpath',
                        help='Where to save YAML file')
    #Model weights file path
    parser.add_argument('--modeldir',
                        help='Where to save model weights')
    #Testing (inference) file paths
    parser.add_argument('--testoutdir',
                        help='Where to save test continuous segmentation maps')
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
