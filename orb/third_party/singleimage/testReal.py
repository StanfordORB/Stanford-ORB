import torch
import numpy as np
from torch.autograd import Variable
import argparse
import os
import models
import torchvision.utils as vutils
import torch.nn as nn
from PIL import Image
import glob
import os.path as osp
import scipy.ndimage as ndimage
import struct
import utils
import dataLoader
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--imList', default='imList.txt', help='path to real images')
parser.add_argument('--output', default = 'output', help='path to saving results')
parser.add_argument('--modelRootInit', default = None, help = 'the directory where the initialization trained model is save')
parser.add_argument('--modelRootsRefine', nargs='+', default=[None, None], help='the directory where the refine models are saved')
parser.add_argument('--modelRootGlob', default = None, help = 'the directory where the global illumination model is saved')
parser.add_argument('--epochIdInit', type=int, default = 14, help = 'the training epoch of the initial network')
parser.add_argument('--epochIdsRefine', nargs = '+', type=int, default = [7, 5], help='the training epoch of the refine network')
parser.add_argument('--epochIdGlob', type=int, default=17, help='the traing epoch of the global illuminationn prediction network')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic testing setting
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()
print(opt)

opt.gpuId = 0

if not osp.isdir(opt.output):
    os.system('mkdir -p %s' % opt.output )

if opt.modelRootInit is None:
    opt.modelRootInit = 'check_initEnvGlob_cascade0'

opt.cascadeLevel = 2
opt.renderMode = 2
opt.refineInputMode = 1
opt.batchSize = 1

if len(opt.modelRootsRefine) != opt.cascadeLevel or opt.modelRootsRefine[0] is None:
    opt.modelRootsRefine = []
    for n in range(1, opt.cascadeLevel+1):
        root = 'check_cascadeEnvGlob'
        root += '_render{0}'.format(opt.renderMode)
        root += '_refine{0}'.format(opt.refineInputMode)
        root += '_cascade{0}'.format(n)
        opt.modelRootsRefine.append(root)

if opt.modelRootGlob is None:
    opt.modelRootGlob = 'check_globalillumination'

opt.seed = 0
torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################
# initalize tensors
segBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize) )  # FIXME changed from 3 to 1
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imBgBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )

# Initial Network
encoderInit = models.encoderInitial()
albedoInit = models.decoderInitial(mode=0)
normalInit = models.decoderInitial(mode=1)
roughInit = models.decoderInitial(mode=2)
depthInit = models.decoderInitial(mode=3)
envInit = models.envmapInitial()
for param in encoderInit.parameters():
    param.requires_grad = False
for param in albedoInit.parameters():
    param.requires_grad = False
for param in normalInit.parameters():
    param.requires_grad = False
for param in roughInit.parameters():
    param.requires_grad = False
for param in depthInit.parameters():
    param.requires_grad = False
for param in envInit.parameters():
    param.requires_grad = False


# Refine Network
encoderRefs, albedoRefs = [], []
normalRefs, roughRefs = [], []
depthRefs, envRefs = [], []
for n in range(0, opt.cascadeLevel):
    encoderRefs.append( models.refineEncoder() )
    for param in encoderRefs[n].parameters():
        param.requires_grad = False

    albedoRefs.append( models.refineDecoder(mode=0) )
    for param in albedoRefs[n].parameters():
        param.requires_grad = False

    normalRefs.append( models.refineDecoder(mode=1) )
    for param in normalRefs[n].parameters():
        param.requires_grad = False

    roughRefs.append( models.refineDecoder(mode=2) )
    for param in roughRefs[n].parameters():
        param.requires_grad = False

    depthRefs.append( models.refineDecoder(mode=3) )
    for param in depthRefs[n].parameters():
        param.requires_grad = False

    envRefs.append( models.refineEnvDecoder() )
    for param in envRefs[n].parameters():
        param.requires_grad = False

renderLayer = models.renderingLayer(gpuId = opt.gpuId, isCuda = opt.cuda )

# Global illumination
globIllu1to2 = models.globalIllumination()
globIllu2to3 = models.globalIllumination()
for param in globIllu1to2.parameters():
    param.requires_grad = False
for param in globIllu2to3.parameters():
    param.requires_grad = False
#########################################

#########################################
# Load the trained model
encoderInit.load_state_dict(torch.load('{0}/encoderInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit) ) )
encoderInit = encoderInit.eval()
albedoInit.load_state_dict(torch.load('{0}/albedoInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit) ) )
albedoInit = albedoInit.eval()
normalInit.load_state_dict(torch.load('{0}/normalInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit) ) )
normalInit = normalInit.eval()
roughInit.load_state_dict(torch.load('{0}/roughInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit) ) )
roughInit = roughInit.eval()
depthInit.load_state_dict(torch.load('{0}/depthInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit) ) )
depthInit = depthInit.eval()
envInit.load_state_dict(torch.load('{0}/envInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit) ) )
envInit = envInit.eval()

globIllu1to2.load_state_dict(torch.load('{0}/globIllu1to2_{1}.pth'.format(opt.modelRootGlob, opt.epochIdGlob) ) )
globIllu1to2 = globIllu1to2.eval()
globIllu2to3.load_state_dict(torch.load('{0}/globIllu2to3_{1}.pth'.format(opt.modelRootGlob, opt.epochIdGlob) ) )
globIllu2to3 = globIllu2to3.eval()

for n in range(0, opt.cascadeLevel):
    encoderRefs[n].load_state_dict(torch.load('{0}/encoderRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n+1, opt.epochIdsRefine[n] ) ) )
    encoderRefs[n] = encoderRefs[n].eval()
    albedoRefs[n].load_state_dict(torch.load('{0}/albedoRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n+1, opt.epochIdsRefine[n]) ) )
    albedoRefs[n] = albedoRefs[n].eval()
    normalRefs[n].load_state_dict(torch.load('{0}/normalRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n+1, opt.epochIdsRefine[n]) ) )
    normalRefs[n] = normalRefs[n].eval()
    roughRefs[n].load_state_dict(torch.load('{0}/roughRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n+1, opt.epochIdsRefine[n]) ) )
    roughRefs[n] = roughRefs[n].eval()
    depthRefs[n].load_state_dict(torch.load('{0}/depthRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n+1, opt.epochIdsRefine[n]) ) )
    depthRefs[n] = depthRefs[n].eval()
    envRefs[n].load_state_dict(torch.load('{0}/envRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n+1, opt.epochIdsRefine[n] ) ) )
    envRefs[n] = envRefs[n].eval()
#########################################

##############  ######################
# Send things into GPU
if opt.cuda:
    segBatch = segBatch.cuda(opt.gpuId)
    imBatch = imBatch.cuda(opt.gpuId)
    imBgBatch = imBgBatch.cuda(opt.gpuId)

    encoderInit = encoderInit.cuda(opt.gpuId)
    albedoInit = albedoInit.cuda(opt.gpuId)
    normalInit = normalInit.cuda(opt.gpuId)
    roughInit = roughInit.cuda(opt.gpuId)
    depthInit = depthInit.cuda(opt.gpuId)
    envInit = envInit.cuda(opt.gpuId)
    globIllu1to2 = globIllu1to2.cuda(opt.gpuId)
    globIllu2to3 = globIllu2to3.cuda(opt.gpuId)
    for n in range(0, opt.cascadeLevel):
        encoderRefs[n] = encoderRefs[n].cuda(opt.gpuId)
        albedoRefs[n] = albedoRefs[n].cuda(opt.gpuId)
        normalRefs[n] = normalRefs[n].cuda(opt.gpuId)
        roughRefs[n] = roughRefs[n].cuda(opt.gpuId)
        depthRefs[n] = depthRefs[n].cuda(opt.gpuId)
        envRefs[n] = envRefs[n].cuda(opt.gpuId)
####################################


####################################
j = 0
with open(opt.imList, 'r') as imIn:
    imList = imIn.readlines()
    imList = [x.strip() for x in imList ]

for n in range(0, len(imList ) ):

    print('Processing %d/%d' % (n+1, len(imList ) ) )
    imgName = imList[n]

    # Read the image with background
    imBg = Image.open(imgName )
    imBg = np.asarray(imBg ).astype(np.float32)
    imBg = (imBg / 255.0) ** (2.2)
    imBg = (2*imBg - 1).transpose([2, 0, 1] )[np.newaxis, :, :, :]

    # Read the segmentation mask
    segName = imgName.replace('input', 'mask')
    seg = Image.open(segName )
    seg = np.asarray(seg ).astype(np.float32) / 255.0
    if seg.shape[2] > 1:
        seg = seg[:, :, 0]
    seg = (seg > 0.999).astype(dtype = int)
    seg = ndimage.binary_erosion(seg, structure = np.ones( (4,4) ) ).astype(dtype=np.float32)
    seg = seg[np.newaxis, np.newaxis, :, :]

    im = imBg * seg

    # Load data from cpu to gpu
    segBatch.data.resize_(seg.shape )
    segBatch.data.copy_(torch.from_numpy(seg ) )

    imBatch.data.resize_(im.shape )
    imBatch.data.copy_( torch.from_numpy(im ) )
    imBgBatch.data.resize_(imBg.shape )
    imBgBatch.data.copy_( torch.from_numpy(imBg ) )

    albedoPreds = []
    normalPreds = []
    roughPreds = []
    depthPreds = []
    SHPreds = []
    globalIllu1s = []
    globalIllu2s = []
    globalIllu3s = []
    renderedEnvs = []

    # Initial Prediction
    inputInit = torch.cat([imBatch, imBgBatch, segBatch], dim=1 )
    x1, x2, x3, x4, x5, x = encoderInit(inputInit )
    albedoPred = albedoInit(x1, x2, x3, x4, x5, x) * segBatch.expand_as(imBatch )
    normalPred = normalInit(x1, x2, x3, x4, x5, x) * segBatch.expand_as(imBatch )
    roughPred = roughInit(x1, x2, x3, x4, x5, x) * segBatch
    depthPred = depthInit(x1, x2, x3, x4, x5, x) * segBatch
    SHPred = envInit(x)

    globalIllu1 = renderLayer.forward(albedoPred, normalPred,
            roughPred, depthPred, segBatch )
    renderedEnv = renderLayer.forwardEnv(albedoPred,
            normalPred, roughPred, SHPred, segBatch)
    inputGlob2 = torch.cat([ (2*globalIllu1-1), \
            albedoPred, normalPred, roughPred, depthPred, segBatch], dim=1)
    globalIllu2 = globIllu1to2(inputGlob2).detach()
    inputGlob3 = torch.cat([globalIllu2, albedoPred, \
            normalPred, roughPred, depthPred, segBatch], dim=1)
    globalIllu3 = globIllu2to3(inputGlob3).detach()
    globalIllu2 = 0.5*(globalIllu2 + 1)
    globalIllu3 = 0.5*(globalIllu3 + 1)


    albedoPreds.append(albedoPred)
    normalPreds.append(normalPred)
    roughPreds.append(roughPred)
    depthPreds.append(depthPred)
    SHPreds.append(SHPred)
    globalIllu1s.append(globalIllu1)
    globalIllu2s.append(globalIllu2)
    globalIllu3s.append(globalIllu3)
    renderedEnvs.append(renderedEnv)

    # Refine the BRDF reconstruction
    for n in range(0, opt.cascadeLevel):
        albedoPred = (albedoPreds[n] * segBatch.expand_as(albedoPred) ).detach()
        normalPred = (normalPreds[n] * segBatch.expand_as(normalPred) ).detach()
        roughPred = (roughPreds[n] * segBatch.expand_as(roughPred) ).detach()
        depthPred = (depthPreds[n] * segBatch.expand_as(depthPred) ).detach()
        SHPred = SHPreds[n].detach()

        globalIllu1 = globalIllu1s[n] * segBatch.expand_as(imBatch ).detach()

        renderedImg = renderedEnv + globalIllu1 + \
                globalIllu2 + globalIllu3
        error = (renderedImg - 0.5*(imBatch + 1) ) * segBatch.expand_as(imBatch)
        inputRefine = torch.cat( [albedoPred, normalPred, roughPred, depthPred, segBatch, \
                imBatch, imBgBatch, error], dim=1)

        x1, x3 = encoderRefs[n](inputRefine.detach() )
        albedoPred = albedoRefs[n](x1, x3) * segBatch.expand_as(imBatch)
        normalPred = normalRefs[n](x1, x3) * segBatch.expand_as(imBatch)
        roughPred = roughRefs[n](x1, x3) * segBatch
        depthPred = depthRefs[n](x1, x3) * segBatch

        SHPred = envRefs[n](x3, SHPred)
        globalIllu1 = renderLayer.forward(albedoPred, normalPred,
                roughPred, depthPred, segBatch )

        globalIllu2 = globIllu1to2(torch.cat([ (2*globalIllu1 - 1), \
                albedoPred, normalPred, roughPred, depthPred, segBatch], dim=1) ).detach()
        globalIllu3 = globIllu2to3(torch.cat([globalIllu2, albedoPred, \
                normalPred, roughPred, depthPred, segBatch], dim=1) ).detach()
        globalIllu2 = 0.5 * (globalIllu2 + 1) * segBatch.expand_as(imBatch )
        globalIllu3 = 0.5 * (globalIllu3 + 1) * segBatch.expand_as(imBatch )
        renderedEnv = renderLayer.forwardEnv(albedoPred, normalPred, roughPred, SHPred, segBatch)

        albedoPreds.append(albedoPred )
        normalPreds.append(normalPred )
        roughPreds.append(roughPred )
        depthPreds.append(depthPred )
        SHPreds.append(SHPred )
        globalIllu1s.append(globalIllu1 )
        globalIllu2s.append(globalIllu2 )
        globalIllu3s.append(globalIllu3 )
        renderedEnvs.append(renderedEnv )

    imgName = osp.join(opt.output, imgName.split('/')[-1] )
    segName = imgName.replace('input', 'mask')

    utils.writeImageToFile(0.5*(imBgBatch + 1), [imgName], isGama = True)
    utils.writeImageToFile(segBatch, [segName], isGama = False )

    # Save the predicted results
    for n in range(0, len(albedoPreds) ):
        albedoName = []
        albedoName.append(imgName.replace('input', 'albedo_%d' % n ) )
        utils.writeImageToFile( 0.5*(albedoPreds[n]+1) * segBatch.expand_as(albedoPreds[n] ),
                albedoName, isGama = False )

    for n in range(0, len(normalPreds) ):
        normalName = []
        normalName.append(imgName.replace('input', 'normal_%d' % n ) )
        utils.writeImageToFile( 0.5*(normalPreds[n]+1) * segBatch.expand_as(normalPreds[n] ),
                normalName, isGama = False )

    for n in range(0, len(roughPreds) ):
        roughName = []
        roughName.append(imgName.replace('input', 'rough_%d' % n) )
        utils.writeImageToFile( 0.5*(roughPreds[n]+1) * segBatch.expand_as(roughPreds[n] ),
                roughName, isGama = False )

    for n in range(0, len(depthPreds) ):
        depthName, depthImName = [], []
        depthName.append(imgName.replace('input', 'depth_%d' % n).replace('png', 'hdf5') )
        # utils.writeDataToFile(depthPreds[n], depthName )

        depthImName.append(imgName.replace('input', 'depth_%d' % n) )
        depthOut = 1 / torch.clamp(depthPreds[n], 1e-6, 10)
        depthOut = (depthOut - 0.25) /0.8
        utils.writeImageToFile( depthOut * segBatch.expand_as(depthPreds[n] ),
                depthImName, isGama = False )

    for n in range(0, len(renderedEnvs) ):
        envImName = []
        envImName.append(imgName.replace('input', 'renderedEnv_%d' % n ) )
        utils.writeImageToFile( renderedEnvs[n] * segBatch.expand_as(renderedEnvs[n] ),
                envImName, isGama = True )

    for n in range(0, len(globalIllu1s) ):
        imP1Name = []
        imP1Name.append(imgName.replace('input', 'renderedBounce1_%d' % n ) )
        utils.writeImageToFile( globalIllu1s[n] * segBatch.expand_as(globalIllu1s[n] ),
                imP1Name, isGama = True )

    for n in range(0, len(globalIllu2s) ):
        imP2Name = []
        imP2Name.append(imgName.replace('input', 'renderedBounce2_%d' % n ) )
        utils.writeImageToFile( globalIllu2s[n] * segBatch.expand_as(globalIllu2s[n] ),
                imP2Name, isGama = True )

    for n in range(0, len(globalIllu3s) ):
        imP3Name = []
        imP3Name.append(imgName.replace('input', 'renderedBounce3_%d' % n ) )
        utils.writeImageToFile( globalIllu3s[n] * segBatch.expand_as(globalIllu3s[n] ),
                imP3Name, isGama = True )

    for n in range(0, len(SHPreds) ):
        shCoefName = []
        shCoefName.append(imgName.replace('input', 'shCoef_%d' % n).replace('png', 'hdf5') )
        # utils.writeDataToFile(SHPreds[n], shCoefName )




