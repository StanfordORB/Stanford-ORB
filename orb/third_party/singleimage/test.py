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
parser.add_argument('--dataRoot', default='../Data/test/', help='path to images')
parser.add_argument('--modelRootInit', default = None, help = 'the directory where the initialization trained model is save')
parser.add_argument('--modelRootsRefine', nargs='+', default=[None, None], help='the directory where the refine models are saved')
parser.add_argument('--modelRootGlob', default = None, help = 'the directory where the global illumination model is saved')
parser.add_argument('--epochIdInit', type=int, default = 14, help = 'the training epoch of the initial network')
parser.add_argument('--epochIdsRefine', nargs = '+', type=int, default = [7, 5], help='the training epoch of the refine network')
parser.add_argument('--epochIdGlob', type=int, default=17, help='the traing epoch of the global illuminationn prediction network')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic testing setting
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpus used for testing network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
# The testing weight
parser.add_argument('--cascadeLevel', type=int, default=2, help='cascade level')
# Refine input mode
parser.add_argument('--renderMode', type=int, default=2, help='Define the render type, \
        0 means render with direct lighting, 1 plus environment map, 2 plus global illumination')
parser.add_argument('--refineInputMode', type=int, default=1, help='Define the type of input for refinement, \
        0 means no feedback, 1 means error feedback')
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

if opt.modelRootInit is None:
    opt.modelRootInit = 'check_initEnvGlob_cascade0'

if len(opt.modelRootsRefine) != opt.cascadeLevel or opt.modelRootsRefine[0] is None:
    opt.modelRootsRefine = []
    for n in range(1, opt.cascadeLevel):
        root = 'check_cascadeEnvGlob'
        root += '_render{0}'.format(2)
        root += '_refine{0}'.format(1)
        root += '_cascade{0}'.format(n)
        opt.modelRootsRefine.append(root)
    for n in range(opt.cascadeLevel, opt.cascadeLevel+1):
        root = 'check_cascadeEnvGlob'
        root += '_render{0}'.format(opt.renderMode)
        root += '_refine{0}'.format(opt.refineInputMode)
        root += '_cascade{0}'.format(n)
        opt.modelRootsRefine.append(root)

if opt.modelRootGlob is None:
    opt.modelRootGlob = 'check_globalillumination'

if opt.experiment is None:
    opt.experiment = 'test_render{0}_refine{1}_cascade{2}'.format(
            opt.renderMode,
            opt.refineInputMode,
            opt.cascadeLevel
            )
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )


opt.seed = 0
torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################
# initalize tensors
albedoBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
normalBatch  = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
roughBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize) )
segBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
depthBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize) )
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imBgBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
SHBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 9) )

imP1Batch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imP2Batch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imP3Batch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imEBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )

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

encoderInit = nn.DataParallel(encoderInit, device_ids = opt.deviceIds )
albedoInit = nn.DataParallel(albedoInit, device_ids = opt.deviceIds )
normalInit = nn.DataParallel(normalInit, device_ids = opt.deviceIds )
roughInit = nn.DataParallel(roughInit, device_ids = opt.deviceIds )
depthInit = nn.DataParallel(depthInit, device_ids = opt.deviceIds )
envInit = nn.DataParallel(envInit, device_ids = opt.deviceIds )
for n in range(0, opt.cascadeLevel ):
    encoderRefs[n] = nn.DataParallel(encoderRefs[n] )
    albedoRefs[n] = nn.DataParallel(albedoRefs[n] )
    normalRefs[n] = nn.DataParallel(normalRefs[n] )
    roughRefs[n] = nn.DataParallel(roughRefs[n] )
    depthRefs[n] = nn.DataParallel(depthRefs[n] )


globIllu1to2 = nn.DataParallel(globIllu1to2, device_ids = opt.deviceIds )
globIllu2to3 = nn.DataParallel(globIllu2to3, device_ids = opt.deviceIds )
#########################################

##############  ######################
# Send things into GPU
if opt.cuda:
    albedoBatch = albedoBatch.cuda(opt.gpuId)
    normalBatch = normalBatch.cuda(opt.gpuId)
    roughBatch = roughBatch.cuda(opt.gpuId)
    depthBatch = depthBatch.cuda(opt.gpuId)
    segBatch = segBatch.cuda(opt.gpuId)
    imBatch = imBatch.cuda(opt.gpuId)
    imBgBatch = imBgBatch.cuda(opt.gpuId)
    SHBatch = SHBatch.cuda(opt.gpuId)

    imP1Batch = imP1Batch.cuda(opt.gpuId)
    imP2Batch = imP2Batch.cuda(opt.gpuId)
    imP3Batch = imP3Batch.cuda(opt.gpuId)
    imEBatch = imEBatch.cuda(opt.gpuId)

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
brdfDataset = dataLoader.BatchLoader(opt.dataRoot, imSize = opt.imageSize )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 8, shuffle = False )

albedoErrsNpList = np.ones( [1, opt.cascadeLevel + 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, opt.cascadeLevel + 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, opt.cascadeLevel + 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, opt.cascadeLevel + 1], dtype = np.float32 )

globalIllu1ErrsNpList= np.ones( [1, opt.cascadeLevel + 1], dtype = np.float32 )
globalIllu2ErrsNpList= np.ones( [1, opt.cascadeLevel + 1], dtype = np.float32 )
globalIllu3ErrsNpList= np.ones( [1, opt.cascadeLevel + 1], dtype = np.float32 )
imgEnvErrsNpList = np.ones( [1, opt.cascadeLevel + 1], dtype=np.float32 )

envErrsNpList = np.ones( [1, opt.cascadeLevel + 1], dtype = np.float32)

epoch = opt.epochIdsRefine[1]
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
j = 0
for i, dataBatch in enumerate(brdfLoader):
    j = j+1
    # Load data from cpu to gpu
    albedo_cpu = dataBatch['albedo']
    albedoBatch.data.resize_(albedo_cpu.size() )
    albedoBatch.data.copy_(albedo_cpu )
    normal_cpu = dataBatch['normal']
    normalBatch.data.resize_(normal_cpu.size() )
    normalBatch.data.copy_(normal_cpu )
    rough_cpu = dataBatch['rough']
    roughBatch.data.resize_(rough_cpu.size() )
    roughBatch.data.copy_(rough_cpu )
    seg_cpu = dataBatch['seg']
    segBatch.data.resize_(seg_cpu.size() )
    segBatch.data.copy_(seg_cpu )
    depth_cpu = dataBatch['depth']
    depthBatch.data.resize_(depth_cpu.size() )
    depthBatch.data.copy_(depth_cpu )
    # Load the image from cpu to gpu
    im_cpu = (dataBatch['imP'] + dataBatch['imE'] + 1) * seg_cpu.expand_as(albedoBatch )
    imBatch.data.resize_(im_cpu.shape )
    imBatch.data.copy_(im_cpu )
    imBg_cpu = 0.5*(dataBatch['imP'] + 1) * seg_cpu.expand_as(albedoBatch ) \
            + 0.5*(dataBatch['imEbg'] + 1)
    imBg_cpu = 2*imBg_cpu - 1
    imBgBatch.data.resize_(imBg_cpu.size() )
    imBgBatch.data.copy_(imBg_cpu )

    # Load the spherical harmonics
    SH_cpu = dataBatch['SH']
    SHBatch.data.resize_(SH_cpu.size() )
    SHBatch.data.copy_(SH_cpu )
    nameBatch = dataBatch['name']

    # Load the image with multiple bounce
    imP1_cpu = dataBatch['imP1']
    imP1Batch.data.resize_(imP1_cpu.size() )
    imP1Batch.data.copy_(imP1_cpu )
    imP2_cpu = dataBatch['imP2']
    imP2Batch.data.resize_(imP2_cpu.size() )
    imP2Batch.data.copy_(imP2_cpu )
    imP3_cpu = dataBatch['imP3']
    imP3Batch.data.resize_(imP3_cpu.size() )
    imP3Batch.data.copy_(imP3_cpu)
    imE_cpu = dataBatch['imE']
    imEBatch.data.resize_(imE_cpu.size() )
    imEBatch.data.copy_(imE_cpu )

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

        if n == opt.cascadeLevel - 1:
            if opt.renderMode == 0:
                renderedImg = globalIllu1
            elif opt.renderMode == 1:
                renderedImg = renderedEnv + globalIllu1
            elif opt.renderMode == 2:
                renderedImg = renderedEnv + globalIllu1 + \
                        globalIllu2 + globalIllu3
            elif opt.renderMode == 3:
                renderedImg = globalIllu1 + globalIllu2 + globalIllu3
            else:
                raise ValueError("The renderMode should be 0, 1, 2 or 3")

            if opt.refineInputMode == 0:
                inputRefine = torch.cat([albedoPred, normalPred, roughPred, depthPred, segBatch, \
                        imBatch, imBgBatch], dim=1)
            elif opt.refineInputMode == 1:
                error = (renderedImg - 0.5*(imBatch + 1) ) * segBatch.expand_as(imBatch)
                inputRefine = torch.cat( [albedoPred, normalPred, roughPred, depthPred, segBatch, \
                        imBatch, imBgBatch, error], dim=1)
            else:
                raise ValueError("The refine mode should be 0, 1" )
        else:
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
        renderedEnvs.append(renderedEnv)

    albedoErrs = []
    normalErrs = []
    roughErrs = []
    depthErrs = []
    globalIllu1Errs = []
    globalIllu2Errs = []
    globalIllu3Errs = []
    imgEnvErrs = []
    envErrs = []

    pixelNum = (torch.sum(segBatch ).cpu().data).item()
    for m in range(0, len(albedoPreds) ):
        albedoErrs.append( torch.sum( (albedoPreds[m] - albedoBatch)
                * (albedoPreds[m] - albedoBatch) * segBatch.expand_as(albedoBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(normalPreds) ):
        normalErrs.append( torch.sum( (normalPreds[m] - normalBatch)
                * (normalPreds[m] - normalBatch) * segBatch.expand_as(normalBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(roughPreds) ):
        roughErrs.append( torch.sum( (roughPreds[m] - roughBatch)
                * (roughPreds[m] - roughBatch) * segBatch ) / pixelNum )
    for m in range(0, len(depthPreds) ):
        depthErrs.append( torch.sum( (depthPreds[m] - depthBatch)
                * (depthPreds[m] - depthBatch) * segBatch ) / pixelNum )
    for m in range(0, len(globalIllu1s) ):
        globalIllu1Errs.append( torch.sum( (globalIllu1s[m] - 0.5*(imP1Batch+1 ) )
                * (globalIllu1s[m] - 0.5 * (imP1Batch+1 ) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(globalIllu2s) ):
        globalIllu2Errs.append( torch.sum( (globalIllu2s[m] - 0.5*(imP2Batch+1) )
                * (globalIllu2s[m] - 0.5*(imP2Batch+1) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(globalIllu3s) ):
        globalIllu3Errs.append( torch.sum( (globalIllu3s[m] - 0.5*(imP3Batch+1) )
                * (globalIllu3s[m] - 0.5*(imP3Batch+1) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(renderedEnvs) ):
        imgEnvErrs.append( torch.sum( (renderedEnvs[m] - 0.5*(imEBatch+1) )
                * (renderedEnvs[m] - 0.5*(imEBatch+1) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(SHPreds) ):
        envErrs.append( torch.mean( (SHPreds[m] - SHBatch)*(SHPreds[m] - SHBatch) ) )

    albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
    normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
    roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0)
    depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)

    globalIllu1ErrsNpList = np.concatenate( [globalIllu1ErrsNpList, utils.turnErrorIntoNumpy(globalIllu1Errs)], axis=0)
    globalIllu2ErrsNpList = np.concatenate( [globalIllu2ErrsNpList, utils.turnErrorIntoNumpy(globalIllu2Errs)], axis=0)
    globalIllu3ErrsNpList = np.concatenate( [globalIllu3ErrsNpList, utils.turnErrorIntoNumpy(globalIllu3Errs)], axis=0)
    imgEnvErrsNpList = np.concatenate( [imgEnvErrsNpList, utils.turnErrorIntoNumpy(imgEnvErrs)], axis=0)

    envErrsNpList = np.concatenate( [envErrsNpList, utils.turnErrorIntoNumpy(envErrs)], axis=0)

    # Output training error
    utils.writeErrToScreen('albedo', albedoErrs, epoch, j)
    utils.writeErrToScreen('normal', normalErrs, epoch, j)
    utils.writeErrToScreen('rough', roughErrs, epoch, j)
    utils.writeErrToScreen('depth', depthErrs, epoch, j)
    utils.writeErrToScreen('globalIllu1', globalIllu1Errs, epoch, j)
    utils.writeErrToScreen('globalIllu2', globalIllu2Errs, epoch, j)
    utils.writeErrToScreen('globalIllu3', globalIllu3Errs, epoch, j)
    utils.writeErrToScreen('imgEnv', imgEnvErrs, epoch, j)
    utils.writeErrToScreen('env', envErrs, epoch, j)

    utils.writeErrToFile('albedo', albedoErrs, testingLog, epoch, j)
    utils.writeErrToFile('normal', normalErrs, testingLog, epoch, j)
    utils.writeErrToFile('rough', roughErrs, testingLog, epoch, j)
    utils.writeErrToFile('depth', depthErrs, testingLog, epoch, j)
    utils.writeErrToFile('globalIllu1', globalIllu1Errs, testingLog, epoch, j)
    utils.writeErrToFile('globalIllu2', globalIllu2Errs, testingLog, epoch, j)
    utils.writeErrToFile('globalIllu3', globalIllu3Errs, testingLog, epoch, j)
    utils.writeErrToFile('imgEnv', imgEnvErrs, testingLog, epoch, j)
    utils.writeErrToFile('env', envErrs, testingLog, epoch, j)

    utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j)

    utils.writeNpErrToScreen('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('globalIllu2Accu', np.mean(globalIllu2ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('globalIllu3Accu', np.mean(globalIllu3ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('imgEnvAccu', np.mean(imgEnvErrsNpList[1:j+1, :], axis=0), epoch, j)

    utils.writeNpErrToScreen('envAccu:', np.mean(envErrsNpList[1:j+1, :], axis=0), epoch, j)

    utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)

    utils.writeNpErrToFile('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('globalIllu2Accu', np.mean(globalIllu2ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('globalIllu3Accu', np.mean(globalIllu3ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('imgEnvAccu', np.mean(imgEnvErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)

    utils.writeNpErrToFile('envAccu:', np.mean(envErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)

    if j == 1 or j % 2000 == 0:
        vutils.save_image( (0.5*(albedoBatch + 1)*segBatch.expand_as(albedoBatch) ).data,
                '{0}/{1}_albedoGt.png'.format(opt.experiment, j) )
        vutils.save_image( (0.5*(normalBatch + 1)*segBatch.expand_as(normalBatch) ).data,
                '{0}/{1}_normalGt.png'.format(opt.experiment, j) )
        vutils.save_image( (0.5*(roughBatch + 1)*segBatch.expand_as(roughBatch) ).data,
                '{0}/{1}_roughGt.png'.format(opt.experiment, j) )
        depthOut = 1 / torch.clamp(depthBatch, 1e-6, 10) * segBatch.expand_as(depthBatch)
        depthOut = (depthOut - 0.25) /0.8
        vutils.save_image( ( depthOut*segBatch.expand_as(depthBatch) ).data,
                '{0}/{1}_depthGt.png'.format(opt.experiment, j) )
        vutils.save_image( ( (0.5*(imBatch + 1)*segBatch.expand_as(imBatch))**(1.0/2.2) ).data,
                '{0}/{1}_im.png'.format(opt.experiment, j) )
        vutils.save_image( ( (0.5*(imBgBatch + 1) )**(1.0/2.2) ).data,
                '{0}/{1}_imBg.png'.format(opt.experiment, j) )

        vutils.save_image( ( (0.5*(imP1Batch + 1) )**(1.0/2.2) * segBatch.expand_as(imP1Batch) ).data,
                '{0}/{1}_imP1.png'.format(opt.experiment, j) )
        vutils.save_image( ( (0.5*(imP2Batch + 1) )**(1.0/2.2) * segBatch.expand_as(imP2Batch) ).data,
                '{0}/{1}_imP2.png'.format(opt.experiment, j) )
        vutils.save_image( ( (0.5*(imP3Batch + 1) )**(1.0/2.2) * segBatch.expand_as(imP3Batch) ).data,
                '{0}/{1}_imP3.png'.format(opt.experiment, j) )

        utils.visualizeSH('{0}/{1}_gtSH.png'.format(opt.experiment, j),
                SHBatch, nameBatch, 128, 256, 8, 8)

        for n in range(0, len(albedoPreds) ):
            vutils.save_image( ( 0.5*(albedoPreds[n] + 1)*segBatch.expand_as(albedoPreds[n]) ).data,
                    '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(normalPreds) ):
            vutils.save_image( ( 0.5*(normalPreds[n] + 1)*segBatch.expand_as(normalPreds[n]) ).data,
                    '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(roughPreds) ):
            vutils.save_image( ( 0.5*(roughPreds[n] + 1)*segBatch.expand_as(roughPreds[n]) ).data,
                    '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(depthPreds) ):
            depthOut = 1 / torch.clamp(depthPreds[n], 1e-6, 10) * segBatch.expand_as(depthPreds[n])
            depthOut = (depthOut - 0.25) /0.8
            vutils.save_image( ( depthOut * segBatch.expand_as(depthPreds[n]) ).data,
                    '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(globalIllu1s) ):
            vutils.save_image( ( ( globalIllu1s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imP1Pred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(globalIllu2s) ):
            vutils.save_image( ( ( globalIllu2s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imP2Pred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(globalIllu3s) ):
            vutils.save_image( ( ( globalIllu3s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imP3Pred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(renderedEnvs) ):
            vutils.save_image( ( ( renderedEnvs[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imEPred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(SHPreds) ):
            utils.visualizeSH('{0}/{1}_predSH_{2}.png'.format(opt.experiment, j, n),
                    SHPreds[n], nameBatch, 128, 256, 8, 8)

testingLog.close()

# Save the error record
np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )
np.save('{0}/globalIllu1_{1}.npy'.format(opt.experiment, epoch), globalIllu1ErrsNpList )
np.save('{0}/globalIllu2_{1}.npy'.format(opt.experiment, epoch), globalIllu2ErrsNpList )
np.save('{0}/globalIllu3_{1}.npy'.format(opt.experiment, epoch), globalIllu3ErrsNpList )
np.save('{0}/envErrs_{1}.npy'.format(opt.experiment, epoch), envErrsNpList )

