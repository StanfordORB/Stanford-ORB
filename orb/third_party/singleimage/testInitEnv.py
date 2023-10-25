import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', default='../Data/test/', help='path to images')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
parser.add_argument('--modelRoot', default=None, help='the path to store samples and models')
parser.add_argument('--epochId', type=int, default=14, help='the number of epochs for testing')
parser.add_argument('--modelRootGlob', default='check_globalillumination', help='the path to store the network for global illumination prediction' )
parser.add_argument('--epochIdGlob', type=int, default=17, help='the training epoch for globalillum prediction' )
# The basic testing setting
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpuId used for training')
# The detail network setting
parser.add_argument('--cascadeLevel', type=int, default=0, help='cascade level')
parser.add_argument('--inputMode', type=int, default=0, help='whether to include back ground' )
opt = parser.parse_args()
print(opt)

if opt.modelRoot is None:
    opt.modelRoot = 'check_initEnvGlob'
    opt.modelRoot += '_cascade0'
    if opt.inputMode == 1:
        opt.modelRoot = opt.modelRoot + '_nobg'

if opt.experiment is None:
    opt.experiment = 'test_initEnvGlob'
    opt.experiment += '_cascade0'
    if opt.inputMode == 1:
        opt.experiment = opt.experiment + '_nobg'

os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )

opt.gpuId = opt.deviceIds[0]

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
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

renderLayer = models.renderingLayer(gpuId = opt.gpuId, isCuda = opt.cuda)

# Global illumination
globIllu1to2 = models.globalIllumination()
globIllu2to3 = models.globalIllumination()
#########################################

#########################################
# Load the weight to the network
encoderInit.load_state_dict(torch.load('{0}/encoderInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
albedoInit.load_state_dict(torch.load('{0}/albedoInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
normalInit.load_state_dict(torch.load('{0}/normalInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
roughInit.load_state_dict(torch.load('{0}/roughInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
depthInit.load_state_dict(torch.load('{0}/depthInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
envInit.load_state_dict(torch.load('{0}/envInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )

globIllu1to2.load_state_dict(torch.load('{0}/globIllu1to2_{1}.pth'.format(opt.modelRootGlob, opt.epochIdGlob) ) )
globIllu2to3.load_state_dict(torch.load('{0}/globIllu2to3_{1}.pth'.format(opt.modelRootGlob, opt.epochIdGlob) ) )

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

for param in globIllu1to2.parameters():
    param.requires_grad = False
for param in globIllu2to3.parameters():
    param.requires_grad = False

encoderInit = nn.DataParallel(encoderInit.eval(), device_ids=opt.deviceIds )
albedoInit = nn.DataParallel(albedoInit.eval(), device_ids=opt.deviceIds )
normalInit = nn.DataParallel(normalInit.eval(), device_ids=opt.deviceIds )
roughInit = nn.DataParallel(roughInit.eval(), device_ids=opt.deviceIds )
depthInit = nn.DataParallel(depthInit.eval(), device_ids=opt.deviceIds )
envInit = nn.DataParallel(envInit.eval(), device_ids = opt.deviceIds)

globIllu1to2 = nn.DataParallel(globIllu1to2.eval(), device_ids = opt.deviceIds)
globIllu2to3 = nn.DataParallel(globIllu2to3.eval(), device_ids = opt.deviceIds)
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
####################################



####################################
brdfDataset = dataLoader.BatchLoader(opt.dataRoot, imSize = opt.imageSize)
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 8, shuffle = True)

j = 0
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

globalIllu1ErrsNpList = np.ones( [1, 1], dtype = np.float32 )
globalIllu2ErrsNpList = np.ones( [1, 1], dtype=np.float32 )
globalIllu3ErrsNpList = np.ones( [1, 1], dtype=np.float32 )
imgEnvErrsNpList = np.ones( [1, 1], dtype=np.float32 )

envErrsNpList = np.ones( [1, 1], dtype = np.float32)

epoch = opt.epochId
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
for i, dataBatch in enumerate(brdfLoader):
    j += 1

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

    im_cpu = (dataBatch['imP'] + dataBatch['imE'] + 1) * seg_cpu.expand_as(normal_cpu)
    imBatch.data.resize_(im_cpu.shape )
    imBatch.data.copy_(im_cpu )

    imBg_cpu = 0.5*(dataBatch['imP'] + 1) * seg_cpu.expand_as(normal_cpu ) \
            + 0.5*(dataBatch['imEbg'] + 1)
    imBg_cpu = 2*imBg_cpu - 1
    if opt.inputMode == 1:
        imBg_cpu = 0 * imBg_cpu
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

    ########################################################
    # Build the cascade network architecture #
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
    inputInit = torch.cat([imBatch, imBgBatch, segBatch], dim=1)
    x1, x2, x3, x4, x5, x = encoderInit(inputInit)
    albedoPred = albedoInit(x1, x2, x3, x4, x5, x) * segBatch.expand_as(imBatch )
    normalPred = normalInit(x1, x2, x3, x4, x5, x) * segBatch.expand_as(imBatch )
    roughPred = roughInit(x1, x2, x3, x4, x5, x) * segBatch
    depthPred = depthInit(x1, x2, x3, x4, x5, x) * segBatch

    SHPred = envInit(x)

    globalIllu1 = renderLayer.forward(albedoPred, normalPred,
            roughPred, depthPred, segBatch)
    globalIllu2 = globIllu1to2(torch.cat([(2*globalIllu1 -1), \
            albedoPred, normalPred, roughPred, depthPred, segBatch], dim=1) )
    globalIllu3 =globIllu2to3(torch.cat([globalIllu2, \
            albedoPred, normalPred, roughPred, depthPred, segBatch], dim=1) )
    renderedEnv = renderLayer.forwardEnv(albedoPred, normalPred, roughPred, SHPred, segBatch)

    albedoPreds.append(albedoPred)
    normalPreds.append(normalPred)
    roughPreds.append(roughPred)
    depthPreds.append(depthPred)
    SHPreds.append(SHPred)
    globalIllu1s.append(globalIllu1)
    globalIllu2s.append(0.5 *(globalIllu2+1) )
    globalIllu3s.append(0.5 *(globalIllu3+1) )
    renderedEnvs.append(renderedEnv )

    ########################################################

    # Compute the error
    albedoErrs = []
    normalErrs = []
    roughErrs = []
    depthErrs = []
    globalIllu1Errs = []
    globalIllu2Errs = []
    globalIllu3Errs = []
    imgEnvErrs = []
    envErrs = []

    pixelNum = torch.sum(segBatch ).cpu().data.item()
    for m in range(0, len(albedoPreds ) ):
        albedoErrs.append( torch.sum( (albedoPreds[m] - albedoBatch)
                * (albedoPreds[m] - albedoBatch) * segBatch.expand_as(albedoBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(normalPreds ) ):
        normalErrs.append( torch.sum( (normalPreds[m] - normalBatch)
                * (normalPreds[m] - normalBatch) * segBatch.expand_as(normalBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(roughPreds ) ):
        roughErrs.append( torch.sum( (roughPreds[m] - roughBatch)
                * (roughPreds[m] - roughBatch) * segBatch ) / pixelNum )
    for m in range(0, len(depthPreds ) ):
        depthErrs.append( torch.sum( (depthPreds[m] - depthBatch)
                * (depthPreds[m] - depthBatch) * segBatch ) / pixelNum )
    for m in range(0, len(globalIllu1s ) ):
        globalIllu1Errs.append( torch.sum( (globalIllu1s[m] - 0.5*(imP1Batch + 1) )
                * (globalIllu1s[m] - 0.5*(imP1Batch + 1) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(globalIllu2s ) ):
        globalIllu2Errs.append( torch.sum( (globalIllu2s[m] - 0.5*(imP2Batch+1) )
                * (globalIllu2s[m] - 0.5*(imP2Batch+1) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(globalIllu3s ) ):
        globalIllu3Errs.append( torch.sum( (globalIllu3s[m] - 0.5*(imP3Batch+1) )
                * (globalIllu3s[m] - 0.5*(imP3Batch+1) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(renderedEnvs) ):
        imgEnvErrs.append( torch.sum( (renderedEnvs[m] - 0.5*(imEBatch+1) )
            * (renderedEnvs[m] - 0.5*(imEBatch+1) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
    for m in range(0, len(SHPreds ) ):
        envErrs.append( torch.mean( (SHPreds[m] - SHBatch)* (SHPreds[m] - SHBatch) ) )

    # Output testing error
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

    albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
    normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
    roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0)
    depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)

    globalIllu1ErrsNpList = np.concatenate( [globalIllu1ErrsNpList, utils.turnErrorIntoNumpy(globalIllu1Errs)], axis=0)
    globalIllu2ErrsNpList = np.concatenate( [globalIllu2ErrsNpList, utils.turnErrorIntoNumpy(globalIllu2Errs)], axis=0)
    globalIllu3ErrsNpList = np.concatenate( [globalIllu3ErrsNpList, utils.turnErrorIntoNumpy(globalIllu3Errs)], axis=0)
    imgEnvErrsNpList = np.concatenate( [imgEnvErrsNpList, utils.turnErrorIntoNumpy(imgEnvErrs)], axis=0)

    envErrsNpList = np.concatenate( [envErrsNpList, utils.turnErrorIntoNumpy(envErrs)], axis=0)

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
        # Save the ground truth and the input
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
        vutils.save_image( ( (0.5*(imEBatch + 1) )**(1.0/2.2) * segBatch.expand_as(imEBatch) ).data,
                '{0}/{1}_imE.png'.format(opt.experiment, j) )
        vutils.save_image( ( (0.5*(imP1Batch + 1) )**(1.0/2.2) * segBatch.expand_as(imP1Batch) ).data,
                '{0}/{1}_imP1.png'.format(opt.experiment, j) )
        vutils.save_image( ( (0.5*(imP2Batch + 1) )**(1.0/2.2) * segBatch.expand_as(imP2Batch) ).data,
                '{0}/{1}_imP2.png'.format(opt.experiment, j) )
        vutils.save_image( ( (0.5*(imP3Batch + 1) )**(1.0/2.2) * segBatch.expand_as(imP3Batch) ).data,
                '{0}/{1}_imP3.png'.format(opt.experiment, j) )

        utils.visualizeSH('{0}/{1}_gtSH.png'.format(opt.experiment, j),
                SHBatch, nameBatch, 128, 256, 8, 8)

        # Save the predicted results
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
            deepthOut = (depthPreds[n] - 0.25) /0.8
            vutils.save_image( ( depthOut * segBatch.expand_as(depthPreds[n]) ).data,
                    '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(renderedEnvs) ):
            vutils.save_image( ( ( renderedEnvs[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imEPred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(globalIllu1s) ):
            vutils.save_image( ( ( globalIllu1s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imP1Pred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(globalIllu2s) ):
            vutils.save_image( ( ( globalIllu2s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imP2Pred_{2}.png'.format(opt.experiment, j, n) )
        for n in range(0, len(globalIllu3s) ):
            vutils.save_image( ( ( globalIllu3s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imP3Pred_{2}.png'.format(opt.experiment, j, n) )
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
np.save('{0}/imgEnv_{1}.npy'.format(opt.experiment, epoch), imgEnvErrsNpList )
np.save('{0}/envError_{1}.npy'.format(opt.experiment, epoch), envErrsNpList )
