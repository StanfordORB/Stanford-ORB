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
# The locationi of training set
parser.add_argument('--dataRoot', default='../Data/train', help='path to images')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
parser.add_argument('--modelRootGlob', default='check_globalillumination', help='the path to store the network for global illumination prediction' )
parser.add_argument('--epochIdGlob', type=int, default=17, help='the training epoch for globalillum prediction' )
# The basic training setting
parser.add_argument('--nepoch', type=int, default=6, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpus used for training network')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for the depth component')
parser.add_argument('--globalIllu1', type=float, default=1, help='the weight of global illumination prediction 1')
parser.add_argument('--globalIllu2', type=float, default=0.01, help='the weight of global illumination prediction 2')
parser.add_argument('--globalIllu3', type=float, default=0.01, help='the weight of global illumination prediction 3')
parser.add_argument('--envWeight', type=float, default=0.01, help = 'the weight of training network for environmap prediction')
parser.add_argument('--imgEnvWeight', type=float, default=0.01, help='the weight of image rendered under environment map')
# The detail network setting
parser.add_argument('--cascadeLevel', type=int, default=2, help='cascade level')
# Refine input mode
parser.add_argument('--renderMode', type=int, default=2, help='Define the render type, \
        0 means render with direct lighting, 1 plus environment map, 2 plus global illumination 3 no environment map')
parser.add_argument('--refineInputMode', type=int, default=1, help='Define the type of input for refinement, \
        0 means no feedback, 1 means error feedback')
opt = parser.parse_args()
print(opt)

assert(opt.cascadeLevel == 1 or opt.cascadeLevel == 2)
opt.gpuId = opt.deviceIds[0]

if opt.experiment is None:
    opt.experiment = 'check_cascadeEnvGlob'
    opt.experiment += '_render{0}'.format(opt.renderMode)
    opt.experiment += '_refine{0}'.format(opt.refineInputMode)
    opt.experiment += '_cascade{0}'.format(opt.cascadeLevel )
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )

albeW, normW = opt.albedoWeight, opt.normalWeight
rougW, deptW = opt.roughWeight, opt.depthWeight
g1W, g2W, g3W = opt.globalIllu1, opt.globalIllu2, opt.globalIllu3
eW = opt.envWeight
imEW = opt.imgEnvWeight

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

albedoPredBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
normalPredBatch  = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
roughPredBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize) )
depthPredBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize) )
SHPredBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 9) )
imP2PredBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imP3PredBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )

# Refine Network
encoderRefs, albedoRefs = [], []
normalRefs, roughRefs = [], []
depthRefs, envRefs = [], []

encoderRefs.append( nn.DataParallel(models.refineEncoder(), device_ids = opt.deviceIds ) )
albedoRefs.append( nn.DataParallel(models.refineDecoder(mode=0), device_ids = opt.deviceIds ) )
normalRefs.append( nn.DataParallel(models.refineDecoder(mode=1), device_ids = opt.deviceIds) )
roughRefs.append( nn.DataParallel(models.refineDecoder(mode=2), device_ids = opt.deviceIds) )
depthRefs.append( nn.DataParallel(models.refineDecoder(mode=3), device_ids=opt.deviceIds) )
envRefs.append( nn.DataParallel(models.refineEnvDecoder(), device_ids = opt.deviceIds) )

renderLayer = models.renderingLayer(gpuId = opt.gpuId, isCuda = opt.cuda)

# Global illumination
globIllu1to2 = models.globalIllumination()
globIllu2to3 = models.globalIllumination()
#########################################

#########################################
# Load weight of network
globIllu1to2.load_state_dict(torch.load('{0}/globIllu1to2_{1}.pth'.format(opt.modelRootGlob, opt.epochIdGlob) ) )
globIllu2to3.load_state_dict(torch.load('{0}/globIllu2to3_{1}.pth'.format(opt.modelRootGlob, opt.epochIdGlob) ) )

for param in globIllu1to2.parameters():
    param.requires_grad = False
for param in globIllu2to3.parameters():
    param.requires_grad = False

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

    albedoPredBatch = albedoPredBatch.cuda(opt.gpuId)
    normalPredBatch  = normalPredBatch.cuda(opt.gpuId)
    roughPredBatch = roughPredBatch.cuda(opt.gpuId)
    depthPredBatch = depthPredBatch.cuda(opt.gpuId)
    SHPredBatch = SHPredBatch.cuda(opt.gpuId)
    imP2PredBatch = imP2PredBatch.cuda(opt.gpuId)
    imP3PredBatch = imP3PredBatch.cuda(opt.gpuId)

    encoderRefs[0] = encoderRefs[0].cuda(opt.gpuId)
    albedoRefs[0] = albedoRefs[0].cuda(opt.gpuId)
    normalRefs[0] = normalRefs[0].cuda(opt.gpuId)
    roughRefs[0] = roughRefs[0].cuda(opt.gpuId)
    depthRefs[0] = depthRefs[0].cuda(opt.gpuId)
    envRefs[0] = envRefs[0].cuda(opt.gpuId)
    globIllu1to2 = globIllu1to2.cuda(opt.gpuId)
    globIllu2to3 = globIllu2to3.cuda(opt.gpuId)
####################################


####################################
# Refine Optimizer
opEncoderRefs, opAlbedoRefs = [], []
opNormalRefs, opRoughRefs = [], []
opDepthRefs, opEnvRefs = [], []
opEncoderRefs.append(optim.Adam(encoderRefs[0].parameters(), lr=1e-4, betas=(0.5, 0.999) ) )
opAlbedoRefs.append(optim.Adam(albedoRefs[0].parameters(), lr=4e-4, betas=(0.5, 0.999) ) )
opNormalRefs.append(optim.Adam(normalRefs[0].parameters(), lr=4e-4, betas=(0.5, 0.999) ) )
opRoughRefs.append(optim.Adam(roughRefs[0].parameters(), lr=4e-4, betas=(0.5, 0.999) ) )
opDepthRefs.append(optim.Adam(depthRefs[0].parameters(), lr=4e-4, betas=(0.5, 0.999) ) )
opEnvRefs.append(optim.Adam(envRefs[0].parameters(), lr=4e-4, betas=(0.5, 0.999) ) )
#####################################


####################################
brdfDataset = dataLoader.BatchLoader(opt.dataRoot, imSize = opt.imageSize, cascade = opt.cascadeLevel-1 )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 8, shuffle=True )

j = 0
albedoErrsNpList = np.ones( [1, 2], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 2], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 2], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 2], dtype = np.float32 )

globalIllu1ErrsNpList= np.ones( [1, 2], dtype = np.float32 )
globalIllu2ErrsNpList= np.ones( [1, 2], dtype = np.float32 )
globalIllu3ErrsNpList= np.ones( [1, 2], dtype = np.float32 )
imgEnvErrsNpList = np.ones( [1, 2], dtype=np.float32 )

envErrsNpList = np.ones( [1, 2], dtype = np.float32)

for epoch in list(range(0, opt.nepoch)):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
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

        # Load the image from cpu to gpu
        im_cpu = (dataBatch['imP'] + dataBatch['imE'] + 1) * seg_cpu.expand_as(normal_cpu )
        imBatch.data.resize_(im_cpu.shape )
        imBatch.data.copy_(im_cpu )

        imBg_cpu = 0.5*(dataBatch['imP'] + 1) * seg_cpu.expand_as(normal_cpu ) \
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


        # Load the output of previous cascade
        albedoPred_cpu = dataBatch['albedoPred']
        albedoPredBatch.data.resize_(albedoPred_cpu.size() )
        albedoPredBatch.data.copy_(albedoPred_cpu)
        normalPred_cpu = dataBatch['normalPred']
        normalPredBatch.data.resize_(normalPred_cpu.size() )
        normalPredBatch.data.copy_(normalPred_cpu )
        roughPred_cpu = dataBatch['roughPred']
        roughPredBatch.data.resize_(roughPred_cpu.size() )
        roughPredBatch.data.copy_(roughPred_cpu)
        depthPred_cpu = dataBatch['depthPred']
        depthPredBatch.data.resize_(depthPred_cpu.size() )
        depthPredBatch.data.copy_(depthPred_cpu )
        imP2Pred_cpu = dataBatch['imP2Pred']
        imP2PredBatch.data.resize_(imP2Pred_cpu.size() )
        imP2PredBatch.data.copy_(imP2Pred_cpu )
        imP3Pred_cpu = dataBatch['imP3Pred']
        imP3PredBatch.data.resize_(imP3Pred_cpu.size() )
        imP3PredBatch.data.copy_(imP3Pred_cpu )
        SHPred_cpu = dataBatch['envPred']
        SHPredBatch.data.resize_(SHPred_cpu.size() )
        SHPredBatch.data.copy_(SHPred_cpu )

        # Clear the gradient in optimizer
        opEncoderRefs[0].zero_grad()
        opAlbedoRefs[0].zero_grad()
        opNormalRefs[0].zero_grad()
        opRoughRefs[0].zero_grad()
        opDepthRefs[0].zero_grad()
        opEnvRefs[0].zero_grad()

        ########################################################
        # Build the cascade network architecture #
        albedoPreds = [albedoPredBatch]
        normalPreds = [normalPredBatch]
        roughPreds = [roughPredBatch]
        depthPreds = [depthPredBatch]
        SHPreds = [SHPredBatch]
        globalIllu1s = []
        globalIllu2s = [0.5*(imP2PredBatch+1) * segBatch.expand_as(imP2PredBatch) ]
        globalIllu3s = [0.5*(imP3PredBatch+1) * segBatch.expand_as(imP3PredBatch) ]
        renderedEnvs = []


        # Refine the BRDF reconstruction
        albedoPred = (albedoPreds[0] * segBatch.expand_as(imBatch) )
        normalPred = (normalPreds[0] * segBatch.expand_as(imBatch) )
        roughPred = (roughPreds[0] * segBatch )
        depthPred = (depthPreds[0] * segBatch )
        SHPred = SHPreds[0].detach()

        globalIllu1 = renderLayer.forward( albedoPred, normalPred,
                roughPred, depthPred, segBatch )
        renderedEnv = renderLayer.forwardEnv( albedoPred, normalPred,
                roughPred, SHPred, segBatch )
        renderedEnvs.append(renderedEnv )
        globalIllu1s.append(globalIllu1 )

        if opt.renderMode == 0:
            renderedImg = globalIllu1
        elif opt.renderMode == 1:
            renderedImg = renderedEnv + globalIllu1
        elif opt.renderMode == 2:
            renderedImg = renderedEnv + globalIllu1 + \
                    globalIllu2s[0] + globalIllu3s[0]
        elif opt.renderMode == 3:
            renderedImg = globalIllu1 + \
                    globalIllu2s[0] + globalIllu3s[0]
        else:
            raise ValueError("The renderMode should be 0, 1, 2 or 3")

        if opt.refineInputMode == 0:
            inputRefine = torch.cat([albedoPred, normalPred, roughPred, depthPred, segBatch, \
                    imBatch, imBgBatch, 0*imBatch], dim=1)
        elif opt.refineInputMode == 1:
            error = (renderedImg - 0.5*(imBatch + 1) ) * segBatch.expand_as(imBatch)
            inputRefine = torch.cat( [albedoPred, normalPred, roughPred, depthPred, segBatch, \
                    imBatch, imBgBatch, error], dim=1)
        else:
            raise ValueError("The refine mode should be 0 or 1" )

        x1, x3 = encoderRefs[0](inputRefine.detach() )
        albedoPred = albedoRefs[0](x1, x3) * segBatch.expand_as(imBatch)
        normalPred = normalRefs[0](x1, x3) * segBatch.expand_as(imBatch)
        roughPred = roughRefs[0](x1, x3) * segBatch
        depthPred = depthRefs[0](x1, x3) * segBatch

        SHPred = envRefs[0](x3, SHPred)

        globalIllu1 = renderLayer.forward(albedoPred, normalPred,
                roughPred, depthPred, segBatch)
        globalIllu2 = globIllu1to2(torch.cat([ (2*globalIllu1 -1), \
                albedoPred, normalPred, roughPred, depthPred, segBatch], dim=1) )
        globalIllu3 =globIllu2to3(torch.cat([globalIllu2, \
                albedoPred, normalPred, roughPred, depthPred, segBatch], dim=1) )
        renderedEnv = renderLayer.forwardEnv(albedoPred, normalPred, roughPred, SHPred, segBatch)

        albedoPreds.append(albedoPred)
        normalPreds.append(normalPred)
        roughPreds.append(roughPred)
        depthPreds.append(depthPred)
        SHPreds.append(SHPred)
        globalIllu1s.append(torch.clamp(globalIllu1, 0, 1) )
        globalIllu2s.append(torch.clamp(0.5 *(globalIllu2+1), 0, 1) )
        globalIllu3s.append(torch.clamp(0.5 *(globalIllu3+1), 0, 1) )
        renderedEnvs.append(torch.clamp(renderedEnv, 0, 1) )

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
            globalIllu1Errs.append( torch.sum( (globalIllu1s[m] - 0.5*(imP1Batch+1) )
                    * (globalIllu1s[m] - 0.5*(imP1Batch+1) ) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
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

        # Back propagate the gradients
        totalErr = albeW * albedoErrs[-1] + normW * normalErrs[-1] + rougW *roughErrs[-1] \
                + deptW * depthErrs[-1] + g1W * globalIllu1Errs[-1] + eW * envErrs[-1] \
                + g2W * globalIllu2Errs[-1] + g3W * globalIllu3Errs[-1] + imEW * imgEnvErrs[-1]
        totalErr.backward()

        # Update the network parameter
        opEncoderRefs[0].step()
        opAlbedoRefs[0].step()
        opNormalRefs[0].step()
        opRoughRefs[0].step()
        opDepthRefs[0].step()
        opEnvRefs[0].step()

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

        utils.writeErrToFile('albedo', albedoErrs, trainingLog, epoch, j)
        utils.writeErrToFile('normal', normalErrs, trainingLog, epoch, j)
        utils.writeErrToFile('rough', roughErrs, trainingLog, epoch, j)
        utils.writeErrToFile('depth', depthErrs, trainingLog, epoch, j)
        utils.writeErrToFile('globalIllu1', globalIllu1Errs, trainingLog, epoch, j)
        utils.writeErrToFile('globalIllu2', globalIllu2Errs, trainingLog, epoch, j)
        utils.writeErrToFile('globalIllu3', globalIllu3Errs, trainingLog, epoch, j)
        utils.writeErrToFile('imgEnv', imgEnvErrs, trainingLog, epoch, j)
        utils.writeErrToFile('env', envErrs, trainingLog, epoch, j)

        albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
        normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
        roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0)
        depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)

        globalIllu1ErrsNpList = np.concatenate( [globalIllu1ErrsNpList, utils.turnErrorIntoNumpy(globalIllu1Errs)], axis=0)
        globalIllu2ErrsNpList = np.concatenate( [globalIllu2ErrsNpList, utils.turnErrorIntoNumpy(globalIllu2Errs)], axis=0)
        globalIllu3ErrsNpList = np.concatenate( [globalIllu3ErrsNpList, utils.turnErrorIntoNumpy(globalIllu3Errs)], axis=0)
        imgEnvErrsNpList = np.concatenate( [imgEnvErrsNpList, utils.turnErrorIntoNumpy(imgEnvErrs)], axis=0)

        envErrsNpList = np.concatenate( [envErrsNpList, utils.turnErrorIntoNumpy(envErrs)], axis=0)

        if j < 1000:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToScreen('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('globalIllu2Accu', np.mean(globalIllu2ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('globalIllu3Accu', np.mean(globalIllu3ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('imgEnvAccu', np.mean(imgEnvErrsNpList[1:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToScreen('envAccu:', np.mean(envErrsNpList[1:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)

            utils.writeNpErrToFile('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('globalIllu2Accu', np.mean(globalIllu2ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('globalIllu3Accu', np.mean(globalIllu3ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('imgEnvAccu', np.mean(imgEnvErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)

            utils.writeNpErrToFile('envAccu:', np.mean(envErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        else:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToScreen('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('globalIllu2Accu', np.mean(globalIllu2ErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('globalIllu3Accu', np.mean(globalIllu3ErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('imgEnvAccu', np.mean(imgEnvErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToScreen('envAccu', np.mean(envErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)

            utils.writeNpErrToFile('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('globalIllu2Accu', np.mean(globalIllu2ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('globalIllu3Accu', np.mean(globalIllu3ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('imgEnvAccu', np.mean(imgEnvErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)

            utils.writeNpErrToFile('envAccu:', np.mean(envErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)


        if j == 1 or j == 1000 or j% 5000 == 0:
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
                    SHBatch, nameBatch, 128, 256, 2, 8)

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
                        SHPreds[n], nameBatch, 128, 256, 2, 8)

    trainingLog.close()

    # Update the training rate
    if (epoch + 1) % 2 == 0:
        for param_group in opEncoderRefs[0].param_groups:
            param_group['lr'] /= 2
        for param_group in opAlbedoRefs[0].param_groups:
            param_group['lr'] /= 2
        for param_group in opNormalRefs[0].param_groups:
            param_group['lr'] /= 2
        for param_group in opRoughRefs[0].param_groups:
            param_group['lr'] /= 2
        for param_group in opDepthRefs[0].param_groups:
            param_group['lr'] /= 2
        for param_group in opEnvRefs[0].param_groups:
            param_group['lr'] /= 2

    # Save the error record
    np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
    np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )
    np.save('{0}/globalIllu1_{1}.npy'.format(opt.experiment, epoch), globalIllu1ErrsNpList )
    np.save('{0}/globalIllu2_{1}.npy'.format(opt.experiment, epoch), globalIllu2ErrsNpList )
    np.save('{0}/globalIllu3_{1}.npy'.format(opt.experiment, epoch), globalIllu3ErrsNpList )
    np.save('{0}/imgEnv_{1}.npy'.format(opt.experiment, epoch), imgEnvErrsNpList )
    np.save('{0}/envErrs_{1}.npy'.format(opt.experiment, epoch), envErrsNpList )

    # save the models
    torch.save(encoderRefs[0].module.state_dict(), '{0}/encoderRefs{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(albedoRefs[0].module.state_dict(), '{0}/albedoRefs{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(normalRefs[0].module.state_dict(), '{0}/normalRefs{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(roughRefs[0].module.state_dict(), '{0}/roughRefs{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(depthRefs[0].module.state_dict(), '{0}/depthRefs{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(envRefs[0].module.state_dict(), '{0}/envRefs{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
