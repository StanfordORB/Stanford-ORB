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
parser.add_argument('--dataRoot', default='../Data/test', help='path to real image distorted by water')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
parser.add_argument('--modelRoot', default=None, help='the path to store samples and models')
# The basic testing setting
parser.add_argument('--epochId', type=int, default=14, help='the number of epochs for testing')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpuId used for training')
# The detail network setting
parser.add_argument('--inputMode', type=int, default=0, help='Determine what kind of image should be taken as input')
parser.add_argument('--cascadeLevel', type=int, default=0, help='cascade level')
opt = parser.parse_args()
print(opt)

if opt.modelRoot is None:
    opt.modelRoot = 'check_init'
    if opt.inputMode == 0:
        opt.modelRoot += '_pointDirect'
    elif opt.inputMode == 1:
        opt.modelRoot += '_pointInDirect'
    opt.modelRoot += '_cascade0'

if opt.experiment is None:
    opt.experiment = 'test_init'
    if opt.inputMode == 0:
        opt.experiment += '_pointDirect'
    elif opt.inputMode == 1:
        opt.experiment += '_pointInDirect'
    opt.experiment += '_cascade0'

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

# Initial Network
encoderInit = models.encoderInitial_point()
albedoInit = models.decoderInitial(mode=0)
normalInit = models.decoderInitial(mode=1)
roughInit = models.decoderInitial(mode=2)
depthInit = models.decoderInitial(mode=3)

renderLayer = models.renderingLayer(gpuId = opt.gpuId, isCuda = opt.cuda)

#########################################
# Load the weight to the network
encoderInit.load_state_dict(torch.load('{0}/encoderInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
albedoInit.load_state_dict(torch.load('{0}/albedoInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
normalInit.load_state_dict(torch.load('{0}/normalInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
roughInit.load_state_dict(torch.load('{0}/roughInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )
depthInit.load_state_dict(torch.load('{0}/depthInit_{1}.pth'.format(opt.modelRoot, opt.epochId) ) )

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

encoderInit = nn.DataParallel(encoderInit.eval(), device_ids=opt.deviceIds )
albedoInit = nn.DataParallel(albedoInit.eval(), device_ids=opt.deviceIds )
normalInit = nn.DataParallel(normalInit.eval(), device_ids=opt.deviceIds )
roughInit = nn.DataParallel(roughInit.eval(), device_ids=opt.deviceIds )
depthInit = nn.DataParallel(depthInit.eval(), device_ids=opt.deviceIds )
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

    encoderInit = encoderInit.cuda(opt.gpuId)
    albedoInit = albedoInit.cuda(opt.gpuId)
    normalInit = normalInit.cuda(opt.gpuId)
    roughInit = roughInit.cuda(opt.gpuId)
    depthInit = depthInit.cuda(opt.gpuId)

####################################



####################################
brdfDataset = dataLoader.BatchLoader(opt.dataRoot, imSize = opt.imageSize)
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 8, shuffle = False)

j = 0
albedoErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
globalIllu1ErrsNpList= np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )

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

    im_cpu = dataBatch['imP']
    imBatch.data.resize_(im_cpu.size() )
    imBatch.data.copy_(im_cpu )

    ########################################################
    # Build the cascade network architecture #
    albedoPreds = []
    normalPreds = []
    roughPreds = []
    depthPreds = []
    globalIllu1s = []
    renderedImgs = []
    errors = []

    globalIllu1Gt = renderLayer.forward(albedoBatch, normalBatch,
            roughBatch, depthBatch, segBatch)

    # Initial Prediction
    inputInit = torch.cat([imBatch, segBatch], dim=1)
    x1, x2, x3, x4, x5, x = encoderInit(inputInit)
    x1, x2, x3, x4, x5, x = x1.detach(), x2.detach(), x3.detach(), \
            x4.detach(), x5.detach(), x.detach()
    albedoPred = albedoInit(x1, x2, x3, x4, x5, x).detach()
    normalPred = normalInit(x1, x2, x3, x4, x5, x).detach()
    roughPred = roughInit(x1, x2, x3, x4, x5, x).detach()
    depthPred = depthInit(x1, x2, x3, x4, x5, x).detach()

    globalIllu1 = renderLayer.forward(albedoPred, normalPred,
            roughPred, depthPred, segBatch)*segBatch.expand_as(albedoPred )

    albedoPreds.append(albedoPred)
    normalPreds.append(normalPred)
    roughPreds.append(roughPred)
    depthPreds.append(depthPred)
    globalIllu1s.append(globalIllu1)

    # Compute the error
    albedoErrs = []
    normalErrs = []
    roughErrs = []
    depthErrs = []
    globalIllu1Errs = []

    pixelNum = torch.sum(segBatch ).cpu().data.item()
    m = 0
    albedoErrs.append( torch.sum( (albedoPreds[m] - albedoBatch)
            * (albedoPreds[m] - albedoBatch) * segBatch.expand_as(albedoBatch) ) / pixelNum / 3.0 )
    normalErrs.append( torch.sum( (normalPreds[m] - normalBatch)
            * (normalPreds[m] - normalBatch) * segBatch.expand_as(normalBatch) ) / pixelNum / 3.0 )
    roughErrs.append( torch.sum( (roughPreds[m] - roughBatch)
            * (roughPreds[m] - roughBatch) * segBatch ) / pixelNum )
    depthErrs.append( torch.sum( (depthPreds[m] - depthBatch)
            * (depthPreds[m] - depthBatch) * segBatch ) / pixelNum )
    globalIllu1Errs.append( torch.sum( (globalIllu1s[m] - globalIllu1Gt )
            * (globalIllu1s[m] - globalIllu1Gt) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )

    # Output testing error
    utils.writeErrToScreen('albedo', albedoErrs, epoch, j)
    utils.writeErrToScreen('normal', normalErrs, epoch, j)
    utils.writeErrToScreen('rough', roughErrs, epoch, j)
    utils.writeErrToScreen('depth', depthErrs, epoch, j)
    utils.writeErrToScreen('globalIllu1', globalIllu1Errs, epoch, j)
    utils.writeErrToFile('albedo', albedoErrs, testingLog, epoch, j)
    utils.writeErrToFile('normal', normalErrs, testingLog, epoch, j)
    utils.writeErrToFile('rough', roughErrs, testingLog, epoch, j)
    utils.writeErrToFile('depth', depthErrs, testingLog, epoch, j)
    utils.writeErrToFile('globalIllu1', globalIllu1Errs, testingLog, epoch, j)
    albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
    normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
    roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0)
    depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)
    globalIllu1ErrsNpList = np.concatenate( [globalIllu1ErrsNpList, utils.turnErrorIntoNumpy(globalIllu1Errs)], axis=0)

    utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[1:j+1, :], axis=0), epoch, j)

    utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)


    if j == 1 or j% 2000 == 0:
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

        # Save the predicted results
        for n in range(0, opt.cascadeLevel + 1):
            vutils.save_image( ( 0.5*(albedoPreds[n] + 1)*segBatch.expand_as(albedoPreds[n]) ).data,
                    '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, n) )
            vutils.save_image( ( 0.5*(normalPreds[n] + 1)*segBatch.expand_as(normalPreds[n]) ).data,
                    '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, n) )
            vutils.save_image( ( 0.5*(roughPreds[n] + 1)*segBatch.expand_as(roughPreds[n]) ).data,
                    '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, n) )
            depthOut = 1 / torch.clamp(depthPreds[n], 1e-6, 10) * segBatch.expand_as(depthPreds[n])
            depthOut = (depthOut - 0.25) /0.8
            vutils.save_image( ( depthOut * segBatch.expand_as(depthPreds[n]) ).data,
                    '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )
            vutils.save_image( ( ( globalIllu1s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                    '{0}/{1}_imPred_{2}.png'.format(opt.experiment, j, n) )

testingLog.close()


# Save the error record
np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )
np.save('{0}/globalIllu1_{1}.npy'.format(opt.experiment, epoch), globalIllu1ErrsNpList )

