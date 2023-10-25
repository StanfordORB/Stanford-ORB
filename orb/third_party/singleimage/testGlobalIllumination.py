import torch
import numpy as np
from torch.autograd import Variable
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils
import dataLoader
from torch.utils.data import DataLoader
import os.path as osp

parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', default='/home/zhl/SiggraphAsia18/Data/test/', help='path to images')
parser.add_argument('--modelRoot', default=None, help='the path to store testing models')
parser.add_argument('--experiment', default=None, help='the path to store testing data')
# The basic testing setting
parser.add_argument('--epochId', type=int, default=17, help='the number of epochs for testing')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')
# The testing weight
parser.add_argument('--globalIllu2',  type=float, default=1, help='the weight of global illumination prediction 2')
parser.add_argument('--globalIllu3', type=float, default=1, help='the weight of global illumination prediction 3')
# The detail network setting
parser.add_argument('--cascadeLevel', type=int, default=0, help='how much level of cascades should we use')
opt = parser.parse_args()
print(opt)

if opt.modelRoot is None:
    opt.modelRoot = 'check_globalillumination'
if opt.experiment is None:
    opt.experiment = 'test_globalillumination'
os.system('mkdir {0}'.format(opt.experiment) )

os.system('cp *.py %s' % opt.experiment )

g2W, g3W = opt.globalIllu2, opt.globalIllu3
opt.gpuId = 0

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

imP1Batch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imP2Batch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imP3Batch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imPBatch  = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )

# Global illumination
globIllu1to2 = models.globalIllumination()
globIllu2to3 = models.globalIllumination()
#########################################

#########################################
# Load weight of network
globIllu1to2.load_state_dict(torch.load('{0}/globIllu1to2_{1}.pth'.format(opt.modelRoot, opt.epochId ) ) )
globIllu2to3.load_state_dict(torch.load('{0}/globIllu2to3_{1}.pth'.format(opt.modelRoot, opt.epochId ) ) )
globIllu1to2 = globIllu1to2.eval()
globIllu2to3 = globIllu2to3.eval()
for param in globIllu1to2.parameters():
    param.requires_grad = False
for param in globIllu2to3.parameters():
    param.requires_grad = False
#########################################


##############  ######################
# Send things into GPU
if opt.cuda:
    albedoBatch = albedoBatch.cuda(opt.gpuId)
    normalBatch = normalBatch.cuda(opt.gpuId)
    roughBatch = roughBatch.cuda(opt.gpuId)
    depthBatch = depthBatch.cuda(opt.gpuId)
    segBatch = segBatch.cuda(opt.gpuId)

    imP1Batch = imP1Batch.cuda(opt.gpuId)
    imP2Batch = imP2Batch.cuda(opt.gpuId)
    imP3Batch = imP3Batch.cuda(opt.gpuId)
    imPBatch  = imPBatch.cuda(opt.gpuId)

    globIllu1to2 = globIllu1to2.cuda(opt.gpuId)
    globIllu2to3 = globIllu2to3.cuda(opt.gpuId)
####################################


####################################
brdfDataset = dataLoader.BatchLoader(opt.dataRoot, imSize = opt.imageSize)
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 8, shuffle = False)

j = 0
directErrsNpList = np.ones( [1, 1 + opt.cascadeLevel], dtype = np.float32)
globalIllu1ErrsNpList= np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32)
globalIllu2ErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32)
globalIllu3ErrsNpList= np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32)
renderErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32)
renderGtErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32)

epoch = opt.epochId
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
for i, dataBatch in enumerate(brdfLoader):
    j += 1
    # Load data from cpu to gpu
    albedo_cpu = dataBatch['albedo']
    albedoBatch.data.resize_(albedo_cpu.shape)
    albedoBatch.data.copy_(albedo_cpu )
    normal_cpu = dataBatch['normal']
    normalBatch.data.resize_(normal_cpu.shape)
    normalBatch.data.copy_(normal_cpu )
    rough_cpu = dataBatch['rough']
    roughBatch.data.resize_(rough_cpu.shape)
    roughBatch.data.copy_(rough_cpu )
    seg_cpu = dataBatch['seg']
    segBatch.data.resize_(seg_cpu.shape)
    segBatch.data.copy_(seg_cpu )
    depth_cpu = dataBatch['depth']
    depthBatch.data.resize_(depth_cpu.shape)
    depthBatch.data.copy_(depth_cpu )

    imP1_cpu = dataBatch['imP1']
    imP1Batch.data.resize_(imP1_cpu.shape)
    imP1Batch.data.copy_(imP1_cpu )
    imP2_cpu = dataBatch['imP2']
    imP2Batch.data.resize_(imP2_cpu.shape)
    imP2Batch.data.copy_(imP2_cpu )
    imP3_cpu = dataBatch['imP3']
    imP3Batch.data.resize_(imP3_cpu.shape)
    imP3Batch.data.copy_(imP3_cpu )
    imP_cpu = dataBatch['imP']
    imPBatch.data.resize_(imP_cpu.shape)
    imPBatch.data.copy_(imP_cpu )

    nameBatch = dataBatch['name']

    nameBatch = [osp.join(opt.experiment, '-'.join(x.split('/')[-2:]) ) for x in nameBatch ]


    ########################################################
    # Build the cascade network architecture #
    globalIllu2s = []
    globalIllu3s = []
    renderImgs = []
    renderImgGts = []
    n = 0
    inputGlob2 = torch.cat([imP1Batch, albedoBatch,
        normalBatch, roughBatch, depthBatch, segBatch], dim=1)
    globalIllu2 = globIllu1to2(inputGlob2.detach() ).detach()
    globalIllu2s.append(globalIllu2 )
    inputGlob3 = torch.cat([globalIllu2s[n], albedoBatch,
        normalBatch, roughBatch, depthBatch, segBatch], dim=1)
    globalIllu3 = globIllu2to3(inputGlob3.detach() )
    globalIllu3s.append(globalIllu3 )
    renderImg = (imP1Batch+globalIllu2+globalIllu3 + 2) * segBatch.expand_as(imP1Batch)
    renderImgGt = (imP1Batch+imP2Batch+imP3Batch + 2) * segBatch.expand_as(imP1Batch)
    renderImgs.append(renderImg )
    renderImgGts.append(renderImgGt )
    ########################################################

    directErrs = []
    globalIllu2Errs = []
    globalIllu3Errs = []
    renderErrs = []
    renderGtErrs = []
    pixelNum = torch.sum(segBatch ).cpu().data.item()
    for m in range(0, n + 1):
        directErrs.append( torch.sum( (imPBatch - imP1Batch)
                * (imPBatch - imP1Batch) * segBatch.expand_as(imPBatch) ) / pixelNum / 3.0 )
        globalIllu2Errs.append( torch.sum( (globalIllu2s[m] - imP2Batch)
                * (globalIllu2s[m] - imP2Batch) * segBatch.expand_as(imP2Batch) ) / pixelNum / 3.0 )
        globalIllu3Errs.append(torch.sum( (globalIllu3s[m] - imP3Batch)
                * (globalIllu3s[m] - imP3Batch) * segBatch.expand_as(imP3Batch) ) / pixelNum / 3.0 )
        renderErrs.append( torch.sum( (renderImgs[m] - imPBatch)
                * (renderImgs[m] - imPBatch) * segBatch.expand_as(imPBatch) ) / pixelNum / 3.0 )
        renderGtErrs.append( torch.sum( (renderImgGts[m] - imPBatch)
                * (renderImgGts[m] - imPBatch) * segBatch.expand_as(imPBatch) ) / pixelNum / 3.0 )


    # Output testing error
    utils.writeErrToScreen('directErrs', directErrs, epoch, j)
    utils.writeErrToScreen('globalIllu2', globalIllu2Errs, epoch, j)
    utils.writeErrToScreen('globalIllu3', globalIllu3Errs, epoch, j)
    utils.writeErrToScreen('renderErrs', renderErrs, epoch, j)
    utils.writeErrToScreen('renderGtErrs', renderGtErrs, epoch, j)

    utils.writeErrToFile('directErrs', directErrs, testingLog, epoch, j)
    utils.writeErrToFile('globalIllu2', globalIllu2Errs, testingLog, epoch, j)
    utils.writeErrToFile('globalIllu3', globalIllu3Errs, testingLog, epoch, j)
    utils.writeErrToFile('renderErrs', renderErrs, testingLog, epoch, j)
    utils.writeErrToFile('renderGtErrs', renderGtErrs, testingLog, epoch, j)

    directErrsNpList = np.concatenate( [directErrsNpList, utils.turnErrorIntoNumpy(directErrs) ], axis=0)
    globalIllu2ErrsNpList = np.concatenate( [globalIllu2ErrsNpList, utils.turnErrorIntoNumpy(globalIllu2Errs)], axis=0)
    globalIllu3ErrsNpList = np.concatenate( [globalIllu3ErrsNpList, utils.turnErrorIntoNumpy(globalIllu3Errs)], axis=0)
    renderErrsNpList = np.concatenate( [renderErrsNpList, utils.turnErrorIntoNumpy(renderErrs)], axis=0 )
    renderGtErrsNpList = np.concatenate( [renderGtErrsNpList, utils.turnErrorIntoNumpy(renderGtErrs)], axis=0 )

    utils.writeNpErrToScreen('globalIllu2_Accu:', np.mean(globalIllu2ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('globalIllu3_Accu:', np.mean(globalIllu3ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('directErrs_Accu:', np.mean(directErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('renderErrs_Accu:', np.mean(renderErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('renderGtErrs_Accu:', np.mean(renderGtErrsNpList[1:j+1, :], axis=0), epoch, j)

    utils.writeNpErrToFile('globalIllu2_Accu:', np.mean(globalIllu2ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('globalIllu3_Accu:', np.mean(globalIllu3ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('directErrs_Accu:', np.mean(directErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('renderErrs_Accu:', np.mean(renderErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('renderGtErrs_Accu:', np.mean(renderGtErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)


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
        vutils.save_image( ( ( 0.5*(imP1Batch + 1)*segBatch.expand_as(imP1Batch))**(1.0/2.2) ).data ,
                '{0}/{1}_imP1.png'.format(opt.experiment, j) )

        vutils.save_image( ( ( 0.5*(imP2Batch + 1)*segBatch.expand_as(imP2Batch))**(1.0/2.2) ).data ,
                '{0}/{1}_imP2.png'.format(opt.experiment, j) )
        imP2NameBatch = [x + '_imP2.png' for x in nameBatch ]
        utils.writeImageToFile( 0.5*(imP2Batch + 1) * segBatch.expand_as(imP2Batch), imP2NameBatch, True)
        vutils.save_image( ( ( 0.5*(imP3Batch + 1)*segBatch.expand_as(imP3Batch))**(1.0/2.2) ).data ,
                '{0}/{1}_imP3.png'.format(opt.experiment, j) )
        imP3NameBatch = [x + '_imP3.png' for x in nameBatch ]
        utils.writeImageToFile( 0.5*(imP3Batch + 1) * segBatch.expand_as(imP3Batch), imP3NameBatch, True)
        vutils.save_image( ( (0.5*(imPBatch + 1)*segBatch.expand_as(imPBatch) )**(1.0/2.2) ).data,
                '{0}/{1}_imP.png'.format(opt.experiment, j) )

        # Save the predicted results
        n = 0
        vutils.save_image( ( ( 0.5*(globalIllu2s[n] + 1)*segBatch.expand_as(imP2Batch) )**(1.0/2.2) ).data,
                '{0}/{1}_imP2Pred_{2}.png'.format(opt.experiment, j, n) )
        imP2PredNameBatch = [x + '_imP2Pred.png' for x in nameBatch ]
        utils.writeImageToFile( 0.5*(globalIllu2s[n] + 1) * segBatch.expand_as(imP2Batch), imP2PredNameBatch, True)
        vutils.save_image( ( ( 0.5*(globalIllu3s[n] + 1)*segBatch.expand_as(imP3Batch) )**(1.0/2.2) ).data,
                '{0}/{1}_imP3Pred_{2}.png'.format(opt.experiment, j, n) )
        imP3PredNameBatch = [x + '_imP3Pred.png' for x in nameBatch ]
        utils.writeImageToFile( 0.5*(globalIllu3s[n] + 1) * segBatch.expand_as(imP3Batch), imP3PredNameBatch, True)

        vutils.save_image( ( ( 0.5*(renderImgs[n] + 1)*segBatch.expand_as(renderImgs[n]) )**(1.0/2.2) ).data,
                '{0}/{1}_imRender_{2}.png'.format(opt.experiment, j, n) )
        vutils.save_image( ( ( 0.5*(renderImgGts[n] + 1)*segBatch.expand_as(renderImgGts[n]) )**(1.0/2.2) ).data,
                '{0}/{1}_imRenderGt_{2}.png'.format(opt.experiment, j, n) )

testingLog.close()

np.save('{0}/directErrs_{1}.npy'.format(opt.experiment, epoch), directErrsNpList[1:, :] )
np.save('{0}/globalIllu2_{1}.npy'.format(opt.experiment, epoch), globalIllu2ErrsNpList[1:, :] )
np.save('{0}/globalIllu3_{1}.npy'.format(opt.experiment, epoch), globalIllu3ErrsNpList[1:, :] )
np.save('{0}/renderErrs_{1}.npy'.format(opt.experiment, epoch), renderErrsNpList[1:, :] )
np.save('{0}/renderGtErrs_{1}.npy'.format(opt.experiment, epoch), renderGtErrsNpList[1:, :] )
