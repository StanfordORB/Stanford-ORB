from __future__ import print_function
import numpy as np
from PIL import Image
import computeSH
import cv2
import os.path as osp
import torch
from torch.autograd import Variable
try:
    import h5py
except:
    pass

def writeErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n].data.item(), end = ' ')
    print('.')

def writeCoefToScreen(coefName, coef, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(coefName), end=' ')
    coefNp = coef.cpu().data.numpy()
    for n in range(0, len(coefNp) ):
        print('%.6f' % coefNp[n], end = ' ')
    print('.')

def writeNpErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n], end = ' ')
    print('.')

def writeErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:'% (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n].data.item() )
    fileOut.write('.\n')

def writeCoefToFile(coefName, coef, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}: ' % (epoch, j) ).format(coefName) )
    coefNp = coef.cpu().data.numpy()
    for n in range(0, len(coefNp) ):
        fileOut.write('%.6f ' % coefNp[n] )
    fileOut.write('.\n')

def writeNpErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n] )
    fileOut.write('.\n')

def turnErrorIntoNumpy(errorArr):
    errorNp = []
    for n in range(0, len(errorArr) ):
        errorNp.append(errorArr[n].data.item() )
    return np.array(errorNp)[np.newaxis, :]

def visualizeSH(dst, coefs, names, nrows, ncols, gridRows, gridCols, edge = 5):
    imgNum = len(names)
    coefs = coefs.data.cpu().numpy()
    assert(gridRows * gridCols >= imgNum)

    nRows = gridRows * nrows + edge * (gridRows + 1)
    nCols = gridCols * ncols + edge * (gridCols + 1)
    imArr = np.zeros([nRows, nCols, 3], dtype = np.float32)
    for rId in range(0, gridRows):
        for cId in range(0, gridCols):
            if rId * gridCols + cId >= imgNum:
                break
            n = rId * gridCols + cId
            name = names[n]
            coef = coefs[n, :].transpose([1, 0])
            sr = edge * (rId+1) + rId * nrows
            sc = edge * (cId+1) + cId * ncols

            root = '/'.join(name.split('/')[0:-1] )
            fileName = name.split('/')[-1]
            camFile = osp.join(root, fileName.split('_')[0] + '.txt')
            with open(camFile, 'r') as f:
                lines = f.readlines()
                angle = lines[0].strip().split(' ')
                angleUp = lines[1].strip().split(', ')
                theta = float(angle[0])
                phi = float(angle[1])
                thetaUp = float(angleUp[0])
                phiUp = float(angleUp[1])
                cameraLoc = np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi), np.cos(theta)], dtype=np.float32 )
                cameraUp = np.array([np.sin(thetaUp) * np.cos(phiUp),
                    np.sin(thetaUp) * np.sin(phiUp), np.cos(thetaUp)], dtype=np.float32 )
                coef = np.concatenate([coef, np.zeros([16, 3], dtype=np.float32)], axis=0 )
                imRecons = computeSH.reconstImageFromSHAfterRotate(coef, cameraLoc, cameraUp, nrows=128, ncols=256, isInv=True)
                imArr[sr : sr+nrows, sc : sc+ncols, :] = imRecons

    imArr = np.clip(imArr, 0, 1)
    imArr = (255 * imArr).astype(np.uint8)
    imArr = Image.fromarray(imArr)
    imArr.save(dst)

def visualizeGtEnvmap(dst, names, nrows, ncols, gridRows, gridCols, edge=5):
    imgNum = len(names)
    assert(gridRows * gridCols >= imgNum)

    nRows = gridRows * nrows + edge * (gridRows + 1)
    nCols = gridCols * ncols + edge * (gridCols + 1)
    imArr = np.zeros([nRows, nCols, 3], dtype = np.float32)
    for rId in range(0, gridRows):
        for cId in range(0, gridCols):
            if rId * gridCols + cId >= imgNum:
                break
            n = rId * gridCols + cId
            sr = edge * (rId+1) + rId * nrows
            sc = edge * (cId+1) + cId * ncols

            name = names[n]
            root = '/'.join(name.split('/')[0:-1] )
            fileName = name.split('/')[-1]
            envFile = osp.join(root, fileName.split('_')[1] + '.txt')
            with open(envFile, 'r') as f:
                envName = f.readlines()[0]
                envName = envName.strip()
            im = cv2.imread(envName, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[:, :, ::-1]
            im = cv2.resize(im, (ncols, nrows), interpolation=cv2.INTER_AREA)
            imArr[sr : sr + nrows, sc : sc + ncols, :] = im
    imArr = np.clip(imArr, 0, 1)
    imArr = (255 * imArr).astype(np.uint8)
    imArr = Image.fromarray(imArr)
    imArr.save(dst)

def computeConfMap(imBatch, segBatch, coef, gpuId):
    im = 0.5 * (imBatch + 1)

    coef0, coef1 = torch.split(coef, 1)
    coef0 = coef0.view(1, 1, 1, 1)
    coef1 = coef1.view(1, 1, 1, 1)
    minIm, _ = torch.min(im, dim=1)
    w0 = (1 - torch.exp( -(1-minIm) / 0.02) ).unsqueeze(1)
    w1 = Variable(0 * torch.FloatTensor(segBatch.size() ).cuda(gpuId) ) + 1
    return coef0 * w0 + coef1 * w1


def writeImageToFile(imgBatch, nameBatch, isGama = False):
    batchSize = imgBatch.size(0)
    for n in range(0, batchSize):
        img = imgBatch[n, :, :, :].data.cpu().numpy()
        img = np.clip(img, 0, 1)
        if isGama:
            img = np.power(img, 1.0/2.2)
        img = (255 *img.transpose([1, 2, 0] ) ).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img )
        img.save(nameBatch[n] )

def writeDataToFile(imgBatch, nameBatch ):
    batchSize = imgBatch.size(0)
    for n in range(0, batchSize):
        img = imgBatch[n, :].data.cpu().numpy()
        name = nameBatch[n]

        hf = h5py.File(name, 'w')
        hf.create_dataset('data', data=img, compression='lzf' )
        hf.close()

def writeDepthToFile(depthBatch, nameBatch):
    batchSize = depthBatch.size(0)
    for n in range(0, batchSize):
        depth = depthBatch[n, :, :, :].data.cpu().numpy().squeeze()
        np.save(nameBatch[n], depth)

def writeEnvToFile(SHBatch, nameBatch):
    batchSize = SHBatch.size(0)
    for n in range(0, batchSize):
        SH = SHBatch[n, :, :].data.cpu().numpy()
        np.save(nameBatch[n], SH)

def writeAlbedoNameToFile(fileName, albedoNameBatch):
    with open(fileName, 'w') as fileOut:
        for n in range(0, len(albedoNameBatch) ):
            fileOut.write('%s\n' % albedoNameBatch[n] )
