import dataLoader
import argparse
import numpy as np
import scipy.io as io

parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', default='../Data/test/', help='path to real image distorted by water')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic testing setting
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')

opt = parser.parse_args()
if opt.experiment is None:
    opt.experiment = 'inDirectDistribution'

brdfDataset = dataLoader.BatchLoader(opt.dataRoot, imSize = opt.imageSize)
imP1Sum = 0
imP2Sum = 0
imP3Sum = 0
imPSum = 0
pixelNum = 60000
for i in range(0, len(brdfDataset) ):
    print('%d/%d' % (i, len(brdfDataset) ) )
    dataBatch = brdfDataset[i]
    seg = dataBatch['seg']
    imP1 = 0.5*(dataBatch['imP1'] + 1) * seg
    imP2 = 0.5*(dataBatch['imP2'] + 1) * seg
    imP3 = 0.5*(dataBatch['imP3'] + 1) * seg
    imP = 0.5*(dataBatch['imP'] + 1) * seg

    imP1Sum += np.sum(imP1 ) / pixelNum / 3.0
    imP2Sum += np.sum(imP2 ) / pixelNum / 3.0
    imP3Sum += np.sum(imP3 ) / pixelNum / 3.0
    imPSum +=  np.sum(imP) / pixelNum / 3.0

imP1Prop = imP1Sum / imPSum
imP2Prop = imP2Sum / imPSum
imP3Prop = imP3Sum / imPSum
imOtherProp = 1 - imP1Prop - imP2Prop - imP3Prop
print('Bounce: %.5f %.5f %.5f %.5f' % (imP1Prop, imP2Prop, imP3Prop, imOtherProp) )
