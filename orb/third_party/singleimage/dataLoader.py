import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils.data import Dataset
import os
import scipy.ndimage as ndimage
try:
    import h5py
except:
    pass


class BatchLoader(Dataset):
    def __init__(self, dataRoot, imSize = 256, isRandom=True, phase='TRAIN', rseed = 0, cascade = None ):
        self.dataRoot = dataRoot
        self.imSize = imSize
        self.phase = phase.upper()

        shapeList = glob.glob(osp.join(dataRoot, 'Shape__*') )
        shapeList = sorted(shapeList)

        self.cascade = cascade

        self.albedoList = []
        self.F0List = []
        for shape in shapeList:
            albedoNames = glob.glob(osp.join(shape, '*albedo.png') )
            albedoNames = sorted(albedoNames )
            self.albedoList = self.albedoList + albedoNames

        if rseed is not None:
            random.seed(rseed)

        # BRDF parameter
        self.normalList = [x.replace('albedo', 'normal') for x in self.albedoList]
        self.roughList = [x.replace('albedo', 'rough') for x in self.albedoList]
        self.segList = [x.replace('albedo', 'seg') for x in self.albedoList]

        # Rendered Image
        self.imPList = [x.replace('albedo', 'imgPoint') for x in self.albedoList]
        self.imEList = [x.replace('albedo', 'imgEnv') for x in self.albedoList]
        self.imP1List = [x.replace('albedo', 'imgPoint_b1') for x in self.albedoList]
        self.imP2List = [x.replace('albedo', 'imgPoint_b2') for x in self.albedoList]
        self.imP3List = [x.replace('albedo', 'imgPoint_b3') for x in self.albedoList]

        # Geometry
        self.depthList = [x.replace('albedo', 'depth').replace('png', 'dat') for x in self.albedoList]

        # Environment Map
        self.SHList = []
        self.nameList = []
        for x in self.albedoList:
            suffix = '/'.join(x.split('/')[0:-1])
            fileName = x.split('/')[-1]
            fileName = fileName.split('_')
            self.SHList.append(osp.join(suffix, '_'.join(fileName[0:2]) + '.npy'  ) )
            self.nameList.append(osp.join(suffix, '_'.join(fileName[0:3]) ) )

        # Permute the image list
        self.count = len(self.albedoList)
        self.perm = list(range(self.count) )
        if isRandom:
            random.shuffle(self.perm)


    def __len__(self):
        return len(self.perm)


    def __getitem__(self, ind):
        # Read segmentation
        seg = 0.5 * self.loadImage(self.segList[self.perm[ind] ] ) + 0.5
        seg = (seg[0, :, :] > 0.999999).astype(dtype = np.int)
        seg = ndimage.binary_erosion(seg, structure = np.ones( (2, 2) ) ).astype(dtype = np.float32 )
        seg = seg[np.newaxis, :, :]

        # Read albedo
        albedo = self.loadImage(self.albedoList[self.perm[ind] ] )
        albedo = albedo * seg

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage(self.normalList[self.perm[ind] ] )
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]
        normal = normal * seg

        # Read roughness
        rough = self.loadImage(self.roughList[self.perm[ind] ] )[0:1, :, :]
        rough = (rough * seg)

        # Read rendered images
        imP = self.loadImage(self.imPList[self.perm[ind] ], isGama = True)
        imP = imP * seg
        imE = self.loadImage(self.imEList[self.perm[ind] ], isGama = True)
        imEbg = imE.copy()
        imE = imE * seg
        imP1 = self.loadImage(self.imP1List[self.perm[ind] ], isGama = True)
        imP1 = imP1 * seg
        imP2 = self.loadImage(self.imP2List[self.perm[ind] ], isGama = True)
        imP2 = imP2 * seg
        imP3 = self.loadImage(self.imP3List[self.perm[ind] ], isGama = True)
        imP3 = imP3 * seg


        with open(self.depthList[self.perm[ind] ], 'rb') as f:
            byte = f.read()
            if len(byte) == 256 * 256 * 3 * 4:
                depth = np.array(struct.unpack(str(256*256*3)+'f', byte), dtype=np.float32)
                depth = depth.reshape([256, 256, 3])[:, :, 0:1]
                depth = depth.transpose([2, 0, 1] )
                depth = depth * seg
            elif len(byte) == 512 * 512 * 3 * 4:
                print(self.depthList[self.perm[ind] ])
                assert(False )


        if not os.path.isfile(self.SHList[self.perm[ind] ] ):
            #print('Fail to load {0}'.format(self.SHList[self.perm[ind] ] ) )
            SH = np.zeros([3, 9], dtype=np.float32)
        else:
            SH = np.load(self.SHList[self.perm[ind] ]).transpose([1, 0] )[:, 0:9]
            SH = SH.astype(np.float32)[::-1, :]
        name = self.nameList[self.perm[ind] ]

        # Scale the input
        scalePoint = 1.7
        imP = (imP + 1) * scalePoint - 1
        imP1 = (imP1 + 1) * scalePoint - 1
        imP2 = (imP2 + 1) * scalePoint - 1
        imP3 = (imP3 + 1) * scalePoint - 1

        # Scale the Environment
        scaleEnv = 0.5
        imE = (imE + 1) * scaleEnv - 1
        imEbg = (imEbg + 1) * scaleEnv - 1
        SH = SH * scaleEnv

        imP = np.clip(imP, -1, 1)
        imP1 = np.clip(imP1, -1, 1)
        imP2 = np.clip(imP2, -1, 1)
        imP3 = np.clip(imP3, -1, 1)
        imE = np.clip(imE, -1, 1)

        imP = imP * seg
        imP1 = imP1
        imP2 = imP2 * seg
        imP3 = imP3 * seg
        imE = imE * seg

        if self.cascade is not None:

            albedoName = self.albedoList[self.perm[ind] ][0:-4] + '_c{0}.hdf5'.format(self.cascade)
            normalName = self.normalList[self.perm[ind] ][0:-4] + '_c{0}.hdf5'.format(self.cascade)
            roughName = self.roughList[self.perm[ind] ][0:-4] + '_c{0}.hdf5'.format(self.cascade)
            depthName = self.depthList[self.perm[ind] ][0:-4] + '_c{0}.hdf5'.format(self.cascade)

            imP2Name = self.imP2List[self.perm[ind] ][0:-4] + '_c{0}.hdf5'.format(self.cascade)
            imP3Name = self.imP3List[self.perm[ind] ][0:-4] + '_c{0}.hdf5'.format(self.cascade)

            envName = depthName.replace('depth', 'env')

            albedoPred = self.loadHdf5(albedoName )
            albedoPred = albedoPred * seg
            normalPred = self.loadHdf5(normalName )
            normalPred = normalPred / np.sqrt(np.maximum(np.sum(normalPred * normalPred, axis=0), 1e-5) )[np.newaxis, :]
            normalPred = normalPred * seg
            roughPred = self.loadHdf5(roughName )
            roughPred = roughPred * seg
            depthPred = self.loadHdf5(depthName )
            depthPred = depthPred * seg

            imP2Pred = self.loadHdf5(imP2Name )
            imP2Pred = imP2Pred * seg
            imP3Pred = self.loadHdf5(imP3Name )
            imP3Pred = imP3Pred * seg

            envPred = self.loadHdf5(envName)

            batchDict = {'albedo': albedo,
                        'normal': normal,
                        'rough': rough,
                        'depth': depth,
                        'seg': seg,
                        'imP': imP,
                        'imE':  imE,
                        'imEbg': imEbg,
                        'imP1': imP1,
                        'imP2': imP2,
                        'imP3': imP3,
                        'SH': SH,
                        'name': name,
                        'albedoName': self.albedoList[self.perm[ind] ],
                        'albedoPred': albedoPred,
                        'normalPred': normalPred,
                        'roughPred': roughPred,
                        'imP2Pred': imP2Pred,
                        'imP3Pred': imP3Pred,
                        'depthPred': depthPred,
                        'envPred': envPred}
        else:
            batchDict = {'albedo': albedo,
                         'normal': normal,
                         'rough': rough,
                         'depth': depth,
                         'seg': seg,
                         'imP': imP,
                         'imE':  imE,
                         'imEbg': imEbg,
                         'imP1': imP1,
                         'imP2': imP2,
                         'imP3': imP3,
                         'SH': SH,
                         'name': name,
                         'albedoName': self.albedoList[self.perm[ind] ]}



        return batchDict


    def loadImage(self, imName, isGama = False):
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName) )
            im = np.zeros([3, self.imSize, self.imSize], dtype=np.float32)
            return im

        im = Image.open(imName)
        im = self.imResize(im)
        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])
        return im

    def imResize(self, im):
        w0, h0 = im.size
        assert( (w0 == h0) )
        im = im.resize((self.imSize, self.imSize), Image.ANTIALIAS)
        return im

    def loadHdf5(self, name ):
        hf = h5py.File(name, 'r')
        data = np.array(hf.get('data' ) )
        return data

