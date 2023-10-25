import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class encoderInitial_point(nn.Module):
    def __init__(self):
        super(encoderInitial_point, self).__init__()
        # Input should be segmentation, image with environment map, image with point light + environment map
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)), True )
        x2 = F.relu(self.bn2(self.conv2(x1)), True )
        x3 = F.relu(self.bn3(self.conv3(x2)), True )
        x4 = F.relu(self.bn4(self.conv4(x3)), True )
        x5 = F.relu(self.bn5(self.conv5(x4)), True )
        x = F.relu(self.bn6(self.conv6(x5)), True )
        return x1, x2, x3, x4, x5, x


class encoderInitial(nn.Module):
    def __init__(self):
        super(encoderInitial, self).__init__()
        # Input should be segmentation, image with environment map, image with point light + environment map
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)), True )
        x2 = F.relu(self.bn2(self.conv2(x1)), True )
        x3 = F.relu(self.bn3(self.conv3(x2)), True )
        x4 = F.relu(self.bn4(self.conv4(x3)), True )
        x5 = F.relu(self.bn5(self.conv5(x4)), True )
        x = F.relu(self.bn6(self.conv6(x5)), True )
        return x1, x2, x3, x4, x5, x


class decoderInitial(nn.Module):
    def __init__(self, mode=0):
        super(decoderInitial, self).__init__()
        # branch for normal prediction
        self.dconv0 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn0 = nn.BatchNorm2d(256)
        self.dconv1 = nn.ConvTranspose2d(in_channels=256+256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(256)
        self.dconv2 = nn.ConvTranspose2d(in_channels=256+256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128+128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dconv4 = nn.ConvTranspose2d(in_channels=64+64,  out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dconv5 = nn.ConvTranspose2d(in_channels=32+32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn5 = nn.BatchNorm2d(64)

        self.convFinal = nn.Conv2d(in_channels=64, out_channels = 3, kernel_size = 5, stride=1, padding=2, bias=True)
        # Mode 0: Albedo
        # Mode 1: Normal
        # Mode 2: Roughness
        # Mode 3: Depth
        self.mode = mode
        assert( mode >= 0 and mode <= 3)

    def forward(self, x1, x2, x3, x4, x5, x):
        x_d1 = F.relu( self.dbn0(self.dconv0(x) ), True)
        x_d1_next = torch.cat( (x_d1, x5), dim = 1)
        x_d2 = F.relu( self.dbn1(self.dconv1(x_d1_next) ), True)
        x_d2_next = torch.cat( (x_d2, x4), dim = 1)
        x_d3 = F.relu( self.dbn2(self.dconv2(x_d2_next) ), True)
        x_d3_next = torch.cat( (x_d3, x3), dim = 1)
        x_d4 = F.relu( self.dbn3(self.dconv3(x_d3_next) ), True)
        x_d4_next = torch.cat( (x_d4, x2), dim = 1)
        x_d5 = F.relu( self.dbn4(self.dconv4(x_d4_next) ), True)
        x_d5_next = torch.cat( (x_d5, x1), dim = 1)
        x_d6 = F.relu( self.dbn5(self.dconv5(x_d5_next) ), True)
        x_orig  = torch.tanh( self.convFinal(x_d6) )
        if self.mode == 0:
            x_out = x_orig
        elif self.mode == 1:
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1) ).expand_as(x_orig);
            x_out = x_orig / norm
        elif self.mode == 2:
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 3:
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = 1 / (0.4 * (x_out + 1) + 0.25 )
        return x_out


class envmapInitial(nn.Module):
    def __init__(self, numCoef = 9):
        super(envmapInitial, self).__init__()
        self.conv =nn.Conv2d(in_channels = 512, out_channels=1024, kernel_size=4, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(1024)
        self.numCoef = numCoef

        self.regression = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(1024, self.numCoef * 3)
                )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x) ) )
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, dim=2)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.regression(x))
        x = x.view( x.size(0), 3, self.numCoef)
        return x



class globalIllumination(nn.Module):
    def __init__(self):
        super(globalIllumination, self).__init__()
        # Encoder, the input will be albedo: 3, normal: 3, depth: 1, roughness: 1, n bounce: 3 segmentation: 1

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=6, stride=2, padding=2, bias=False)
        #self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, dilation=3, stride=2, padding=7, bias=False)
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, dilation=3, stride=2, padding=7, bias=False)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        # Decoder
        self.dconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(256)
        self.dconv2 = nn.ConvTranspose2d(in_channels=256 + 256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128 + 128, out_channels=64, kernel_size=6, dilation=3, stride=2, padding=7, bias=False)
        #self.dconv3 = nn.ConvTranspose2d(in_channels=128 + 128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dconv4 = nn.ConvTranspose2d(in_channels=64 + 64,  out_channels=32, kernel_size=6, dilation=3, stride=2, padding=7, bias=False)
        #self.dconv4 = nn.ConvTranspose2d(in_channels=64 + 64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dconv5 = nn.ConvTranspose2d(in_channels=32 + 32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn5 = nn.BatchNorm2d(64)
        self.convFinal = nn.Conv2d(in_channels=64, out_channels = 3, kernel_size = 5, stride=1, padding=2, bias=True)
        #self.convFinal = nn.Conv2d(in_channels=32, out_channels = 3, kernel_size = 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)), True )
        x2 = F.relu(self.bn2(self.conv2(x1)), True )
        x3 = F.relu(self.bn3(self.conv3(x2)), True )
        x4 = F.relu(self.bn4(self.conv4(x3)), True )
        x5 = F.relu(self.bn5(self.conv5(x4)), True )
        x_d1 = F.relu( self.dbn1(self.dconv1(x5) ), True)
        x_d1_next = torch.cat( (x_d1, x4), dim = 1)
        x_d2 = F.relu( self.dbn2(self.dconv2(x_d1_next) ), True)
        x_d2_next = torch.cat( (x_d2, x3), dim = 1)
        x_d3 = F.relu( self.dbn3(self.dconv3(x_d2_next) ), True)
        x_d3_next = torch.cat( (x_d3, x2), dim = 1)
        x_d4 = F.relu( self.dbn4(self.dconv4(x_d3_next) ), True)
        x_d4_next = torch.cat( (x_d4, x1), dim = 1)
        x_d5 = F.relu( self.dbn5(self.dconv5(x_d4_next) ), True)
        x_out  = torch.tanh( self.convFinal(x_d5) )
        return x_out


class residualBlock(nn.Module):
    def __init__(self, nchannels=128):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nchannels, out_channels=nchannels, kernel_size=3, dilation = 2, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(nchannels)
        self.conv2 = nn.Conv2d(in_channels=nchannels, out_channels=nchannels, kernel_size=3, dilation = 2, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(nchannels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x) ), True)
        y = F.relu(self.bn2(self.conv2(y) ), True)
        return y+x

class refineEncoder(nn.Module):
    def __init__(self):
        super(refineEncoder, self).__init__()
        # Encoder: segmentation: 1
        # albedo: 3, normal: 3 roughness: 1, depth: 1, two input image 7, total chnum = 15
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.res1 = residualBlock(128)
        self.res2 = residualBlock(128)
        self.res3 = residualBlock(128)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x) ) )
        x2 = F.relu(self.bn2(self.conv2(x1) ) )
        x3 = self.res1(x2)
        x3 = self.res2(x3)
        x3 = self.res3(x3)
        return x1, x3

class refineEncoder(nn.Module):
    def __init__(self):
        super(refineEncoder, self).__init__()
        # Encoder: the input will be error: 3 segmentation: 1
        # albedo: 3, normal: 3 roughness: 1, depth: 1, two input image 6, total chnum = 18
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.res1 = residualBlock(128)
        self.res2 = residualBlock(128)
        self.res3 = residualBlock(128)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x) ) )
        x2 = F.relu(self.bn2(self.conv2(x1) ) )
        x3 = self.res1(x2)
        x3 = self.res2(x3)
        x3 = self.res3(x3)
        return x1, x3

class refineDecoder(nn.Module):
    def __init__(self, mode = 0):
        super(refineDecoder, self).__init__()
        self.res1 = residualBlock(128)
        self.res2 = residualBlock(128)
        self.res3 = residualBlock(128)
        self.dconv0 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn0 = nn.BatchNorm2d(64)
        self.dconv1 = nn.ConvTranspose2d(in_channels=64 + 64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(64)
        self.convFinal = nn.Conv2d(in_channels=64, out_channels = 3, kernel_size = 5, stride=1, padding=2, bias=True)
        # Mode 0: Albedo
        # Mode 1: Normal
        # Mode 2: Roughness
        # Mode 3: Depth
        self.mode = mode

    def forward(self, x1, x3):
        x3 = self.res1(x3)
        x3 = self.res2(x3)
        x3 = self.res3(x3)
        x_d0 = F.relu(self.dbn0(self.dconv0(x3) ), True)
        x_d0_next = torch.cat( (x1, x_d0), dim=1)
        x_d1 = F.relu(self.dbn1(self.dconv1(x_d0_next) ), True)
        x_orig = torch.tanh(self.convFinal(x_d1) )
        if self.mode == 0:
            x_out = x_orig
        elif self.mode == 1:
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1) ).expand_as(x_orig);
            x_out = x_orig / norm
        elif self.mode == 2:
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 3:
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = 1 / (0.4 * (x_out + 1) + 0.25 )
        return x_out

class refineEnvDecoder(nn.Module):
    def __init__(self):
        super(refineEnvDecoder, self).__init__()
        self.numCoef = 9
        self.conv1 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size = 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.AvgPool2d(kernel_size = 4)
        self.projection = nn.Sequential(
                    nn.Linear(3 * self.numCoef, 512),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(512, 512)
                )
        self.regression = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(512, self.numCoef * 3)
                )


    def forward(self, x, pred):
        x = F.relu(self.bn1(self.conv1(x) ) )
        x = F.relu(self.bn2(self.conv2(x) ) )
        x = F.relu(self.bn3(self.conv3(x) ) )
        x = F.relu(self.bn4(self.conv4(x) ) )
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        pred = pred.view( pred.size(0), 3*self.numCoef)
        pred = self.projection(pred)
        x = torch.cat([pred, x], dim=1)
        x = torch.tanh(self.regression(x) )
        x = x.view( x.size(0), 3, self.numCoef)
        return x

class renderingLayer():
    def __init__(self, imSize=256, fov=60, F0=0.05, cameraPos = [0, 0, 0], lightPos = [0, 0, 0],
            lightPower=5.95, gpuId = 0, isCuda = True):
        self.imSize = imSize
        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.lightPos = np.array(lightPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.lightPower = lightPower
        self.yRange = self.xRange = 1 * np.tan(self.fov/2)
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imSize),
                np.linspace(-self.yRange, self.yRange, imSize) )
        y = np.flip(y, axis=0)
        z = -np.ones((imSize, imSize), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - self.pCoord
        l = self.lightPos  - self.pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        l = l / np.sqrt(np.maximum(np.sum(l*l, axis=1), 1e-12)[:, np.newaxis, :, :] )

        v = v.astype(dtype = np.float32)
        l = l.astype(dtype = np.float32)
        h = (v + l) / 2;
        h = h / np.sqrt(np.maximum(np.sum(h*h, axis=1), 1e-12)[:, np.newaxis, :, :] )
        h = h.astype(dtype = np.float32)

        self.v = Variable(torch.from_numpy(v) )
        self.l = Variable(torch.from_numpy(l) )
        self.h = Variable(torch.from_numpy(h) )
        temp = Variable(torch.FloatTensor(1, 1, 1, 1) )
        self.pCoord = Variable(torch.from_numpy(self.pCoord) )
        self.lightPos = Variable(torch.from_numpy(self.lightPos) )

        if isCuda:
            self.v = self.v.cuda(gpuId)
            self.l = self.l.cuda(gpuId)
            self.h = self.h.cuda(gpuId)
            self.pCoord = self.pCoord.cuda(gpuId)
            temp = temp.cuda(gpuId)
            self.lightPos = self.lightPos.cuda(gpuId)

        vdh = torch.sum( (self.v * self.h), dim = 1)
        vdh = vdh.unsqueeze(0)
        temp.data[0] = 2.0
        self.frac0 = F0 + (1-F0) * torch.pow(temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)
        self.vdh = vdh
        self.gpuId = gpuId
        self.sDist = None


    def forward(self, diffusePred, normalPred, roughPred, distPred, segBatch):
        diffuseBatch = (diffusePred + 1.0)/2.0 / np.pi
        roughBatch = (roughPred + 1.0)/2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(normalPred * self.v.expand_as(normalPred), dim = 1), 0, 1)
        ndh = torch.clamp(torch.sum(normalPred * self.h.expand_as(normalPred), dim = 1), 0, 1)
        ndl = torch.clamp(torch.sum(normalPred * self.l.expand_as(normalPred), dim = 1), 0, 1)

        if len(ndv.size()) == 3:
            ndv = ndv.unsqueeze(1)
            ndh = ndh.unsqueeze(1)
            ndl = ndl.unsqueeze(1)

        frac = alpha2 * self.frac0.expand_as(alpha)
        nom0 = ndh * ndh * (alpha2 - 1) + 1
        nom1 = ndv * (1 - k) + k
        nom2 = ndl * (1 - k) + k
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom

        coord3D = self.pCoord.expand_as(diffuseBatch) * distPred.expand_as(diffuseBatch)
        dist2Pred = torch.sum( (self.lightPos.expand_as(coord3D) - coord3D) \
                * (self.lightPos.expand_as(coord3D) - coord3D), dim=1).unsqueeze(1)
        color = (diffuseBatch + specPred.expand_as(diffusePred) ) * ndl.expand_as(diffusePred) * \
                self.lightPower / torch.clamp(dist2Pred.expand_as(diffusePred), 1e-6)
        color = color * segBatch.expand_as(diffusePred)
        return torch.clamp(color, 0, 1)


    def forwardEnv(self, diffusePred, normalPred, roughPred, SHPred, segBatch):
        diffuseBatch = (diffusePred + 1) / 2.0 / np.pi
        batchSize = diffusePred.size(0)
        c1, c2, c3, c4, c5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708
        L0, L1_1, L10, L11, L2_2, L2_1, L20, L21, L22 = torch.split(SHPred, 1, dim=2)
        nx, ny, nz = torch.split(normalPred, 1, dim=1)
        L0, L1_1, L10, L11 = L0.contiguous(), L1_1.contiguous(), L10.contiguous(), L11.contiguous()
        L2_2, L2_1, L20, L21, L22 = L2_2.contiguous(), L2_1.contiguous(), \
                L20.contiguous(), L21.contiguous(), L22.contiguous()
        L0 = L0.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L1_1 = -L1_1.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L10 = L10.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L11 = -L11.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L2_2 = L2_2.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L2_1 = -L2_1.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L20 = L20.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L21 = -L21.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L22 = L22.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        nx = nx.expand([batchSize, 3, self.imSize, self.imSize] )
        ny = ny.expand([batchSize, 3, self.imSize, self.imSize] )
        nz = nz.expand([batchSize, 3, self.imSize, self.imSize] )

        x = c1*L22*nx + c1*L2_2*ny + c1*L21*nz + c2*L11
        y = c1*L2_2*nx - c1*L22*ny + c1*L2_1*nz + c2*L1_1
        z = c1*L21*nx + c1*L2_1*ny + c3*L20*nz + c2*L10
        w = c2*L11*nx + c2*L1_1*ny + c2*L10*nz + c4*L0 - c5*L20

        radiance = nx*x + ny*y + nz*z + w

        color = diffuseBatch * radiance * segBatch.expand_as(diffusePred)
        return torch.clamp(color, 0, 1)

