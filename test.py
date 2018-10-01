
import torch

from models.iRevNet import iRevNet


model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                nChannels=None, nClasses=1000, init_ds=2,
                dropout_rate=0., affineBN=True, in_shape=[3, 224, 224],
                mult=4)
y = model(torch.randn(1, 3, 224, 224))