import caffe
import os
import numpy as np
from PIL import Image

model = pretrainedmodels.dict['cafferesnet101'](num_classes=1000, pretrained='imagenet')
newmodel = list(model.children())[:-1]
net = nn.Sequential(*newmodel)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        im = caffe.io.load_image(filename)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        net.forward()
        pool5 = net.blobs['pool5'].data
        print(pool5)

