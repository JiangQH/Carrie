from carrie.layers.convolution import Convolution
import numpy as np
from PIL import Image
import cv2
NEW_HEIGHT = 224
NEW_WIDTH = 224
CHANNEL = 3
BATCH_SIZE = 1
def test_conv():
    conv = Convolution('conv1_1')
    img = np.asarray(Image.open('../data/test.jpg'))
    img = cv2.resize(img, (NEW_HEIGHT, NEW_WIDTH))
    img = img.transpose((2, 1, 0))
    data = np.zeros((BATCH_SIZE, CHANNEL, NEW_HEIGHT, NEW_WIDTH))
    data[0, ...] = img
    bottoms = [data]
    conv.initJob(bottoms)
    y = conv.forward(bottoms)


