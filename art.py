import lasagne
import numpy as np
import pickle
import skimage.transform
import scipy
from time import gmtime, strftime

import theano
import theano.tensor as T

from lasagne.utils import floatX

import matplotlib.pyplot as plt
import glob
import sys
import getopt
#matplotlib inline

# VGG-19, 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: non-commercial use only

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

# from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
# from lasagne.nonlinearities import softmax

from PIL import Image

# from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers import Pool2DLayer as PoolLayer
import os

time = strftime("%Y-%m-%d %H-%M-%S", gmtime()).replace(" ", "_");
os.mkdir(time);


IMAGE_W = 600

# Note: tweaked to use average pooling instead of maxpooling
def build_model():
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    return net

def deprocess(x):
    x = np.copy(x[0])
    x += MEAN_VALUES

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Download the normalized pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl
# (original source: https://bethgelab.org/deepneuralart/)

# build VGG net and load weights

net = build_model()

values = pickle.load(open('vgg19_normalized.pkl'))['param values']
lasagne.layers.set_all_param_values(net['pool5'], values)



MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (IMAGE_W, w*IMAGE_W/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*IMAGE_W/w, IMAGE_W), preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])

styleDirectory = 'style/monet'
styleImage = ''
contentImage = 'content/brad.jpg'
K = 0

try:
    opts, args = getopt.getopt(sys.argv[1:],"hs:c:k:",["style=","content="])
except getopt.GetoptError:
    print 'art.py -s <styleImage|styleDirectory> -c <contentImage> -k <number of neighbors>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'art.py -s <styleImage|styleDirectory> -c <contentImage> -k <number of neighbors>'
        sys.exit()
    elif opt in ("-s", "--style"):
        if not os.path.exists(arg):
            print 'Style directory or image does not exist'
            sys.exit(2)

        if (".jpg" in arg or ".png" in arg):
            styleImage = arg
        else:
            styleDirectory = arg
    elif opt in ("-c", "--content"):
        if (".jpg" not in arg and ".png" not in arg):
            print 'Invalid content image. Must contain a valid extension'
            sys.exit(2)

        if not os.path.exists(arg):
            print 'Content image does not exist'
            sys.exit(2)

        contentImage = arg
    elif opt in ("-k"):
        K = int(arg)
        if K < 0:
            print 'K value must be a postive integer'
            sys.exit(2)        
        K = arg

if styleImage:
    styleImageNames = [styleImage]
else:
    styleImageNames = glob.glob(styleDirectory + '/*');

if K == 0:
    K = len(styleImageNames)


if int(len(styleImageNames)) < int(K):
    print 'K value must be at least the number of style images'
    sys.exit(2)

for image in styleImageNames:
    if not os.path.exists(image):
        print 'Image does not exist'
        sys.exit(2)

    if ".jpg" not in image and ".png" not in image:
        styleImageNames.remove(image)

styleImagesList = np.empty([len(styleImageNames), 3* 600*600]) 
styleImages = {}

# Make list of file names to save
styleImgFileNames = {}
for name in styleImageNames:
    styleImgFileNames[name] = name.split('/')[-1]

for imgstr in styleImageNames:
    print imgstr
    artImg = plt.imread(imgstr)
    rawim, artImg = prep_image(artImg)
    artImgArr = np.reshape(artImg,[1, 3* 600*600])
    styleImages[imgstr] = artImgArr

rawphoto, contentImg = prep_image(plt.imread(contentImage))
contentImgReshaped = np.reshape(contentImg, [1, 3*600*600])

# find nearest style images to the content
normDistances = []
for imgstr in styleImageNames:
    dist = (np.linalg.norm(styleImages[imgstr] - contentImgReshaped), imgstr)
    normDistances.append(dist)

normDistances.sort(key=lambda tup: tup[0])

nearestStyle = normDistances[0:int(K)]
nearestStyleImages = []
map(lambda imageNeighbor: nearestStyleImages.append(prep_image(plt.imread(imageNeighbor[1]))[1]), nearestStyle)
map(lambda imageNeighbor: plt.imsave(time + '/' + styleImgFileNames[imageNeighbor[1]], plt.imread(imageNeighbor[1])), nearestStyle)

def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g


def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]
    
    loss = 1./2 * ((x - p)**2).sum()
    return loss


def style_loss(A, X, layer):

    loss = 0
    for img in A:
        N = img[layer].shape[1]
        M = img[layer].shape[2] * img[layer].shape[3]

        S = gram_matrix(img[layer])
        G = gram_matrix(X[layer])
    
        loss = loss + 1./(4 * N**2 * M**2) * ((G - S)**2).sum()

    return loss/4;

def style_loss_mean(A, X, layer):

    S = []
    for img in A:
        S.append(gram_matrix(img[layer]));

    S = sum(S)/len(S);
    
    G = gram_matrix(X[layer])
    
    N = A[0][layer].shape[1]
    M = A[0][layer].shape[2] * A[0][layer].shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((G - S)**2).sum()

    return loss;

def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()

layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
layers = {k: net[k] for k in layers}

# Precompute layer activations for photo and artwork
input_im_theano = T.tensor4()
outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

photo_features = {k: theano.shared(output.eval({input_im_theano: contentImg}))
                  for k, output in zip(layers.keys(), outputs)}
art_features = []
for i in range(0,int(K)):
    art_features.append({k: theano.shared(output.eval({input_im_theano: nearestStyleImages[i]}))
                     for k, output in zip(layers.keys(), outputs)});

# ADDED
# Get expressions for layer activations for generated image
newimg = np.copy(contentImg).reshape(1, 3, IMAGE_W, IMAGE_W)
generated_image = theano.shared(floatX(newimg))

# white noise
#generated_image.set_value(floatX(np.copy((10 * newimg.std() * np.random.random(newimg.shape)))));

#
generated_image.set_value(floatX(np.copy(newimg + (2 * newimg.std() * np.random.random(newimg.shape)))));



gen_features = lasagne.layers.get_output(layers.values(), generated_image)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

# Define loss function
losses = []

# content loss
losses.append(.05 * content_loss(photo_features, gen_features, 'conv4_2'))

# style loss
losses.append(0.2e8 * style_loss(art_features, gen_features, 'conv2_1'))
losses.append(0.2e8 * style_loss(art_features, gen_features, 'conv3_1'))
losses.append(0.2e8 * style_loss(art_features, gen_features, 'conv1_1'))
losses.append(0.2e8 * style_loss(art_features, gen_features, 'conv4_1'))
losses.append(0.2e8 * style_loss(art_features, gen_features, 'conv5_1'))

# total variation penalty
losses.append(0.1e-4 * total_variation_loss(generated_image))

total_loss = sum(losses)

grad = T.grad(total_loss, generated_image)

# Theano functions to evaluate loss and gradient
f_loss = theano.function([], total_loss)
f_grad = theano.function([], grad)

# Helper functions to interface with scipy.optimize
def eval_loss(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return f_loss().astype('float64')

def eval_grad(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return np.array(f_grad()).flatten().astype('float64')

# Initialize with a noise image
#generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

x0 = generated_image.get_value().astype('float64')

# Optimize, saving the result periodically
for i in range(50):
    print(i)
    scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
    x0 = generated_image.get_value().astype('float64')
    plt.imsave(time + '/' + str(i) + '.png',deprocess(x0))

# plt.figure(figsize=(12,12))
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.gca().xaxis.set_visible(False)    
#     plt.gca().yaxis.set_visible(False)    
#     plt.imshow(deprocess(xs[i]))
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12,12))
# plt.imshow(deprocess(xs[-1]), interpolation='nearest')
# plt.show()


