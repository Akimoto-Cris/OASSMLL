import cv2
import random
import numpy as np


def imshow(image, delay=0, mode='RGB', title='show'):
    if mode == 'RGB':
        demo_image = image[..., ::-1]
    else:
        demo_image = image

    cv2.imshow(title, demo_image)
    if delay >= 0:
        cv2.waitKey(delay)

def transpose(image):
    return image.transpose((1, 2, 0))

def denormalize(image, mean=None, std=None, dtype=np.uint8, tp=True):
    if tp:
        image = transpose(image)
        
    if mean is not None:
        image = (image * std) + mean
    
    if dtype == np.uint8:
        image *= 255.
        return image.astype(np.uint8)
    else:
        return image

def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w, c = shape
        cam = cv2.resize(cam, (w, h))
    cam = cv2.applyColorMap(cam, mode)
    return cam

def decode_from_colormap(data, colors):
    ignore = (data == 255).astype(np.int32)

    mask = 1 - ignore
    data *= mask

    h, w = data.shape
    image = colors[data.reshape((h * w))].reshape((h, w, 3))

    ignore = np.concatenate([ignore[..., np.newaxis], ignore[..., np.newaxis], ignore[..., np.newaxis]], axis=-1)
    image[ignore.astype(np.bool)] = 255
    return image

def normalize(cam, epsilon=1e-5):
    cam = np.maximum(cam, 0)
    max_value = np.max(cam, axis=(0, 1), keepdims=True)
    return np.maximum(cam - epsilon, 0) / (max_value + epsilon)




