import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))
import matplotlib.pyplot as plt
import cv2
import numpy as np

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def get_aligned_face(image_path):
    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB) #ada2 ada3
    img = Image.fromarray(img)
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0]
    except Exception as e:
        face = None
        box=None

    if(face != None):
        bboxes=list(np.reshape(bboxes,5))
        bboxes.pop()
        bboxes=[int(x) for x in bboxes ]
        a=bboxes.pop(0)
        bboxes.append(a)
        bboxes=[bboxes]
        box = [tuple(x) for x in bboxes]
    return face,box
