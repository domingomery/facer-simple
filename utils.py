# The idea of this code is to verify is the subject present in the query image 
# is present in the session image.
#
# see example in facer_demo.py
#
# D. Mery, UC, October, 2018
# http://dmery.ing.puc.cl

import scipy.io
import numpy as np
import face_recognition
from PIL import Image
import cv2
from skimage.transform import resize
from sklearn.metrics.pairwise import euclidean_distances
from keras.models import load_model

def imread(filename):
    image = face_recognition.load_image_file(filename)
    return image

def imreadx(filename,show_img):
    image = imread(filename)
    if show_img == 1:
        imshow(image)
    return image

def load_fr_model(fr_method):
    if  fr_method == 2:
        model_path = '/Users/domingomery/Python/keras-facenet/model/keras/facenet_keras.h5'
        fr_model = load_model(model_path)
    else:
        fr_model = 1
    return fr_model

def imshow(image):
    pil_image = Image.fromarray(image)
    pil_image.show()

def uninormalize(vector):
    norm=np.linalg.norm(vector)
    if norm==0:
        norm=np.finfo(vector.dtype).eps
    return vector/norm

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def face_descriptor(image,method,model,uninorm):
    if method == 0: # dlib-original does not work always :(
        desc    = face_recognition.face_encodings(image)[0]
    elif method == 1:  
        fl = None #dlib without face detection
        x    = face_recognition.face_encodings(image,fl)
        if len(x)==0:
            fl = [[0,0,len(image)-1,len(image[0])-1]]
            desc = face_recognition.face_encodings(image,fl)[0]
        else:
            desc = x[0]
    elif method == 2: #facenet
        image_size = 160
        img1 = prewhiten(image)
        img2 = resize(img1, (image_size, image_size), mode='reflect')
        img3 = img2.reshape(1,img2.shape[0],img2.shape[1],img2.shape[2])
        px = model.predict_on_batch(img3[0:1,0:image_size,0:image_size,0:3])
        desc = l2_normalize(px)
        desc = desc.reshape(desc.shape[1])
    if uninorm == 1:
        desc = uninormalize(desc)
    return desc

def session_descriptors(S,faces,fr_method,fr_model,uninorm):
    n = len(faces)
    D = [0] * n
    i = 0
    for i in range(n):
        top, right, bottom, left = faces[i]
        face_i = S[top:bottom, left:right]
        D[i] = face_descriptor(face_i,fr_method,fr_model,uninorm)
        i = i+1
    return D

def face_scores(D1,d2,method,print_scr):
    if method == 0: # cosine similarity
        scr = np.matmul(D1,d2)
    elif method == 1: # euclidean distance
        d2 = d2.reshape(1,d2.shape[0])
        scr = euclidean_distances(D1,d2)
    if print_scr==1:
        print("[facer] : scores ")
        print(scr)
    return scr

def face_detection(image,method):
    if method == 0:
        faces  = face_recognition.face_locations(image)
    elif method == 1:
        faces  = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    else:
        print("[facer] : error - face detection method " +str(method)
              + " not defined")
    return faces


def is_face(scr,theta,sc_method):
    if sc_method == 0: # cosine similarity
        i = np.argmax(scr)
        sc = scr[i]
        if sc<theta:
            i=-1
    else:
        i = np.argmin(scr)
        sc = scr[i]
        if sc>theta:
            i=-1
    return i

def show_face(S,face,i,show_img):
    # face = faces[i]
    top, right, bottom, left = face
    face_i = S[top:bottom, left:right]
    if show_img==1:
        imshow(face_i)
    
def print_definitions(fd_method,fr_method,sc_method,uninorm,theta):
    fd_str = ['HOG','CNN','not-defined','not-defined','not-defined','not-defined']
    fr_str = ['Dlib','Dlib+','FaceNet','not-defined','not-defined','not-defined']
    sc_str = ['CosSim','Euclidean']
    print("[facer] : fd = "+fd_str[fd_method]+", "
                  + "fr = "+fr_str[fr_method]+", "
                  + "sc = "+sc_str[sc_method]
                  + "(uninorm="+str(uninorm)+"), "
                  + "th = "+str(theta))
    return


