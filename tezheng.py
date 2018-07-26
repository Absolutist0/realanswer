# -*- coding: utf-8 -*-
from PIL import Image
from numpy import *
from scipy.ndimage import filters
import scipy
import matplotlib.pyplot as plt
import skimage.color as color
import skimage.feature as feature
from skimage.feature import greycomatrix
import time
import os
import cv2
# import pandas as pd
from sklearn import decomposition

set_printoptions(threshold='nan')  # 输出全部数据


def PIF(img):
    # f='D:\\ruok\\pic.jpg'
    # im=Image.open(f)
    # im=im.resize((64, 64),Image.ANTIALIAS)    
    # im=img.convert('L')
    # pix=im.load()
    act=img
    imm=img
    # print imm
        # print im.size
    k=3  
    value=[]
    emt1=zeros((k,64))
    emt2=zeros((64+2*k,k))
    # print emt1.shape
    # print emt2.shape
    imm=vstack((emt1,imm,emt1))
    # print imm.shape
    imm=hstack((emt2,imm,emt2))
    for x in range(k,64+k):
        for y in range(k,64+k):
            tmp=array(imm[x-k:x+k+1,y-k:y+k+1])
            tmp=tmp.flatten()
            temp=abs(tmp-imm[x][y])
            global kpsum
            kpsum=temp.sum()/49
            # global value
            value.append(kpsum)
    value=array(value)
    meanP=(value.sum())/(64*64)
    # print meanP
    pif=zeros((64,64))
    pif2=zeros((64,64))
    pif3=zeros((64,64))
    pif4=zeros((64,64))
    for x in range(k,64+k):
        for y in range(k,64+k):
            tmp=array(imm[x-k:x+k+1,y-k:y+k+1])
            tmp=tmp.flatten()
            a=0
            temp=abs(tmp-imm[x][y]) > meanP
            a=temp.sum()
            # print a
            pif[x-k][y-k]=a/49.0
    # return pif
        # print '1'
    pif2=vstack((emt1,pif,emt1))
    pif2=hstack((emt2,pif2,emt2))
    for x in range(k,64+k):
        for y in range(k,64+k):
            tmp=array(pif2[x-k:x+k+1,y-k:y+k+1])
            tmp=tmp.flatten()
            temp=tmp >= 0.5
            a=temp.sum()
            pif2[x-k][y-k]=a/49.0
    for x in range(k,64+k):
        for y in range(k,64+k): 
            if pif[x-k][y-k]>=0.5 and pif2[x-k][y-k]>=0.5:
                pif3[x-k][y-k]=1
            else:
                pif3[x-k][y-k]=0       
    # print pif

    pif4=pif3*act
    return pif4
def des2feature(des,num_words,centures):
    '''
    des:单幅图像的SIFT特征描述
    num_words:视觉单词数/聚类中心数
    centures:聚类中心坐标   num_words*128
    return: feature vector 1*num_words
    '''
    img_feature_vec=np.zeros((1,num_words),'float32')
    for i in range(des.shape[0]):
        feature_k_rows=np.ones((num_words,128),'float32')
        feature=des[i]
        feature_k_rows=feature_k_rows*feature
        feature_k_rows=np.sum((feature_k_rows-centures)**2,1)
        index=np.argmax(feature_k_rows)
        img_feature_vec[0][index]+=1
    return img_feature_vec    
def calh(img):
    chist=[]
    for i in range(3):
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        chist.append(hist)
    # chist = (array(chist)).flatten()
    # chist = array(hist)/ float(img.shape[0] * img.shape[1])
    chist = array(chist).flatten()/ float(img.shape[0] * img.shape[1])
    # print chist.shape
    return chist
def GLCM(img):
    img=img // 8
    glcm= greycomatrix(img,distances=[2,8,16],angles=[0,pi/4,pi/2,pi*(3/4)],levels=32,symmetric=True,normed=True)
    glcm= glcm.flatten()
    # print glcm.shape
    return glcm
def SIFT(img_paths):
    # img = img//8
    # print img.shape
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
    kps,features=sift.detectAndCompute(img,None)
    
    # kp, des = sift.detectAndCompute(img,None)
    # features = features.flatten()
    # print kps

    # print array(des).shape
    # os.system('pause')
    return features

def LBP(img, name):
    img=img // 8
    lbp = feature.local_binary_pattern(img,24,3,'uniform')

    max_bins = int(lbp.max() + 1)

    nlbp, _ = histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    # nlbp = (lbp-img.min()) / float(img.max()-img.min())
    # print nlbp
    return nlbp


def HOG(img, name):
    # img=PIF(img)
    img=img // 8
    f = feature.hog(img,pixels_per_cell =(8,8), cells_per_block=(2, 2), block_norm='L1')

    # f = [round(each, 6) for each in f]

    return f


if __name__ == '__main__':

    fp = 'D:\\ruok\\ds2018\\ds2018\\bear\\1da4cf18-8744-11e8-8093-1c39473dbc6d.jpg.jpg'
    img = Image.open(fp)
    img=  img.resize((64, 64)).convert('L')
    img = array(img)
    # calh(img)
    SIFT(img)


