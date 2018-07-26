import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from scipy.ndimage import gaussian_filter
# import skimage.filters as sf
import skimage.color as color
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from tezheng import *
# import Descriptors as des


def SVM(X, y,type):
    classifier = OneVsRestClassifier(SVC(kernel = type,probability = True))
    #classifier = SVC(probability = True)
    classifier.fit(X, y)
    return classifier


if __name__ == '__main__':
    t0=time.time()

    srcfeature = []

    dstfeature = []

    srckind = []

    dstkind = []

    total = 0

    path = 'D:\\ruok\\ds2018\\ds2018'

    for i in os.listdir(path):

        print '123'
        
        if total >= 3:
            break

        path2 = path + '\\' + i

        K = 0
        j = 0
        for each in os.listdir(path2):
            
            img = Image.open(path2 + '\\' + each)
            # print path2 + '\\' + each
            # col =des.autoCorrelogram(array(img.resize((64, 64))))
            
            # print 'col:'+str(j)
            # # j=j+1
            # yimg =  gaussian_filter(img.resize((128, 128)).convert('L'),sigma=20) 
            # histimg = array(img.resize((128,128)).convert('RGB'))
            img =  array(img.resize((128, 128)).convert('L'))
            # img =  array(img)
            # tmp =  calh(histimg)
            # tmp = append(tmp,GLCM(img))
            tmp = SIFT(img)
            tmp =  GLCM(img)
            # tmp = append(tmp,GLCM(img))
            # print img.shape
            tmp =  append(tmp,LBP(img, each))
            # print tmp.shape

            #print 'LBP:' + str(tmp.shape)

            tmp =  append(tmp, HOG(img, each))

            #print 'total:' + str(tmp.shape)
            # print tmp.shape
            if K < 40:
                dstfeature.append(tmp)
                K = K + 1
                dstkind.append(total)
                continue

            srcfeature.append(tmp)

            srckind.append(total)

            # os.system('pause')

        total = total+1

    # time.clock()
    srcfeature = array(srcfeature);dstfeature = array(dstfeature)
    print srcfeature.shape


    # 降维
    pca = decomposition.PCA(n_components=2000, copy=True, whiten=False)

    srcfeature=pca.fit_transform(srcfeature)

    # print time.clock()

    dstfeature=pca.transform(dstfeature)

    # print time.clock()
    print srcfeature.shape

    pre = SVM(srcfeature, srckind,'linear')

    ans = pre.predict_proba(dstfeature)

    w,h = ans.shape

    tmp = 0
    top5 = 0
    top3 = 0
    top1 = 0

    for each in range(w):
        
        mysorted=np.sort(ans[each])
        # print mysorted.shape
        
        # if ans[each][2] <= ans[each][dstkind[tmp]]:
        #     right = right + 1
        if mysorted[-5]<=ans[each][dstkind[tmp]]:
            top5=top5 + 1
        if mysorted[-3]<=ans[each][dstkind[tmp]]:
            top3=top3+1
        if mysorted[-1]<=ans[each][dstkind[tmp]]:
            top1=top1+1
        tmp = tmp + 1

    print 'sigmond:'

    print 'top5:'+str(top5)
    print 'top3:'+str(top3)
    print 'top1:'+str(top1)

    print len(dstfeature)

    # print 'clock:' + str(time.clock())
    print time.clock()-t0

    '''

    pre = SVM(srcfeature, srckind,'poly')

    ans = pre.predict_proba(dstfeature)

    w,h = ans.shape

    tmp = 0
    right = 0

    for each in range(w):
        
        np.sort(ans[each])
        
        if ans[each][4] <= ans[each][dstkind[tmp]]:
            right = right + 1

        tmp = tmp + 1

    print 'poly:'

    print right

    print len(dstfeature)

    print 'clock:' + str(time.clock())

    pre = SVM(srcfeature, srckind,'linear')

    ans = pre.predict_proba(dstfeature)

    w,h = ans.shape

    tmp = 0
    right = 0

    for each in range(w):
        
        np.sort(ans[each])
        
        if ans[each][4] <= ans[each][dstkind[tmp]]:
            right = right + 1

        tmp = tmp + 1

    print 'linear:'

    print right

    print len(dstfeature)

    print 'clock:' + str(time.clock())

    pre = SVM(srcfeature, srckind,'precomputed')

    ans = pre.predict_proba(dstfeature)

    w,h = ans.shape

    tmp = 0
    right = 0

    for each in range(w):
        
        np.sort(ans[each])
        
        if ans[each][4] <= ans[each][dstkind[tmp]]:
            right = right + 1

        tmp = tmp + 1

    print 'precomputed:'

    print right

    print len(dstfeature)

    print 'clock:' + str(time.clock())
    
    '''

            
    


