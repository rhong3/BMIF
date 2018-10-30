'''
Prepare pd from image tiles
RH 0717
'''

import os
import pandas as pd
import sklearn.utils as sku
import numpy as np
import cv2


pos_path = '../img/pos'
neg_path = '../img/neg'
pos_pattern = '../img/pos/{}'
neg_pattern = '../img/neg/{}'


def image_ids_in(root_dir, ignore=['.DS_Store']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def augmentation():
    poslist = image_ids_in(pos_path)
    neglist = image_ids_in(neg_path)
    for i in poslist:
        posdir = pos_pattern.format(i)
        im = cv2.imread(posdir)
        im = cv2.resize(im, (299, 299))
        cv2.imwrite(pos_pattern.format(i), im)
        im90 = np.rot90(im)
        im90 = cv2.resize(im90, (299, 299))
        cv2.imwrite(pos_pattern.format(i+'90.png'), im90)
        im180 = np.rot90(im,2)
        im180 = cv2.resize(im180, (299, 299))
        cv2.imwrite(pos_pattern.format(i+'180.png'), im180)
        im270 = np.rot90(im,3)
        im270 = cv2.resize(im270, (299, 299))
        cv2.imwrite(pos_pattern.format(i+'270.png'), im270)
        imflp = np.flip(im, 1)
        imflp = cv2.resize(imflp, (299, 299))
        cv2.imwrite(pos_pattern.format(i+'f.png'), imflp)

    for j in neglist:
        negdir = neg_pattern.format(j)
        im = cv2.imread(negdir)
        im = cv2.resize(im, (299, 299))
        cv2.imwrite(neg_pattern.format(j), im)
        im90 = np.rot90(im)
        im90 = cv2.resize(im90, (299, 299))
        cv2.imwrite(neg_pattern.format(j + '90.png'), im90)
        im180 = np.rot90(im, 2)
        im180 = cv2.resize(im180, (299, 299))
        cv2.imwrite(neg_pattern.format(j + '180.png'), im180)
        im270 = np.rot90(im, 3)
        im270 = cv2.resize(im270, (299, 299))
        cv2.imwrite(neg_pattern.format(j + '270.png'), im270)
        imflp = np.flip(im, 1)
        imflp = cv2.resize(imflp, (299, 299))
        cv2.imwrite(neg_pattern.format(j + 'f.png'), imflp)


def samplesum():
    poslist = image_ids_in(pos_path)
    poslist = sorted(poslist)
    neglist = image_ids_in(neg_path)
    neglist = sorted(neglist)
    postenum = int(len(poslist)*0.2)
    negtenum = int(len(neglist)*0.2)
    totpd = []
    pospd = []
    negpd = []
    telist = []
    trlist = []
    postemplist = []
    negtemplist = []

    for i in poslist:
        posdir = pos_pattern.format(i)
        pdp = [posdir, 1]
        pospd.append(pdp)
        totpd.append(pdp)
        postemplist.append(pdp)
        if len(postemplist) == 5:
            if len(telist) < postenum:
                s = np.random.random_sample()
                if s > 0.6:
                    telist.extend(postemplist)
                else:
                    trlist.extend(postemplist)
            else:
                trlist.extend(postemplist)
            postemplist = []

    for j in neglist:
        negdir = neg_pattern.format(j)
        pdn = [negdir, 0]
        negpd.append(pdn)
        totpd.append(pdn)
        negtemplist.append(pdn)
        if len(negtemplist) == 5:
            if len(telist) < negtenum+postenum:
                s = np.random.random_sample()
                if s > 0.6:
                    telist.extend(negtemplist)
                else:
                    trlist.extend(negtemplist)
            else:
                trlist.extend(negtemplist)
            negtemplist = []

    totpd = pd.DataFrame(totpd, columns = ['path', 'label'])
    pospd = pd.DataFrame(pospd, columns = ['path', 'label'])
    negpd = pd.DataFrame(negpd, columns = ['path', 'label'])
    tepd = pd.DataFrame(telist, columns = ['path', 'label'])
    trpd = pd.DataFrame(trlist, columns=['path', 'label'])
    totpd = sku.shuffle(totpd)
    pospd = sku.shuffle(pospd)
    negpd = sku.shuffle(negpd)
    tepd = sku.shuffle(tepd)
    trpd = sku.shuffle(trpd)

    return totpd, pospd, negpd, tepd, trpd

# augmentation()
tot, pos, neg, te, tr = samplesum()

tot.to_csv('../img/tot_sample.csv', index = False)
pos.to_csv('../img/pos_sample.csv', index = False)
neg.to_csv('../img/neg_sample.csv', index = False)
tr.to_csv('../img/tr_sample.csv', index = False)
te.to_csv('../img/te_sample.csv', index = False)