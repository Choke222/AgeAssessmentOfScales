# coding: utf-8
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
#マスク画像を作成するライブラリ

#呼び出し用関数
#返り値はマスク画像

def create(img,l,ksize = 20,dir_path = ''):
    #マスクを作るための前処理
    mask_item = make_mask_item(img,int(l/24),dir_path)

    ###マスク作成###
    mask = mask_labeling(mask_item.astype(np.uint8))

    return mask

#マスクを作るための前処理
def make_mask_item(img,ksize = 20,dir_path = ''):
    img_dst = copy.copy(img)
    #img_dst = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,ksize,0)
    #cv2.imwrite('mask.bmp', img_dst)
    img_dst = cv2.blur(img_dst, ksize=(ksize,ksize))
    cv2.imwrite(dir_path + '/mask_item_blur' + '.bmp', img_dst)
    # 大津の手法
    ret, img_dst = cv2.threshold(img_dst, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite(dir_path + '/mask_item_threshold' + '.bmp', img_dst)


    return img_dst

#マスクを作るメイン処理
def mask_labeling(img):
    dst_image = copy.copy(img)
    #マスク以外の白を削除
    nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(img)
    if nLabels > 1:
        #print(data[1:,4])
        #max_indexはマスク、それ以外がノイズ
        max_index = np.argmax(data[1:,4]) + 1
        '''x, y, w, h, size = data[max_index]
        x_ = x + w
        y_ = y + h'''

        #delite_index = []

        for index in range(1,nLabels):
            if index == max_index:
                continue
            dst_image = np.where(labelImages == index, 0, dst_image)
            '''x0, y0, w, h, size = data[index]
            x1 = x0 + w
            y1 = y0 + w

            count = 0
            count = count + inSquare(x, x_, y, y_ ,x0 ,y0)
            count = count + inSquare(x, x_, y, y_ ,x0 ,y1)
            count = count + inSquare(x, x_, y, y_ ,x1 ,y0)
            count = count + inSquare(x, x_, y, y_ ,x1 ,y1)
            if count < 2:
                dst_image = np.where(labelImages == index, 0, dst_image)'''
    else:
        #nLabelsが１の場合マスク以外ない
        dst_image = np.where(dst_image == 0,0,1)
        print('mask only')
    #マスクの黒い穴を埋める操作
    nega_image = cv2.bitwise_not(dst_image)
    nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(nega_image)
    if nLabels > 1:
        #max_indexは背景それ以外がマスク内の黒抜け
        max_index = np.argmax(data[1:,4]) + 1
        for index in range(1,nLabels):
            if index == max_index:
                continue
            dst_image = np.where(labelImages == index, 255, dst_image)
    dst_image = np.where(dst_image == 0,0,1)
    return dst_image

#ラベル同士の位置関係を判定
def inSquare(x, x_, y, y_ ,X ,Y):
    if x < X and X < x_:
        if y < Y and Y < y_:
            return 1
    return 0

