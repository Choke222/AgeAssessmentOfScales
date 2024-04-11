#import
from turtle import distance
import cv2
import numpy as np
import glob
import copy
from datetime import datetime
import os
from tqdm import tqdm
import statistics
import math
from scipy import interpolate
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
#import:Neural用自作ライブラリ
import Img_Proc
import neural_age

#import:自作ライブラリ系
import binarization
import center
import normalization
import age
import mask
import AccCalc_not7sai

def get_args():
    parser = argparse.ArgumentParser(description='Use dataset')
    parser.add_argument('--dataset', '-d', type=str, default=False, help='Use restzone dataset')
    parser.add_argument('--raw-dataset', '-r', type=str, default=False, help='Use raw dataset')
    parser.add_argument('--percent', '-p', type=int, default=10, help='Fake Annual Zone percentage')
    return parser.parse_args()

#画像読み込み
args = get_args()

dataset = args.dataset
percent = args.percent
raw_dataset=args.raw_dataset
if raw_dataset == False:
    input_dir="../dataset/image/raw/"
else:
    input_dir="../dataset/image/"+raw_dataset+"/raw/"

estimate_dir="../dataset/image/"+dataset+"/restzone/"

ext="*.jpg"

#保存用ディレクトリの準備
print(input_dir + ext)
image_path_list = glob.glob(input_dir + ext)
result_output_dir = "../dataset/result/"+dataset+"/"

#中心判定結果
centerx = None
centery = None
correct_count=[0,0,0,0,0]
total = [0,0,0,0,0]
sumt2=0
sumt4=0
sumt6=0
sumt8=0
sumt10=0
flop=0

true_age=[]
pred_age=[]

for image_path in tqdm(image_path_list):
    # print("確認")
    #拡張子なしのファイル名を取得(保存名に利用)
    base_name=os.path.splitext(os.path.basename(image_path))[0]

    #元画像の読み込み
    original_Image=cv2.imread(image_path,1)

    #検出画像の読み込み
    print(estimate_dir+base_name+".bmp")
    estimate_Image=cv2.imread(estimate_dir+base_name+".jpg",1)
    
    ######前処理########
    basic_Image,l=Img_Proc.Pre_proc(original_Image)

    #結果保存用ディレクトリ
    os.makedirs(result_output_dir+base_name,exist_ok=True)
    each_output_dir=result_output_dir+base_name

    #マスク処理
    mask_Image = mask.create(basic_Image,l,int(l/25),dir_path = each_output_dir)
    cv2.imwrite(each_output_dir+'/mask_labeling_' + base_name + '.bmp', mask_Image * 255)
    cv2.imwrite(each_output_dir+'/masking_image_' + base_name + '.bmp', mask_Image * basic_Image)
    print('maked mask.....')

    #中心判定
    center_position = center.create(basic_Image, each_output_dir)
    print('centersumt3:',center_position)

    #########################
    ###元画像の極座標展開####
    #########################
    #np.set_printoptions(threshold=np.inf)
    
    y_size=1800
    x_size=900
    #元画像:中心に合わせて極座標展開
    flags = cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
    linear_polar_raw_image=cv2.warpPolar(original_Image,(x_size,y_size),(center_position[0],center_position[1]),l,flags)
    cv2.imwrite(each_output_dir+'/linear_polar_raw_' + base_name + '.bmp', linear_polar_raw_image)

    #画像の左半分を切り抜き
    #linear_polar_image=linear_polar_image[:,0:int(x_size/2)]
    #cv2.imwrite(each_output_dir+"/extend_image"+base_name+".bmp",linear_polar_image)
    
    ###########################
    ###マスク画像の極座標展開##
    ###########################
    mask_Image = mask_Image * 255
    linear_polar_mask = cv2.warpPolar(mask_Image,(x_size,y_size),(center_position[0],center_position[1]),l,flags)
    print(each_output_dir+"/extend_mask_image_"+base_name+".bmp")
    cv2.imwrite(each_output_dir+"/extend_mask_image_"+base_name+".bmp",linear_polar_mask)

    #輪郭を数px増やす
    linear_polar_mask_uint8 = np.array(linear_polar_mask, dtype=np.uint8)
    ContoursMask,hierarchy = cv2.findContours(linear_polar_mask_uint8,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    linear_polar_mask = cv2.drawContours(linear_polar_mask_uint8,ContoursMask, -1, (0,255,0), 4)
    #cv2.imwrite(each_output_dir+"/extend_mask_image_Contour"+base_name+".bmp",ResultContour)
    #最大の横幅のサイズを算出
    MaxAreaSize,MaxIndex = neural_age.WidthSize_Calc(linear_polar_mask)

    ###########################
    ###検出画像の極座標展開####
    ###########################
    linear_polar_image=cv2.warpPolar(estimate_Image,(x_size,y_size),(center_position[0],center_position[1]),l,flags)
    cv2.imwrite(each_output_dir+"/extend_label_image_"+base_name+".bmp",linear_polar_image)
    
    if linear_polar_image.ndim != 2:
        gray_linear_Image = cv2.cvtColor(linear_polar_image,cv2.COLOR_BGR2GRAY)
    else:
        gray_linear_Image = linear_polar_image
        np.set_printoptions(threshold=np.inf)
        linear_polar_image = cv2.cvtColor(linear_polar_image,cv2.COLOR_GRAY2BGR)

    #linear_polar_image=linear_polar_image[:,0:int(x_size/2)]

    #線の端を点として抽出
    thre=20
    contours,hierarchy=cv2.findContours(gray_linear_Image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: cv2.contourArea(x) > thre, contours))
    save_file=each_output_dir+"/extend_image"+base_name+"_rest_zone"+".txt"
    
    cv2.drawContours(linear_polar_image,contours, -1, color=(0, 0, 255), thickness=5)
    cv2.imwrite(each_output_dir+"/extend_image_"+base_name+"_contours"+".bmp",linear_polar_image)
    y_max=-1
    y_min=20000

    #領域の上端点を取得
    tail_list = neural_age.TailPixel_Counter(contours)
    
    #circle_image=linear_polar_image
    #上端点の結果を描画
    #for AreaCnt in range(len(tail_list)):
        #cv2.circle(circle_image, (tail_list[AreaCnt][0],tail_list[AreaCnt][1]), 15, (255,255,0), thickness=2, lineType=cv2.LINE_8)

    #TODO:関数化、下端を受け取る単一処理に変更
    #領域内の最も先頭にあるpixelを取得
    
    for AreaLabel in range(len(contours)):
        y_min=20000
        for PixelLabel in range(len(contours[AreaLabel])):
            if y_min >= contours[AreaLabel][PixelLabel][0][1]:
                head_pixel = contours[AreaLabel][PixelLabel][0]
                y_min = contours[AreaLabel][PixelLabel][0][1]
        head_pixel=head_pixel.tolist()
        
        #自身以外のpixelとの距離が最も短い領域との補間を行う
        distance_min=20000
        min_label=-1

        DistanceY=150
        DistanceX=60
        Degree=20

        #TODO:関数化、補間にまつわる距離や角度算出部分
        for k in range(len(tail_list)):
            theta=neural_age.getRadian(tail_list[k],head_pixel)
            #print("Theta:"+str(theta))
            # a = tail_list[k][0]-head_pixel[0]
            # b = tail_list[k][1]-head_pixel[1]
            # d = math.sqrt(math.pow(a,2)+math.pow(b,2))
            if AreaLabel!=k and \
               tail_list[k][0]-head_pixel[0] < DistanceY and \
               abs(tail_list[k][1]-head_pixel[1]) < DistanceX and \
               abs(theta) < Degree:
                dist = neural_age.euclid(tail_list[k],head_pixel)
                if dist <= distance_min:
                    distance_min=dist
                    min_label=k

        #下端の描画
        #cv2.circle(circle_image, (head_pixel[0],head_pixel[1]), 15, (255,255,0), thickness=2, lineType=cv2.LINE_8)

        pt_x=[]
        pt_y=[]
        #TODO:関数化、補間の要素構築
        if min_label != -1:
            # list -> numpy
            area_a = np.array(contours[min_label])
            area_b = np.array(contours[AreaLabel])

            # squeezeによって要素数1の次元を削除
            area_a = np.squeeze(area_a)
            area_b = np.squeeze(area_b)

            arg_a=area_a.argsort(axis=0)

            #y軸のpxを基準にソート
            area_a_sorted=area_a[np.argsort(area_a[:, 1])]
            area_b_sorted=area_b[np.argsort(area_b[:, 1])]


            # x,y要素に分割
            # for lb in range(len(area_a)):
            #     pt_x.append(area_a[lb][0])
            #     pt_y.append(area_a[lb][1])

            # for lb in range(len(area_b)):
            #     pt_x.append(area_b[lb][0])
            #     pt_y.append(area_b[lb][1])

            # 点の両端のみを配列に代入
            pt_x.append(tail_list[min_label][0])
            pt_y.append(tail_list[min_label][1])

            pt_x.append(head_pixel[0])
            pt_y.append(head_pixel[1])

            # pt_x=np.array(pt_x)
            # pt_y=np.array(pt_y)

            # spline補間によって曲線を算出
            sp_x,sp_y = neural_age.spline3(pt_x,pt_y,100,1)
            
            #TODO:関数化、補間の後処理
            # 再結合
            pt_x=np.array(sp_x)
            pt_y=np.array(sp_y)

            con = np.stack([pt_x,pt_y],1)
        
            #画像上に描画
            if abs(sp_x[0]-sp_x[-1])<= DistanceX:
                for cnt in range(len(con)):
                    cv2.circle(linear_polar_image, (int(con[cnt][0]),int(con[cnt][1])), 0, (0,0,255), thickness=5, lineType=cv2.LINE_8)
            
    cv2.imwrite(each_output_dir+"/extend_image_"+base_name+"_before_hokan_red"+".bmp",linear_polar_image)

    #両端の結果画像の表示
    #cv2.imwrite(each_output_dir+"/extend_image"+base_name+"_circle_blue"+".bmp",circle_image)

    #白い部分を黒くする
    linear_polar_image = Img_Proc.Post_Proc(linear_polar_image)
    linear_polar_image = cv2.cvtColor(linear_polar_image, cv2.COLOR_RGB2GRAY)
    
    #マスクと休止帯画像を重ねる
    linear_polar_image = linear_polar_image * (linear_polar_mask)
    linear_polar_image = np.clip(linear_polar_image,0,255)
    linear_polar_image = np.array(linear_polar_image, dtype=np.uint8)

    #保存
    #cv2.imwrite(each_output_dir+"/extend_image_"+base_name+"_circle"+".bmp",gray_linear_Image)
    cv2.imwrite(each_output_dir+"/extend_image_Mask_Overlay"+".bmp",linear_polar_image)
    t4=time.time()
    # linear_polar_image = linear_polar_image
    contours,hierarchy=cv2.findContours(linear_polar_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: cv2.contourArea(x) > 50, contours))
    linear_polar_mask = cv2.drawContours(linear_polar_image,contours, -1, (255,255,255), 2)
    cv2.imwrite(each_output_dir+"/extend_image_hokango"+".bmp",linear_polar_image)
    # 逆変換した画像を生成
    flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    linear_polar_inverse_image = cv2.warpPolar(linear_polar_image, (2080,1600),(center_position[0],center_position[1]),l, flags)
    cv2.imwrite(each_output_dir + "/inverse_image_" + base_name + ".bmp",linear_polar_inverse_image)

    #休止帯カウント  
   
    count = [0] * y_size
    flag = [0] * y_size
    flag_default = [0] * y_size

    #TODO:関数化年齢判定の取りまとめ部分
    #全体の何％を休止帯とみなすか決定
    
    FakeAreaPercent = percent
    FakeAreaLabel = int(MaxAreaSize * FakeAreaPercent / 100)
    # for i in range(0,FakeAreaLabel):
    #      flag_default[i] = 1

    #休止帯の計数
    #外側のループ:輪郭抽出した複数の領域から任意の領域を参照
    #内側のループ:任意の領域について輪郭のピクセルを参照
    
    for AreaLabel in range(len(contours)):
        
        for PixelLabel in range(len(contours[AreaLabel])):
            #1度計数に用いた領域は使わない
            #x方向の座標が一定以上の場合のみ年齢にカウントする
            if flag[contours[AreaLabel][PixelLabel][0][1]]==0 and \
                FakeAreaLabel <= contours[AreaLabel][PixelLabel][0][0]:
                count[contours[AreaLabel][PixelLabel][0][1]]+=1
                flag[contours[AreaLabel][PixelLabel][0][1]]+=1
        flag = [0] * y_size

    print(base_name)
    length=np.arange(0, 1800)
    plt.bar(length,count)
    plt.xlabel("y")
    plt.ylabel("Annual ring count")
    plt.savefig(each_output_dir + "/age_hist.png")

    f = open(each_output_dir + "/age_hist.txt", 'w')
    f.write(str(count))
    f.close()

    count=[s for s in count if s > 2 and s < 7]
    
    answer_age = int(base_name[-1])
    total[answer_age-3] += 1

    true_age.append(int(base_name[-1]))
    pred_age.append(max(count,default=3))

    print("Pred:"+str(max(count,default=3)))
    if len(count) != 0 and answer_age == max(count):
        #print("Pred:"+str(max(count,default=3)))
        correct_count[answer_age-3]+=1
    print("=====================")
    #exit()