import math
from scipy import interpolate
import numpy as np
import cv2

def WidthSize_Calc(PolarImage : np):
    """鱗の横幅サイズを算出する関数

    Augs: 
        PolarImage(np) : 極座標展開後のマスク画像 

    Returns: 
        int : 鱗領域部分の最大サイズ
        int : 鱗領域部分における一番右側のx座標
    """
    
    height = PolarImage.shape[0]
    width = PolarImage.shape[1]
    MaxArea = 0
    MaxIndex = 0
    for h in range(height):
        # 白い領域の数を計数
        ScaleAreaCount = np.count_nonzero(PolarImage[h] == 255)
        # 一番面積が大きかった時，鱗領域部分の一番右の座標を取得
        if ScaleAreaCount > MaxArea:
            MaxArea = ScaleAreaCount
            # 配列を左右反転
            WidthFlip = np.flip(PolarImage[h])
            # 最大値のうち一番左側のインデックスを取得
            FlipIndex = np.argmax(WidthFlip)
            # 反転したインデックスを戻すことで一番右側のインデックスを取得
            MaxIndex = width - 1 -FlipIndex

    # debug用
    # print("width  "+str(width))
    # print("FlipIndex  "+str(FlipIndex))
    # print("MaxIndex  "+str(MaxIndex))
    # print("MaxArea  "+str(MaxArea))
    return MaxArea,MaxIndex

def euclid(a : list  , b : list ):
    """2点のユークリッド距離を求める関数

    Augs: 
        a,b(list) : a,bの2点の2次元座標

    Returns: 
        int : 2点のユークリッド距離
    """

    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))

def spline3(x,y,point,deg):
    """spline補間を行う関数

    Augs: 
        x(list) : x座標のデータ
        y(list) : y座標のデータ
        point   : 点の分割数
        deg     : 次数

    Returns: 
        int : 補間結果の点におけるx,y座標
    """
    tck,u = interpolate.splprep([x,y],k=deg,s=0)
    u = np.linspace(0,1,num=point,endpoint=True) 
    spline = interpolate.splev(u,tck)
    return spline[0],spline[1]

def getRadian(a,b):
    """角度を算出する

    Augs:
        a(list) : 点aの座標データ[x,y]
        b(list) : 点bの座標データ[x,y]
    
    Returns: 
        theta(float) : 角度
    """
    #ベクトルを算出
    vec1=[a[0]-a[0],abs(b[1]-a[1])]
    vec2=[a[0]-b[0],b[1]-a[1]]

    #内積、ノルムを算出
    inner=np.inner(vec1,vec2)
    absvec1=np.linalg.norm(vec1)
    absvec2=np.linalg.norm(vec2)

    #cosθを算出しθへと変換
    cos_theta=inner/(absvec1*absvec2)
    theta=math.degrees(math.acos(cos_theta))
    return theta

def TailPixel_Counter(RestzoneArea):
    """各領域の末尾のピクセルを記録する
    -> 各領域の末尾→領域内で最もY方向のラベルが大きい

    Augs:
        RestzoneArea : cv2.findContoursによって領域検出を行った結果
    Returns: 
        TailList(List) : 入力した各領域の末尾のピクセル
    """
    TailList=[]
    #各領域を順に探索
    for AreaLabel in range(len(RestzoneArea)):
        y_max=-1
        #ある1つの領域のピクセルを探索
        for PicelLabel in range(len(RestzoneArea[AreaLabel])):
            #領域内で最もY方向のラベルが大きかったら更新
            if y_max <= RestzoneArea[AreaLabel][PicelLabel][0][1]:
                TailPicel = RestzoneArea[AreaLabel][PicelLabel][0]
                y_max = RestzoneArea[AreaLabel][PicelLabel][0][1]
        #領域内で最もY方向のラベルが大きい要素をリストに追加
        TailList.append(TailPicel)
    return TailList

def HeadPixel_Counter(RestzoneArea):
    """各領域の先頭のピクセルを記録する
    -> 各領域の先頭→領域内でY方向のラベルが小さい

    Augs:
        RestzoneArea : cv2.findContoursによって領域検出を行った結果
    Returns: 
        TailList(List) : 入力した各領域の先頭のピクセル
    """
    HeadList=[]

    #各領域を順に探索
    for AreaLabel in range(len(RestzoneArea)):
        #ある1つの領域のピクセルを探索
        y_min=20000
        for PixelLabel in range(len(RestzoneArea[AreaLabel])):
            if y_min >= RestzoneArea[AreaLabel][PixelLabel][0][1]:
                head_pixel = RestzoneArea[AreaLabel][PixelLabel][0]
                y_min = RestzoneArea[AreaLabel][PixelLabel][0][1]
        HeadList.append(head_pixel)
    return HeadList

def AgeEstimate(RestzoneArea,y_size):

    """ 補間後の結果から年齢判定を行う
    Augs:
        RestzoneArea : 補間後に領域抽出で検出した休止帯データ
        y_size : 画像の縦方向のサイズ

    Returns: 
        basic_Image(ndarray) : 入力画像をグレー画像にした後、ネガポジ変換を行った結果
    """

    flag = [0] * y_size
    for AreaLabel in range(len(RestzoneArea)):
        for PixelLabel in range(len(RestzoneArea[AreaLabel])):
            if flag[RestzoneArea[AreaLabel][PixelLabel][0][1]]==0:
                count[RestzoneArea[AreaLabel][PixelLabel][0][1]]+=1
                flag[RestzoneArea[AreaLabel][PixelLabel][0][1]]+=1
        flag = [0] * y_size
    
    #画像要素から休止帯の通過した数が0,1であるものを削除
    count=[s for s in count if s != 0]
    count=[s for s in count if s != 1]
