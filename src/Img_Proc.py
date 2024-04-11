import math
from scipy import interpolate
import numpy as np
import cv2 


def Pre_proc(original_image):
    """画像の前処理を行う

    Augs:
        original_image : cv2で読み込んだnumpy形式のデータ
    Returns: 
        basic_Image(ndarray) : 入力画像をグレー画像にした後、ネガポジ変換を行った結果
        l : 画像の縦横の長辺
    """
    #グレースケール変換
    gray_Image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    #ネガポジ変換
    basic_Image = cv2.bitwise_not(gray_Image)
    #画像サイズの取得
    h,w=basic_Image.shape
    #長辺のサイズを取得
    l = w if w > h else h
    return basic_Image,l

def Post_Proc(linear_polar_image):
    """補間・領域検出に用いた部分のみを画像中に示す関数
    ->PSPNetで検出した部分だが、補間には用いなかった領域を削除する
    ->補間に利用・生成された領域は赤、PSPNetによって検出された領域は白

    Augs:
        linear_polar_image : 極座標展開済み画像に対して補間を行った結果の画像
    Returns: 
        OutputImage : 補間に用いた領域、補間で出した線のみが描画された画像
    """

    black=np.array([0,0,0])
    #print(linear_polar_image)
    #画像中の各ピクセルを探索
    for X_Pixel in range(len(linear_polar_image)):
        for Y_Pixel in range(len(linear_polar_image[X_Pixel])):
            #RGBの各要素が白に近い場合、黒に変換する
            if linear_polar_image[X_Pixel][Y_Pixel][0]>=245 and \
               linear_polar_image[X_Pixel][Y_Pixel][1]>=245 and \
               linear_polar_image[X_Pixel][Y_Pixel][2]>=245:
               linear_polar_image[X_Pixel][Y_Pixel]=black
    OutputImage = linear_polar_image
    return OutputImage

def Pile_Image(ScaleImage,MaskImage,RestzoneImage,OutputDir):
    """鱗画像に休止帯画像，マスク画像を重ねる関数

    Augs: 
        ScaleImage(np) : 極座標展開後の元画像
        MaskImage(np) : 極座標展開後のマスク画像
        RestzoneImage(np) : 極座標展開後の休止帯画像

    Returns: なし(画像の出力のみを行うため)
    """
    #マスクと休止帯画像を重ねる
    RestzoneImage = RestzoneImage * (MaskImage)
    RestzoneImage = np.clip(RestzoneImage,0,255)
    RestzoneImage = np.array(RestzoneImage, dtype=np.uint8)
    cv2.imwrite(OutputDir+"/extend_image_test"+".bmp",RestzoneImage)

    #元画像へ休止帯画像を重ねる
    
    #休止帯画像を白黒反転
    RestzoneImage_Reverse = 255 - RestzoneImage
    cv2.imwrite(OutputDir+"/extend_RestzoneImage_Reverse"+".bmp",RestzoneImage_Reverse)

    #休止帯画像をRGB変換

    #白い部分を透過
    transparence = (0,0,0)
    #result = np.where(front==transparence, back, front)
    #元画像に重ねる




