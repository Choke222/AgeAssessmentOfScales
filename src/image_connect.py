from email.mime import image
import glob
import sys,os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import cv2
from natsort import natsorted
import argparse

def cv2pil(image):
    ''' type:OpenCV -> type:PIL '''
    new_image = image.copy()
    if new_image.ndim == 2:  # Mono
        pass
    elif new_image.shape[2] == 3:  # Color
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # transmission
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def change_backcolor(path,uid,num,key,tail):
    image_path_tm=uid+"-"+str(num)+"-"+key+tail
    img_tm = cv2.imread(path+image_path_tm)
    if ~(np.all(img_tm == 0)):
        for i in range(img_tm.shape[0]):
            for j in range(img_tm.shape[1]):
                if (img_tm[i][j][0] < 15) and (img_tm[i][j][1] < 15) and (img_tm[i][j][2] < 15):
                    img_tm[i][j]=[255,255,255]
                else:
                    img_tm[i][j]=[0,0,0]
    
    img=cv2pil(img_tm)
    return img

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def join(path):
    file_path=path
    # save_dir="../dataset/restzone/"

    tail=".jpg"
    files=glob.glob(file_path+"*"+tail)
    files=natsorted(files)
    vertical=10
    width=13
    key=file_path.split("/")[-2]
    print(files[0])
    tmp=files[0].split("/")
    uid=tmp[-1].split("-")[0]

    image_list = [[0] * width for i in range(vertical)]

	#Resize and merge images horizontally and vertically
    for w in range(width):
        for h in range(vertical):
            num=h+w*10+1
            image_path=uid+"-"+str(num)+"-"+key+tail
            print(file_path+image_path)
            img = Image.open(file_path+image_path)
            img=change_backcolor(file_path,uid,num,key,tail)
            img = img.resize((160,160))
            img=np.asarray(img)
            image_list[h][w]=img

    im_tail=concat_tile(image_list)

    os.makedirs(save_dir,exist_ok=True)
    cv2.imwrite(save_dir+"/"+key+tail,im_tail)

if __name__ =="__main__":
    args = sys.argv
    res_file=args[1]
    save_dir=args[2]
    print(res_file,save_dir)

    files = os.listdir(res_file+"/")
    files_dir = [f for f in files if os.path.isdir(os.path.join(res_file, f))]
    join(res_file+"/")