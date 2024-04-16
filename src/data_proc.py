import glob
import sys,os
from PIL import Image
import numpy as np
import cv2
from natsort import natsorted

def image_cut(file_name,h_split,w_split,count):

    img=Image.open(file_name)
    box=(0,0,img.width,img.height)
    region=img.crop(box)
    subregion=[]

    #Size acquisition for each division
    ds_w=float(img.width)/h_split
    ds_h=float(img.height)/w_split
    
    for i in range(h_split):
        #Specify the position of both sides of the split
        wmin=int(ds_w*i)
        wmax=int(ds_w*(i+1))
        for j in range(w_split):
            hmin=int(ds_h*j)
            hmax=int(ds_h*(j+1))
            #Create a box with the obtained data
            box=(wmin,hmin,wmax,hmax)
            subregion.append(region.crop(box))

    path_split=file_name.split("/")
    tmp=path_split[-1].split("-")
    print(tmp[0])
    filename_img=os.path.splitext(os.path.basename(file_name))[0]
    img_dir="../dataset/patchimg/"+filename_img
    os.makedirs(img_dir,exist_ok=True)
    for num in range(int(h_split*w_split)):
        output_filename=img_dir+"/"+str(count)+"-"+str(num+1)+"-"+tmp[0]
        print(output_filename)
        subregion[num].save(output_filename)

args = sys.argv
file_path=args[1]
print(file_path)
files=glob.glob(file_path+"*")

files=natsorted(files)
np.set_printoptions(threshold=np.inf)

i=1
for file in files:
    print(file)
    image_cut(file,13,10,i)
    i+=1
