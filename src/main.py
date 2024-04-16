#import
from turtle import distance
import cv2
import numpy as np
import glob
from datetime import datetime
import os
from tqdm import tqdm
from scipy import interpolate
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
# Original library
import Img_Proc
import neural_age
import center
import mask

def get_args():
    parser = argparse.ArgumentParser(description='Use dataset')
    parser.add_argument('--dataset', '-d', type=str, default=False, help='Use restzone dataset')
    parser.add_argument('--raw-dataset', '-r', type=str, default=False, help='Use raw dataset')
    parser.add_argument('--percent', '-p', type=int, default=10, help='Fake Annual Zone percentage')
    return parser.parse_args()

#Load image
args = get_args()

dataset = args.dataset
raw_dataset=args.raw_dataset
percent = args.percent

input_dir="../dataset/image/"+raw_dataset+".jpg"
estimate_dir="../dataset/restzone/"+dataset+"/"

ext="*.jpg"

#Prepare directory for storage
print("input_dir: ",input_dir)
print("estimate_dir: ", estimate_dir)
image_path=input_dir
result_output_dir = "../dataset/result/"

#Result of determineing of the center
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

#Get file name without extension (used for save name)
base_name=os.path.splitext(os.path.basename(image_path))[0]
print("base_name: ", base_name)

#Loading the original image
original_Image=cv2.imread(image_path,1)

#Load detected image
print(estimate_dir+base_name+".jpg")
estimate_Image=cv2.imread(estimate_dir+base_name+".jpg",1)

######Pre-processing########
basic_Image,l=Img_Proc.Pre_proc(original_Image)

#Directory for storing results
os.makedirs(result_output_dir+base_name,exist_ok=True)
each_output_dir=result_output_dir+base_name
print("each_output_dir: ",each_output_dir)

#Mask processing
mask_Image = mask.create(basic_Image,l,int(l/25),dir_path = each_output_dir)
cv2.imwrite(each_output_dir+'/mask_labeling_' + base_name + '.bmp', mask_Image * 255)
cv2.imwrite(each_output_dir+'/masking_image_' + base_name + '.bmp', mask_Image * basic_Image)
print('maked mask.....')

#Determineing of the center
center_position = center.create(basic_Image, each_output_dir)
print('centersumt3:',center_position)

y_size=1800
x_size=900
#Polar expansion of the original image
flags = cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
linear_polar_raw_image=cv2.warpPolar(original_Image,(x_size,y_size),(center_position[0],center_position[1]),l,flags)
cv2.imwrite(each_output_dir+'/linear_polar_raw_' + base_name + '.bmp', linear_polar_raw_image)

###########################
###Polar expansion of the mask image##
###########################
mask_Image = mask_Image * 255
linear_polar_mask = cv2.warpPolar(mask_Image,(x_size,y_size),(center_position[0],center_position[1]),l,flags)
# print(each_output_dir+"/extend_mask_image_"+base_name+".bmp")
cv2.imwrite(each_output_dir+"/extend_mask_image_"+base_name+".bmp",linear_polar_mask)

#Increase contour by a few px
linear_polar_mask_uint8 = np.array(linear_polar_mask, dtype=np.uint8)
ContoursMask,hierarchy = cv2.findContours(linear_polar_mask_uint8,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
linear_polar_mask = cv2.drawContours(linear_polar_mask_uint8,ContoursMask, -1, (0,255,0), 4)
#Calculate the maximum width size
MaxAreaSize,MaxIndex = neural_age.WidthSize_Calc(linear_polar_mask)

###########################
###Polar coordinate expansion of detection image####
###########################
linear_polar_image=cv2.warpPolar(estimate_Image,(x_size,y_size),(center_position[0],center_position[1]),l,flags)
cv2.imwrite(each_output_dir+"/extend_label_image_"+base_name+".bmp",linear_polar_image)

if linear_polar_image.ndim != 2:
    gray_linear_Image = cv2.cvtColor(linear_polar_image,cv2.COLOR_BGR2GRAY)
else:
    gray_linear_Image = linear_polar_image
    np.set_printoptions(threshold=np.inf)
    linear_polar_image = cv2.cvtColor(linear_polar_image,cv2.COLOR_GRAY2BGR)


#Extract line ends as points
thre=20
contours,hierarchy=cv2.findContours(gray_linear_Image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
contours = list(filter(lambda x: cv2.contourArea(x) > thre, contours))
save_file=each_output_dir+"/extend_image"+base_name+"_rest_zone"+".txt"

cv2.drawContours(linear_polar_image,contours, -1, color=(0, 0, 255), thickness=5)
cv2.imwrite(each_output_dir+"/extend_image_"+base_name+"_contours"+".bmp",linear_polar_image)
y_max=-1
y_min=20000

#Get the upper end point of the region
tail_list = neural_age.TailPixel_Counter(contours)

for AreaLabel in range(len(contours)):
    y_min=20000
    for PixelLabel in range(len(contours[AreaLabel])):
        if y_min >= contours[AreaLabel][PixelLabel][0][1]:
            head_pixel = contours[AreaLabel][PixelLabel][0]
            y_min = contours[AreaLabel][PixelLabel][0][1]
    head_pixel=head_pixel.tolist()
    
    #Interpolate with the region with the shortest distance to pixels other than itself
    distance_min=20000
    min_label=-1
    DistanceY=150
    DistanceX=60
    Degree=20

    #Distance and angle calculation part related to interpolation
    for k in range(len(tail_list)):
        theta=neural_age.getRadian(tail_list[k],head_pixel)
        if AreaLabel!=k and \
            tail_list[k][0]-head_pixel[0] < DistanceY and \
            abs(tail_list[k][1]-head_pixel[1]) < DistanceX and \
            abs(theta) < Degree:
            dist = neural_age.euclid(tail_list[k],head_pixel)
            if dist <= distance_min:
                distance_min=dist
                min_label=k

    pt_x=[]
    pt_y=[]
    #Element construction of interpolation
    if min_label != -1:
        # list -> numpy
        area_a = np.array(contours[min_label])
        area_b = np.array(contours[AreaLabel])

        #Remove dimension with element number 1 by squeeze
        area_a = np.squeeze(area_a)
        area_b = np.squeeze(area_b)

        arg_a=area_a.argsort(axis=0)

        #Sorting based on y-axis px
        area_a_sorted=area_a[np.argsort(area_a[:, 1])]
        area_b_sorted=area_b[np.argsort(area_b[:, 1])]

        #Assign only the ends of points to the array
        pt_x.append(tail_list[min_label][0])
        pt_y.append(tail_list[min_label][1])

        pt_x.append(head_pixel[0])
        pt_y.append(head_pixel[1])

        #Calculate the curve by spline interpolation
        sp_x,sp_y = neural_age.spline3(pt_x,pt_y,100,1)
        
        #Post-processing of interpolation
        pt_x=np.array(sp_x)
        pt_y=np.array(sp_y)

        con = np.stack([pt_x,pt_y],1)
    
        #Drawing on image
        if abs(sp_x[0]-sp_x[-1])<= DistanceX:
            for cnt in range(len(con)):
                cv2.circle(linear_polar_image, (int(con[cnt][0]),int(con[cnt][1])), 0, (0,0,255), thickness=5, lineType=cv2.LINE_8)
        
cv2.imwrite(each_output_dir+"/extend_image_"+base_name+"_before_hokan_red"+".bmp",linear_polar_image)

#Blacken the white areas.
linear_polar_image = Img_Proc.Post_Proc(linear_polar_image)
linear_polar_image = cv2.cvtColor(linear_polar_image, cv2.COLOR_RGB2GRAY)

#Overlay mask and dormant zone image
linear_polar_image = linear_polar_image * (linear_polar_mask)
linear_polar_image = np.clip(linear_polar_image,0,255)
linear_polar_image = np.array(linear_polar_image, dtype=np.uint8)

#Save
cv2.imwrite(each_output_dir+"/extend_image_Mask_Overlay"+".bmp",linear_polar_image)

contours,hierarchy=cv2.findContours(linear_polar_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
contours = list(filter(lambda x: cv2.contourArea(x) > 50, contours))
linear_polar_mask = cv2.drawContours(linear_polar_image,contours, -1, (255,255,255), 2)
cv2.imwrite(each_output_dir+"/extend_image_hokango"+".bmp",linear_polar_image)
#Generate inverse transformed image
flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
linear_polar_inverse_image = cv2.warpPolar(linear_polar_image, (2080,1600),(center_position[0],center_position[1]),l, flags)
cv2.imwrite(each_output_dir + "/inverse_image_" + base_name + ".bmp",linear_polar_inverse_image)

#Counting of resting zones 

count = [0] * y_size
flag = [0] * y_size
flag_default = [0] * y_size

#age determination

FakeAreaPercent = percent
FakeAreaLabel = int(MaxAreaSize * FakeAreaPercent / 100)

#Counting of resting zones
#Outside loop: refer to any region from multiple contour-extracted regions
#Inner loop: reference to any contour pixel for any region

for AreaLabel in range(len(contours)):
    
    for PixelLabel in range(len(contours[AreaLabel])):
        #Counts age only if x-coordinates are above a certain level
        if flag[contours[AreaLabel][PixelLabel][0][1]]==0 and \
            FakeAreaLabel <= contours[AreaLabel][PixelLabel][0][0]:
            count[contours[AreaLabel][PixelLabel][0][1]]+=1
            flag[contours[AreaLabel][PixelLabel][0][1]]+=1
    flag = [0] * y_size

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

print("Pred age:"+str(max(count,default=3)))
if len(count) != 0 and answer_age == max(count):
    correct_count[answer_age-3]+=1
print("=====================")
#exit()
