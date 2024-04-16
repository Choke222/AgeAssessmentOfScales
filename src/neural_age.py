import math
from scipy import interpolate
import numpy as np
import cv2

def WidthSize_Calc(PolarImage : np):
    """
    Function to calculate width size of scales
    Augs: (np) : Mask image after polar coordinates expansion 
        PolarImage(np) : mask image after polar coordinates expansion 
    Returns: int : maximum size of the scale area portion 
        int : maximum size of scale area
        int : x-coordinate of the rightmost side of the scale area
    """
    height = PolarImage.shape[0]
    width = PolarImage.shape[1]
    MaxArea = 0
    MaxIndex = 0
    for h in range(height):
        # Count the number of white areas
        ScaleAreaCount = np.count_nonzero(PolarImage[h] == 255)
        # Obtain the rightmost coordinate of the scaled area when the area is the largest.
        if ScaleAreaCount > MaxArea:
            MaxArea = ScaleAreaCount
            # Reverse array left to right
            WidthFlip = np.flip(PolarImage[h])
            # Get the leftmost index of the largest value
            FlipIndex = np.argmax(WidthFlip)
            # Retrieve the rightmost index by reversing the inverted index back
            MaxIndex = width - 1 -FlipIndex

    return MaxArea,MaxIndex

def euclid(a : list  , b : list ):
    """
    Function to find Euclidean distance between two points
    Augs: function to find the Euclidean distance of a point 
        a,b(list) : 2D coordinates of two points a,b
    Returns: int : Euclidean distance of two points 
        int : Euclidean distance of two points
    """

    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))

def spline3(x,y,point,deg):
    """
    Function to perform a spline interpolation
    Augs: x(list) : data of x-coordinates 
        x(list) : data of x-coordinate
        y(list) : data of y-coordinate
        point : number of point divisions
        deg : degree
    Returns: int : x,y coordinates of the interpolated result 
        int : x,y coordinates of the interpolated point
    """
    tck,u = interpolate.splprep([x,y],k=deg,s=0)
    u = np.linspace(0,1,num=point,endpoint=True) 
    spline = interpolate.splev(u,tck)
    return spline[0],spline[1]

def getRadian(a,b):
    """
    Calculate angles
    Augs: : Calculate the angle of the point a.
        a(list) : coordinate data of point a[x,y].
        b(list) : coordinate data of point b[x,y]
    
    Returns: theta(float) : angle 
        theta(float) : angle
    """
    #Calculate vector
    vec1=[a[0]-a[0],abs(b[1]-a[1])]
    vec2=[a[0]-b[0],b[1]-a[1]]

    #Calculate inner product and norm
    inner=np.inner(vec1,vec2)
    absvec1=np.linalg.norm(vec1)
    absvec2=np.linalg.norm(vec2)

    #Calculate cosθ and convert to θ
    cos_theta=inner/(absvec1*absvec2)
    theta=math.degrees(math.acos(cos_theta))
    return theta

def TailPixel_Counter(RestzoneArea):
    """
    Record the last pixel of each region
    -> end of each region -> largest label in Y direction in the region
    Augs:.
        RestzoneArea : result of area detection by cv2.findContours
    Returns: TailList(List) 
        TailList(List) : Pixels at the end of each input region
    """
    TailList=[]
    #Search each region in turn
    for AreaLabel in range(len(RestzoneArea)):
        y_max=-1
        #Search for pixels in one area
        for PicelLabel in range(len(RestzoneArea[AreaLabel])):
            #Update if the label in the Y direction is the largest in the region
            if y_max <= RestzoneArea[AreaLabel][PicelLabel][0][1]:
                TailPicel = RestzoneArea[AreaLabel][PicelLabel][0]
                y_max = RestzoneArea[AreaLabel][PicelLabel][0][1]
        #Add the element with the largest Y-direction label in the region to the list

        TailList.append(TailPicel)
    return TailList

def HeadPixel_Counter(RestzoneArea):
    """
    Record the first pixel of each region
    -> beginning of each region -> small label in Y direction in the region
    Augs:.
        RestzoneArea : result of area detection by cv2.findContours
    Returns: TailList(List) 
        TailList(List) : Pixels at the beginning of each input region
    """
    HeadList=[]

    #Search each region in turn
    for AreaLabel in range(len(RestzoneArea)):
        #Search for pixels in one area
        y_min=20000
        for PixelLabel in range(len(RestzoneArea[AreaLabel])):
            if y_min >= RestzoneArea[AreaLabel][PixelLabel][0][1]:
                head_pixel = RestzoneArea[AreaLabel][PixelLabel][0]
                y_min = RestzoneArea[AreaLabel][PixelLabel][0][1]
        HeadList.append(head_pixel)
    return HeadList

def AgeEstimate(RestzoneArea,y_size):

    """
    Age determination from interpolated results
    Augs: (for the first time)
        RestzoneArea : Rest zone data detected by region extraction after interpolation
        y_size : Vertical size of the image
    Returns: Basic_Image(ndarray) 
        basic_Image(ndarray) : Result of negative-positive transformation after converting input image to gray image
    """

    flag = [0] * y_size
    for AreaLabel in range(len(RestzoneArea)):
        for PixelLabel in range(len(RestzoneArea[AreaLabel])):
            if flag[RestzoneArea[AreaLabel][PixelLabel][0][1]]==0:
                count[RestzoneArea[AreaLabel][PixelLabel][0][1]]+=1
                flag[RestzoneArea[AreaLabel][PixelLabel][0][1]]+=1
        flag = [0] * y_size
    
    #Delete image elements with 0,1 number of resting zones passed from the image element.
    count=[s for s in count if s != 0]
    count=[s for s in count if s != 1]
