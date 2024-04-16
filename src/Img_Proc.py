import math
from scipy import interpolate
import numpy as np
import cv2 

def Pre_proc(original_image):
    """
    Perform image preprocessing
    Augs: (in the following format)
        original_image : numpy format data read by cv2
    Returns: basic_Image(ndarray) 
        basic_Image(ndarray) : result of negative-positive transformation after converting input image to gray image
        l : long and short sides of the image
    """
    #Gray scale conversion
    gray_Image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    #Negative to positive conversion
    basic_Image = cv2.bitwise_not(gray_Image)
    #Get image size
    h,w=basic_Image.shape
    #Get the size of the long side
    l = w if w > h else h
    return basic_Image,l

def Post_Proc(linear_polar_image):
    """
    Function to show only the areas in the image that were used for interpolation and region detection
    ->Regions detected by PSPNet but not used for interpolation are deleted.
    ->Red for areas used/generated for interpolation, white for areas detected by PSPNet
    Augs:.
        linear_polar_image : Result of interpolation on the image with expanded polar coordinates
    Returns: linear_polar_image 
        OutputImage : Image in which only the area used for interpolation and the line produced by interpolation are drawn.
    """

    black=np.array([0,0,0])
    #Explore each pixel in the image
    for X_Pixel in range(len(linear_polar_image)):
        for Y_Pixel in range(len(linear_polar_image[X_Pixel])):
            #Convert each RGB element to black if it is close to white

            if linear_polar_image[X_Pixel][Y_Pixel][0]>=245 and \
               linear_polar_image[X_Pixel][Y_Pixel][1]>=245 and \
               linear_polar_image[X_Pixel][Y_Pixel][2]>=245:
               linear_polar_image[X_Pixel][Y_Pixel]=black
    OutputImage = linear_polar_image
    return OutputImage

def Pile_Image(ScaleImage,MaskImage,RestzoneImage,OutputDir):
    """
    Function to superimpose a dormant band image and a mask image on a scale image
    Augs: ScaleImage(np) 
        ScaleImage(np) : source image after expansion in polar coordinates
        MaskImage(np) : mask image after expansion in polar coordinates
        RestzoneImage(np) : Restzone image after expansion in polar coordinates
    Returns: None (for output of image only)
    """
    #Overlay mask and dormant zone image
    RestzoneImage = RestzoneImage * (MaskImage)
    RestzoneImage = np.clip(RestzoneImage,0,255)
    RestzoneImage = np.array(RestzoneImage, dtype=np.uint8)
    cv2.imwrite(OutputDir+"/extend_image_test"+".bmp",RestzoneImage)

    #Overlay the pause strip image on the source image
    
    #Invert the pause image to black and white
    RestzoneImage_Reverse = 255 - RestzoneImage
    cv2.imwrite(OutputDir+"/extend_RestzoneImage_Reverse"+".bmp",RestzoneImage_Reverse)

    #RGB conversion of pause zone image

    #Transparent white areas
    transparence = (0,0,0)
    #Overlay on original image




