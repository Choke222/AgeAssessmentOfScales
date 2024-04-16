# coding: utf-8
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def create(img,dir_path):
    center_position = check_center(img, dir_path)

    return center_position

#Spectrum calculation
def spec(F):
    spc = np.fft.fftshift(F)

    spc[spc < 1] = 1
    P = np.log10(spc)
    P_norm = P/np.amax(P)

    return np.uint8(np.around(P_norm*255))
#Find the angle between pixels for the upper half of the Fourier amplitude
#To multiply the numerical values of each pixel as weights and sum them up
#Result is the angle of this region
#find the center of each block by finding the angle of each block
#return is the angle of the maximum intensity value
def vecter(dif):
    #Size to cut through center of strength
    #B=0:1*1
    #B=1:3*3
    B =1
    #中心座標用
    centerx = np.int(dif.shape[1]/2)
    centery = np.int(dif.shape[0]/2)

    #Get the angle (complex) for each coordinate and save it to set
    set_lx = np.zeros((centery, dif.shape[1]))
    set_ly = np.zeros((centery, dif.shape[1]))
    for l in range(centery):
        set_lx[l,:] = np.arange(dif.shape[1])
    for l in range(dif.shape[1]):
        set_ly[:,l] = np.arange(centery)
    set = complex_(centerx,centery,set_lx,set_ly)

    #Center strength is preserved and the center is clipped
    dif_center = dif[centery, centerx]
    dif[centery-B:centery+B+1, centerx-B:centerx+B+1] = np.zeros((2*B+1,2*B+1))
    if dif_center * 0.01 > np.max(dif):
        return 0.1

    #Locate the maximum intensity outside the clipped center and save the coordinates
    dif_tmp = np.where(np.max(dif[:centery,:]) > dif[:centery,:], 0, dif[:centery,:])
    idx = np.unravel_index(np.argmax(dif_tmp), dif_tmp.shape)

    dif_tmp2 = dif[:centery,:] - dif_tmp
    idx2 = np.unravel_index(np.argmax(dif_tmp2), dif_tmp2.shape)
    r1 = np.angle(set[idx])
    r2 = np.angle(set[idx2])

    xid = idx[0]
    yid = idx[1]
    sum1 = np.sum(dif[yid-1:yid+2,xid-1:xid+2])

    xid_local = xid - centerx
    yid_local = yid - centery

    xid_ = -yid_local + centerx
    yid_ = xid_local + centery
    sum2 = np.sum(dif[yid_-1:yid_+2,xid_-1:xid_+2])

    #Apply the maximum intensity as intensity to the #set complex.
    sum = np.sum(set * dif_tmp[:centery,:])
    rad = np.angle(sum)
    abs = np.abs(sum)
    rad = np.angle(set[idx])

    return rad
#Return angle from center in complex

def complex_(centerx,centery,x,y):
    X = x - centerx
    Y = y - centery

    abs = np.sqrt(np.square(X) + np.square(Y))

    C = (X + Y * 1j)
    return C/np.abs(C)
#Draw a line from the center of the block at #x,y coordinates to pixcelspace at angle rad

def drow(img, pixcelspace, rad, x, y, blockx, blocky):
    global height, width
    wait = 1

    if rad == 0.1 or rad == 0 or rad == np.pi/2:
        return

    hx = blockx/2
    hy = blocky/2
    X = x + hx
    Y = y + hy
    a = np.tan(rad)
    x_y0 = 0
    x_yheight = 0


    if rad == 0:
        y_ = np.int(Y)
        for x_ in range(width):
            if y_ < 0 or y_ >= height:
                continue
            pixcelspace[y_,x_] = pixcelspace[y_,x_] + wait
        #img.line([(0,y_), (width,y_)], fill='red')
        cv2.line(img,(x_y0,0), (x_yheight,height), (0,0,255))
    else:
        x_y0 = np.int(X - Y/a)
        x_yheight = np.int((height - Y)/a + X)
        if np.abs(rad) > np.pi/4 and np.abs(rad) < np.pi*3/4:
            for y_ in range(height):
                x_ = np.int((y_ - Y)/a + X)
                if x_ < 0 or x_ >= width:
                    continue
                pixcelspace[y_,x_] = pixcelspace[y_,x_] + wait

        else:
            for x_ in range(width):
                y_ = np.int(a*(x_ - X) + Y)
                if y_ < 0 or y_ >= height:
                    continue
                pixcelspace[y_,x_] = pixcelspace[y_,x_] + wait
        #img.line([(0,x_y0), (height, x_yheight)], fill='red')
        #img.line([(x_y0,0), (x_yheight,height)], fill='red')
        cv2.line(img,(x_y0,0), (x_yheight,height), (0,0,255))
def drow_(write_Image, rad, x, y, blockx, blocky):
    height, width = write_Image.shape
    wait = 1
    #Excluded as no significant pattern
    if rad == 0 or rad == np.pi/2:
        return

    hx = blockx/2
    hy = blocky/2
    X = x + hx
    Y = y + hy
    a = np.tan(rad)

    x_y0 = np.int(X - Y/a)
    x_yheight = np.int((height - Y)/a + X)
    if np.abs(rad) > np.pi/4 and np.abs(rad) < np.pi*3/4:
        for y_ in range(height):
            x_ = np.int((y_ - Y)/a + X)
            if x_ < 0 or x_ >= width:
                continue
            write_Image[y_,x_] = write_Image[y_,x_] + wait

    else:
        for x_ in range(width):
            y_ = np.int(a*(x_ - X) + Y)
            if y_ < 0 or y_ >= height:
                continue
            write_Image[y_,x_] = write_Image[y_,x_] + wait
    #img.line([(0,x_y0), (height, x_yheight)], fill='red')
    #img.line([(x_y0,0), (x_yheight,height)], fill='red')
    cv2.line(write_Image,(x_y0,0), (x_yheight,height), (0,0,255))

# center decision
t0=time.time()
def check_center(image, dir_path, blocknum = 28, overspl = 2):
    global height, width
    #Center coordinates
    center_position = []
    #Image of the result of the final judgment
    center_image = image.copy()
    #Image of lines drawn in each block
    lines_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Num of loops
    loop_num = 3
    #Save the upper left coordinate of the image cropped during the loop
    position_h = np.array([])
    position_w = np.array([])

    #Increase accuracy by performing center determination multiple times.
    for ln in range(loop_num):
        #If it is the second time or later, cut out the center area and determine the center again.
        if ln >= 1:
            #Size to cut out
            scale_h = height/2
            scale_w = width/2
            #Coordinates of the center of scales
            center_X = np.int(blockx * (maxdata[1] + 1))
            center_Y = np.int(blocky * (maxdata[2] + 1))
            #Crop the image around the center of the scale
            height1 = np.int(center_X - scale_w/2)
            height2 = np.int(center_X + scale_w/2)
            width1 = np.int(center_Y - scale_h/2)
            width2 = np.int(center_Y + scale_h/2)
            if height1 < 0:
                height1 = 0
            if height2 > height:
                height2 = height
            if width1 < 0:
                width1 = 0
            if width2 > width:
                width2 = width
            image = image[width1:width2,height1:height2]
            position_h = np.append(position_h,height1)
            position_w = np.append(position_w,width1)

        height = image.shape[0]
        width = image.shape[1]

        outim = np.zeros(image.shape)
        outim = image.copy()
        lines_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        #Block size
        blockx = np.int(image.shape[1]/blocknum)
        blocky = np.int(image.shape[0]/blocknum)
        #Weight of the drawn line in each pixel
        pixcelspace = np.zeros(image.shape)

        for x in range(blocknum-overspl):
            for y in range(blocknum-overspl):
                X = blockx * x
                Xn = blockx * (x + overspl)
                Y = blocky * y
                Yn = blocky * (y + overspl)
                #focus:Cropped image
                focus = image[Y:Yn,X:Xn]
                #If you don't see anything in the image, CONTINUE
                if np.max(focus) == 0:
                    #continue
                    v = 0.1
                else:
                    focus_fft = np.fft.fft2(focus)
                    focus_fftshift = np.fft.fftshift(focus_fft)
                    focus_abs = np.absolute(focus_fftshift)
                    dif = focus_abs
                    v = vecter(dif)
                out0 = outim[Y:Y+blocky,X:X+blockx].shape[0]
                out1 = outim[Y:Y+blocky,X:X+blockx].shape[1]

                if v == 0.1:
                    #print('x:{0}¥ty:{1}'.format(x,y))
                    outim[Y:Y+blocky,X:X+blockx] = np.zeros((out0,out1))
                elif v == 0:
                    #print('x:{0}¥ty:{1}'.format(x,y))
                    outim[Y:Y+blocky,X:X+blockx] = np.ones((out0,out1))*150
                else:
                    drow(lines_image, pixcelspace, v, X, Y, blockx, blocky)

        #Determine 3 blocks with high density
        maxdata = [0, 0, 0]
        snddata = [0, 0, 0]
        trddata = [0, 0, 0]

        center = image.copy()
        for x in range(blocknum-overspl):
            for y in range(blocknum-overspl):
                X = blockx * x
                Xn = blockx * (x + overspl)
                Y = blocky * y
                Yn = blocky * (y + overspl)
                tmp = np.sum(pixcelspace[Y:Yn,X:Xn])
                if tmp > maxdata[0]:
                    trddata = snddata
                    snddata = maxdata
                    maxdata = [tmp,x,y]
                elif tmp > snddata[0]:
                    trddata = snddata
                    snddata = [tmp,x,y]
                elif tmp > trddata[0]:
                    trddata = [tmp,x,y]

        #Reset figure drawn in loop
        plt.clf()

        #Calculated to the 3rd place of the central judgment candidate
        """
        X = blockx * trddata[1]
        Xn = blockx * (trddata[1] + overspl)
        Y = blocky * trddata[2]
        Yn = blocky * (trddata[2] + overspl)
        center[Y:Yn,X:Xn] = 50
        X = blockx * snddata[1]
        Xn = blockx * (snddata[1] + overspl)
        Y = blocky * snddata[2]
        Yn = blocky * (snddata[2] + overspl)
        center[Y:Yn,X:Xn] = 100
        """
        X = blockx * maxdata[1]
        Xn = blockx * (maxdata[1] + overspl)
        Y = blocky * maxdata[2]
        Yn = blocky * (maxdata[2] + overspl)
        #center[Y:Yn,X:Xn] = 150
        #Rectangle without fill thickness 10
        cv2.rectangle(center, (X, Y), (Xn, Yn),color=(0,0,255), thickness=10)
        #Save the image of the determined center
        cv2.imwrite(dir_path + '/center_image_blocknum{0}_{1}.png'.format(blocknum,ln), center)

    #Correct misalignment between the cropped image and the original image.
    sum_h = np.int(np.sum(position_h))
    sum_w = np.int(np.sum(position_w))

    #Calculated to the 3rd place of the central judgment candidate
    """
    X = blockx * trddata[1] + sum_h
    Xn = blockx * (trddata[1] + overspl) + sum_h
    Y = blocky * trddata[2] + sum_w
    Yn = blocky * (trddata[2] + overspl) + sum_w
    center_image[Y:Yn,X:Xn] = 50
    X = blockx * snddata[1] + sum_h
    Xn = blockx * (snddata[1] + overspl) + sum_h
    Y = blocky * snddata[2] + sum_w
    Yn = blocky * (snddata[2] + overspl) + sum_w
    center_image[Y:Yn,X:Xn] = 100
    """
    X = blockx * maxdata[1] + sum_h
    Xn = blockx * (maxdata[1] + overspl) + sum_h
    Y = blocky * maxdata[2] + sum_w
    Yn = blocky * (maxdata[2] + overspl) + sum_w
    #center_image[Y:Yn,X:Xn] = 150
    #Rectangle without fill thickness 10
    cv2.rectangle(center_image, (X, Y), (Xn, Yn),color=(0,0,255), thickness=10)

    center_position = [np.int((X+Xn)/2),np.int((Y+Yn)/2)]

    #Save the image of the center judgment result
    cv2.imwrite(dir_path + '/center_image_blocknum{0}.png'.format(blocknum), center_image)
    #Save center coordinates
    np.savetxt(dir_path + '/profile{0}.txt'.format(blocknum),[np.int((X+Xn)/2),np.int((Y+Yn)/2)])

    #Saved to see where the odd data was when determining the angle
    cv2.imwrite(dir_path + '/check_rad{0}.png'.format(blocknum), outim)
    #Save line-only data
    pixcelspace = pixcelspace*255/np.max(pixcelspace)
    cv2.imwrite(dir_path + '/pixcelspace{0}.png'.format(blocknum), np.uint8(pixcelspace))


    return center_position
def check_center_(image, dir_path, blocknum = 28, overspl = 2):
    global height, width
    #Center coordinates
    center_position = []
    #Image for cropping
    triming_Image = image.copy()
    #Image of the result of the final judgment
    center_image = image.copy()
    #Images of lines drawn in each block
    lines_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #Num of loops
    loop_num = 3
    #Save the upper left coordinate of the image cropped during the loop
    position_h = np.array([])
    position_w = np.array([])

    #Increase accuracy by performing center determination multiple times.
    for ln in range(loop_num):
        #Image size
        height = triming_Image.shape[0]
        width = triming_Image.shape[1]

        #Block Size
        blockx = np.int(image.shape[1]/blocknum)
        blocky = np.int(image.shape[0]/blocknum)

        #outim = np.zeros(triming_Image.shape)
        #outim = triming_Image.copy()
        #lines_image = cv2.cvtColor(triming_Image, cv2.COLOR_GRAY2BGR)

        #Weight of the drawn line in each pixel
        #pixcelspace = np.zeros(image.shape)
        pixcelspace = find_pattern_angle(triming_Image, blocknum, overspl)


        #Determine 3 blocks with high density
        maxdata = [0, 0, 0]
        snddata = [0, 0, 0]
        trddata = [0, 0, 0]

        center = triming_Image.copy()
        for x in range(blocknum-overspl):
            for y in range(blocknum-overspl):
                X = blockx * x
                Xn = blockx * (x + overspl)
                Y = blocky * y
                Yn = blocky * (y + overspl)
                tmp = np.sum(pixcelspace[Y:Yn,X:Xn])
                if tmp > maxdata[0]:
                    trddata = snddata
                    snddata = maxdata
                    maxdata = [tmp,x,y]
                elif tmp > snddata[0]:
                    trddata = snddata
                    snddata = [tmp,x,y]
                elif tmp > trddata[0]:
                    trddata = [tmp,x,y]

        #Reset figure drawn in loop
        plt.clf()

        #Calculated to the 1st candidate of the central judgment
        X = blockx * maxdata[1]
        Xn = blockx * (maxdata[1] + overspl)
        Y = blocky * maxdata[2]
        Yn = blocky * (maxdata[2] + overspl)
        #Rectangle without fill thickness 10
        center[Y:Y+10,X:Xn] = 150
        center[Y-10:Yn,X:Xn] = 150
        center[Y:Yn,X:X+10] = 150
        center[Y:Yn,X-10:Xn] = 150
        #Save the image of the determined center
        cv2.imwrite(dir_path + '/center_image_blocknum{0}_{1}.png'.format(blocknum,ln), center)
        #Coordinates of the center of the scale
        center_X = np.int(blockx * (maxdata[1] + overspl/2))
        center_Y = np.int(blocky * (maxdata[2] + overspl/2))

        triming_Image, height1, width1 = triming(image,center_X,center_Y)

        if ln < loop_num - 1:
            position_h = np.append(position_h,height1)
            position_w = np.append(position_w,width1)



    #Remedy the misalignment between the cropped image and the original image
    sum_h = np.int(np.sum(position_h))
    sum_w = np.int(np.sum(position_w))

    #Calculated to the 3rd place of the central judgment candidate
    '''X = blockx * trddata[1] + sum_h
    Xn = blockx * (trddata[1] + overspl) + sum_h
    Y = blocky * trddata[2] + sum_w
    Yn = blocky * (trddata[2] + overspl) + sum_w
    center_image[Y:Yn,X:Xn] = 50
    X = blockx * snddata[1] + sum_h
    Xn = blockx * (snddata[1] + overspl) + sum_h
    Y = blocky * snddata[2] + sum_w
    Yn = blocky * (snddata[2] + overspl) + sum_w
    center_image[Y:Yn,X:Xn] = 100'''
    X = blockx * maxdata[1] + sum_h
    Xn = blockx * (maxdata[1] + overspl) + sum_h
    Y = blocky * maxdata[2] + sum_w
    Yn = blocky * (maxdata[2] + overspl) + sum_w
    #center_image[Y:Yn,X:Xn] = 150
    #Rectangle without fill thickness 10
    center[Y:Y+10,X:Xn] = 150
    center[Y-10:Yn,X:Xn] = 150
    center[Y:Yn,X:X+10] = 150
    center[Y:Yn,X-10:Xn] = 150

    center_position = [np.int((X+Xn)/2),np.int((Y+Yn)/2)]

    #Save the image of the center judgment result
    cv2.imwrite(dir_path + '/center_image_blocknum{0}.png'.format(blocknum), center_image)
    #Save center coordinates
    np.savetxt(dir_path + '/profile{0}.txt'.format(blocknum),[np.int((X+Xn)/2),np.int((Y+Yn)/2)])

    #Saved to see where the odd data was when determining the angle
    #cv2.imwrite(dir_path + '/check_rad{0}.png'.format(blocknum), outim)
    #Save line-only data
    pixcelspace = pixcelspace*255/np.max(pixcelspace)
    cv2.imwrite(dir_path + '/pixcelspace{0}.png'.format(blocknum), np.uint8(pixcelspace))

    t1=time.time()-t0
    print("center decision")
    print(t1)
    return center_position

def triming(image,center_X,center_Y):
    height, width = image.shape
    #Cut-out size
    scale_h = height/2
    scale_w = width/2

    #Crop the image around the center of the scale
    height1 = np.int(center_X - scale_w/2)
    height2 = np.int(center_X + scale_w/2)
    width1 = np.int(center_Y - scale_h/2)
    width2 = np.int(center_Y + scale_h/2)
    #If outside the image, keep it within the image
    if height1 < 0:
        height1 = 0
    if height2 > height:
        height2 = height
    if width1 < 0:
        width1 = 0
    if width2 > width:
        width2 = width
    return image[width1:width2,height1:height2], width1, height1

def find_pattern_angle(image, blocknum, overspl):
    #Block size
    blockx = np.int(image.shape[1]/blocknum)
    blocky = np.int(image.shape[0]/blocknum)
    #Images for writing angle lines in each block
    drow_Image = np.zeros((image.shape))

    for x in range(blocknum-overspl):
        for y in range(blocknum-overspl):
            X = blockx * x
            Xn = blockx * (x + overspl)
            Y = blocky * y
            Yn = blocky * (y + overspl)
            #Image focus in each block
            focus = image[Y:Yn,X:Xn]
            #If you don't see anything in the image, CONTINUE
            if np.max(focus) == 0:
                continue
            else:
                focus_fft = np.fft.fft2(focus)
                focus_fftshift = np.fft.fftshift(focus_fft)
                focus_abs = np.absolute(focus_fftshift)
                angle = vecter(focus_abs)

            drow_(drow_Image, angle, X, Y, blockx, blocky)

    return drow_Image


def sample(image, loop_num, center_position):
    #If it is the second time or later, cut out the area around the center and determine the center again.
    if ln >= 1:
        #Cut-out size
        scale_h = height/2
        scale_w = width/2
        #Coordinates of the center of the scale
        center_X = np.int(blockx * (maxdata[1] + 1))
        center_Y = np.int(blocky * (maxdata[2] + 1))
        #Crop the image around the center of the scale
        height1 = np.int(center_X - scale_w/2)
        height2 = np.int(center_X + scale_w/2)
        width1 = np.int(center_Y - scale_h/2)
        width2 = np.int(center_Y + scale_h/2)
        if height1 < 0:
            height1 = 0
        if height2 > height:
            height2 = height
        if width1 < 0:
            width1 = 0
        if width2 > width:
            width2 = width
        image = image[width1:width2,height1:height2]
        position_h = np.append(position_h,height1)
        position_w = np.append(position_w,width1)

    height = image.shape[0]
    width = image.shape[1]

    outim = np.zeros(image.shape)
    outim = image.copy()
    lines_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #Block size
    blockx = np.int(image.shape[1]/blocknum)
    blocky = np.int(image.shape[0]/blocknum)
    #Weight of the drawn line in each pixel
    pixcelspace = np.zeros(image.shape)

    for x in tqdm(range(blocknum-overspl)):
        for y in range(blocknum-overspl):
            X = blockx * x
            Xn = blockx * (x + overspl)
            Y = blocky * y
            Yn = blocky * (y + overspl)
            #focus:Cropped image
            focus = image[Y:Yn,X:Xn]
            #If you don't see anything in the image, CONTINUE
            if np.max(focus) == 0:
                #continue
                v = 0.1
            else:
                focus_fft = np.fft.fft2(focus)
                focus_fftshift = np.fft.fftshift(focus_fft)
                focus_abs = np.absolute(focus_fftshift)
                dif = focus_abs
                v = vecter(dif)
            out0 = outim[Y:Y+blocky,X:X+blockx].shape[0]
            out1 = outim[Y:Y+blocky,X:X+blockx].shape[1]

            if v == 0.1:
                #print('x:{0}¥ty:{1}'.format(x,y))
                outim[Y:Y+blocky,X:X+blockx] = np.zeros((out0,out1))
            elif v == 0:
                #print('x:{0}¥ty:{1}'.format(x,y))
                outim[Y:Y+blocky,X:X+blockx] = np.ones((out0,out1))*150
            else:
                drow(lines_image, pixcelspace, v, X, Y, blockx, blocky)

    #Determine 3 blocks with high density
    maxdata = [0, 0, 0]
    snddata = [0, 0, 0]
    trddata = [0, 0, 0]

    center = image.copy()
    for x in range(blocknum-overspl):
        for y in range(blocknum-overspl):
            X = blockx * x
            Xn = blockx * (x + overspl)
            Y = blocky * y
            Yn = blocky * (y + overspl)
            tmp = np.sum(pixcelspace[Y:Yn,X:Xn])
            if tmp > maxdata[0]:
                trddata = snddata
                snddata = maxdata
                maxdata = [tmp,x,y]
            elif tmp > snddata[0]:
                trddata = snddata
                snddata = [tmp,x,y]
            elif tmp > trddata[0]:
                trddata = [tmp,x,y]

    #Reset figure drawn in loop
    plt.clf()

    #Calculated to the 3rd place of the central judgment candidate
    X = blockx * trddata[1]
    Xn = blockx * (trddata[1] + overspl)
    Y = blocky * trddata[2]
    Yn = blocky * (trddata[2] + overspl)
    center[Y:Yn,X:Xn] = 50
    X = blockx * snddata[1]
    Xn = blockx * (snddata[1] + overspl)
    Y = blocky * snddata[2]
    Yn = blocky * (snddata[2] + overspl)
    center[Y:Yn,X:Xn] = 100
    X = blockx * maxdata[1]
    Xn = blockx * (maxdata[1] + overspl)
    Y = blocky * maxdata[2]
    Yn = blocky * (maxdata[2] + overspl)
    center[Y:Yn,X:Xn] = 150
    #Save the image of the determined center
    cv2.imwrite(dir_path + '/center_image_blocknum{0}_{1}.png'.format(blocknum,ln), center)
