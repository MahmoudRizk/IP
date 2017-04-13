import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import rotate as rt
import math
from math import e, sqrt, pi
"""
filename = '1.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

plt.imshow(img, cmap='gray',  interpolation='bicubic')
plt.show()
"""
########################3
"""
folder_path = 'MCQ/train/'

for img_index, img_name in enumerate(os.listdir(folder_path)):
    img_path = folder_path + img_name
    #print img_path
    img = cv2.imread(img_path)
    edges = cv2.Canny(img,100,200)
    im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print len(contours[6])
    #print len(contours[7])
    #print len(contours[8])
    #print len(contours[9])
    c = []
    for i in range(0,25):
        if len(contours[i]) >= 170 and len(contours[i]) <= 250:
            c.append(contours[i])
    #print 'len(c): ', len(c)
    cv2.drawContours(img, c, -1, (0,0,255), 3)
    cv2.imwrite('out/'+ str(img_index) +'.png',img)
plt.imshow(img)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
"""

def hough_big_circles(image):
    circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,minDist=30,
                                param1=50,param2=16,minRadius=30,maxRadius=40)

    if len(circles[0])>2:
        for i in range(0,len(circles[0])):
            try:
                if circles[0][i][1] < 1400:
                    circles = np.delete(circles[0],i,0)
                    circles = np.expand_dims(circles, axis=0)
                    print circles
            except:
                print 'from debug', circles
    # 253,242, 230, 216, 195
    try:
        if circles[0][0][0] > circles[0][1][0]:
            circles[0][0][0], circles[0][1][0] = circles[0][1][0], circles[0][0][0]
            circles[0][0][1], circles[0][1][1] = circles[0][1][1], circles[0][0][1]
            circles[0][0][2], circles[0][1][2] = circles[0][1][2], circles[0][0][2]
    except:
        print 'from debug:',circles
    return circles

def draw_circles(image, circles):
    # ensure at least some circles were found
    if circles is not None:
        print 'true'
    	# convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    	# loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
    		# draw the circle in the output image, then draw a rectangle
    		# corresponding to the center of the circle
            if(y>=60-7 and y<=60+7):
                cv2.circle(image, (x, y), r, (255, 0, 0), 4)
            elif(x<=90+7):
                cv2.circle(image, (x, y), r, (255, 0, 0), 4)
            elif(x<=418+7 and x>=313+7):
                cv2.circle(image, (x, y), r, (255, 0, 0), 4)
            elif(x<=747+7 and x>=644+7):
                cv2.circle(image, (x, y), r, (255, 0, 0), 4)
            elif( (x>90 and x<=145) or (x>420 and x<476) or (x>750 and x<800) ): # A answer
                cv2.circle(image, (x, y), r, (0, 0, 255), 4)
            elif( (x>150 and x<190 ) or (x>480 and x<518) or (x>810 and x<850)): #B answer
                cv2.circle(image, (x, y), r, (255, 0, 255), 4)
            elif( (x>195 and x<230 ) or (x>525 and x<560) or (x>855 and x<890)): #c answer
                cv2.circle(image, (x, y), r, (255, 255, 0), 4)
            elif( (x>235 and x<290 ) or (x>565 and x<620) or (x>895)): #D answer
                cv2.circle(image, (x, y), r, (0, 255, 255), 4)
            else:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    		cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        return image

def get_top_circles(circles, threshold):
    # Return list of coordinates of circles above horizontal threshold
    top_circles = []
    for i in range(0,len(circles[0])):
        if circles[0][i][1] <= threshold :
            top_circles.append(circles[0][i])

    top_circles.sort(key=lambda top_circles:top_circles[:][0])
    return top_circles

folder_path = 'MCQ/train/'
for img_index, img_name in enumerate(os.listdir(folder_path)):
    img_path = folder_path + img_name

    if img_index == 50:
        break

    """
    if img_index != 264:
        continue
    """

    print img_index

    # detect circles in the image
    image = cv2.imread(img_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = image.copy()
    #ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    image = cv2.medianBlur(image,5)
    image = cv2.medianBlur(image,5)
    image = cv2.medianBlur(image,5)
    image = cv2.GaussianBlur(image,(5,5),0)
    edges_2 = cv2.Canny(image,100,110)



############################################################################
    # small circles:
    #circles = cv2.HoughCircles(edges_2,cv2.HOUGH_GRADIENT,1,minDist=30,
    #                            param1=10,param2=5,minRadius=5,maxRadius=10)
##############################################################################

    # Big circles
    # 30 40 param2=18 param1=50
    circles = hough_big_circles(edges_2) #1 before rotation
    print circles
    print len(circles[0])

    a = circles[0][1][1] - circles[0][0][1]
    b = circles[0][1][0] - circles[0][0][0]
    theta = math.atan(a/b)
    theta = math.degrees(theta)

    image = rt.rotate_bound(image,-1 * theta)
    output = rt.rotate_bound(output,-1 * theta)
    edges_2 = rt.rotate_bound(edges_2,-1 * theta)

    edges_2 = cv2.Canny(image,100,110)
    circles = hough_big_circles(edges_2) #2 after rotation

    rows, cols, _ = image.shape
    c = (200) - (circles[0][0][0])
    d = (1350) - (circles[0][0][1])
    print 'd:', d
    print 'c:', c
    M = np.float32([[1,0,(c)/2],[0,1,(d)/2]])

    image = cv2.warpAffine(image,M,(cols,rows))
    output = cv2.warpAffine(image,M,(cols,rows))
    edges_2 = cv2.warpAffine(image,M,(cols,rows))

    cv2.circle(output, (200, 1350 ), 20, (0, 255, 255), 7)

    #cv2.circle(output, (80, 525 ), 20, (0, 0, 0), 7)
    #cv2.circle(output, (80, 1250 ), 20, (0, 0, 0), 7)
    #cv2.circle(output, (1075, 1250 ), 20, (0, 0, 0), 7)
    #cv2.circle(output, (1075, 525 ), 20, (0, 0, 0), 7)

    output = output[525:1250, 80:1075]
    image =  image[525+100:1250+100, 80:1075]
    edges_2 = edges_2[525:1250, 80:1075]

    edges_2 = cv2.Canny(edges_2,100,110)


    ############################################################################
    # small circles:
    circles = cv2.HoughCircles(edges_2,cv2.HOUGH_GRADIENT,1,minDist=35,
                                param1=10,param2=5,minRadius=5,maxRadius=10)
    ##############################################################################

    top_circles = get_top_circles(circles, 55)



    a = top_circles[-1][1] - top_circles[0][1]
    b = top_circles[-1][0] - top_circles[0][0]
    theta = math.atan(a/b)
    theta = math.degrees(theta)
    print 'theta:', theta

    image = rt.rotate_bound(image,-1 * theta)
    output = rt.rotate_bound(output,-1 * theta)
    edges_2 = rt.rotate_bound(edges_2,-1 * theta)


    #top_circles = np.expand_dims(top_circles, axis=0)
    #output = draw_circles(output,top_circles)

    #output = output[13:690, 26:972]
    #edges_2 = edges_2[13:690, 26:972]

    d = 0
    circles = cv2.HoughCircles(edges_2,cv2.HOUGH_GRADIENT,1,minDist=35,
                                param1=10,param2=5,minRadius=7,maxRadius=10)

    while(True):

        #circles = cv2.HoughCircles(edges_2,cv2.HOUGH_GRADIENT,1,minDist=35,
        #                        param1=10,param2=5,minRadius=7,maxRadius=10)

        top_circles = get_top_circles(circles,55)
        if len(top_circles) == 0:
            top_circles = get_top_circles(circles,60 + d)

        rows, cols, _ = output.shape
        #c = (200) - int(circles[0][0][0])
        print 'top_circles:', len(top_circles)
        if(len(top_circles) == 0):
            print 'len_circles:', 0
            break
        sum = 0
        for i in range(0,len(top_circles)):
            sum = sum + top_circles[0][1]
        average_y = sum / len(top_circles)
        d = (60) - (average_y)
        m = 10
        s = 7
        x = d

        print 'd:', d

        if(img_index == -1 or True and False ):
            output_2 = output.copy()
            cv2.line(output_2,(20,60),(930,60),(255,0,0),5)
            output_2 = draw_circles(output_2,circles)
            plt.imshow(output_2)
            plt.title('Before Image'), plt.xticks([]), plt.yticks([])
            plt.show()


        if(abs(d)<2):
            break
        #print 'c:', c

        d_2 = e**(-0.5*(float(d-m)/s)**2) * d
        print 'd_2 before:', d_2
        if abs(d_2) < 0.0000009:
            d_2 = e**(-0.5*(float(d-m)/s)**2) * d * 10000000
        if abs(d_2) < 0.000009:
            d_2 = e**(-0.5*(float(d-m)/s)**2) * d * 1000000
        if abs(d_2) < 0.00009:
            d_2 = e**(-0.5*(float(d-m)/s)**2) * d * 100000
        if abs(d_2) < 0.0009:
            d_2 = e**(-0.5*(float(d-m)/s)**2) * d * 10000
        if abs(d_2) < 0.009:
            d_2 = e**(-0.5*(float(d-m)/s)**2) * d * 1000
        if abs(d_2) < 0.09:
            d_2 = e**(-0.5*(float(d-m)/s)**2) * d * 100
        print 'd_2:', d_2


        M = np.float32([[1,0,0],[0,1,(d_2)/2]])

        image = cv2.warpAffine(image,M,(cols,rows))
        output = cv2.warpAffine(image,M,(cols,rows))
        edges_2 = cv2.warpAffine(image,M,(cols,rows))
        edges_2 = cv2.Canny(output,100,110)


        for i in range(0,circles.shape[1]):
            #print 'circles_before:',circles[0][i][1]
            circles[0][i][1] = circles[0][i][1] + d_2
            #print 'circles_after:',circles[0][i][1]


        if(img_index == -1 or True and False ):
            output_2 = output.copy()
            cv2.line(output_2,(20,60),(930,60),(255,0,0),5)
            output_2 = draw_circles(output_2,circles)
            plt.imshow(output_2)
            plt.title('After Image'), plt.xticks([]), plt.yticks([])
            plt.show()


    #edges_2 = cv2.Canny(edges_2,100,110)
    #circles = cv2.HoughCircles(edges_2,cv2.HOUGH_GRADIENT,1,minDist=35,
    #                            param1=10,param2=5,minRadius=5,maxRadius=10)


    output = np.full(output.shape,255)
    output = draw_circles(output,circles)
    sum = 0
    index = 0
    tmp = []
    for i in range(0,circles.shape[1]):
        if circles[0][i][0] < 95:
            sum = sum + circles[0][i][0]
            index = index + 1
            tmp.append(circles[0][i][0])
    average = sum / index
    average = int(average)
    x = average + 30
    print 'sum:', sum
    print 'average:', average
    print 'number_of_left_circles:', index

    cv2.line(output,(x,10),(x,720),(0,0,0),5)
    cv2.line(output,(x+225,10),(x+225,720),(0,0,0),5)
    cv2.line(output,(x+555,10),(x+555,720),(0,0,0),5)

    # middle 3 verticle line

    cv2.line(output,(x+331,10),(x+331,720),(0,0,0),5)
    cv2.line(output,(x+658,10),(x+658,720),(0,0,0),5)

    # first sub 3 lines
    cv2.line(output,(x+55,10),(x+55,720),(0,0,0),3)
    cv2.line(output,(x+95,10),(x+95,720),(0,0,0),3)
    cv2.line(output,(x+135,10),(x+135,720),(0,0,0),3)

    # second sub 3 lines
    cv2.line(output,(x+385,10),(x+385,720),(0,0,0),3)
    cv2.line(output,(x+425,10),(x+425,720),(0,0,0),3)
    cv2.line(output,(x+465,10),(x+465,720),(0,0,0),3)

    # third sub 3 lines
    cv2.line(output,(x+705,10),(x+705,720),(0,0,0),3)
    cv2.line(output,(x+755,10),(x+755,720),(0,0,0),3)
    cv2.line(output,(x+795,10),(x+795,720),(0,0,0),3)


    cv2.line(output,(20,60),(930,60),(0,0,0),5)

    delta_1 = 60
    dellta_2 = 40
    for i in range(0,14):
        cv2.line(output,(20,(60+delta_1) + i * dellta_2),(930,(60+delta_1) + i * dellta_2 ),(0,0,0),3)
    #plt.imshow(edges_2)
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()


    vis = np.concatenate((image, output), axis=1)
    #vis = output
    cv2.imwrite('out/'+ str(img_index) +'.png',vis)

    """
    output_2 = output.copy()
    cv2.line(output_2,(20,60),(930,60),(255,0,0),5)
    output_2 = draw_circles(output_2,circles)
    plt.imshow(output_2)
    plt.title('After Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    """


"""
image = cv2.imread('4.png')
blur = cv2.GaussianBlur(image,(9,9),0)
edges_1 = cv2.Canny(image,100,200)
edges_2 = cv2.Canny(blur,100,200)

plt.subplot(121),plt.imshow(image)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])


plt.subplot(222),plt.imshow(blur)
plt.title('Blur Image'), plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(edges_1, cmap='gray')
plt.title('Canny_1 Image'), plt.xticks([]), plt.yticks([])


plt.subplot(122),plt.imshow(edges_2, cmap='gray')
plt.title('Canny_2 Image'), plt.xticks([]), plt.yticks([])

plt.show()
"""
