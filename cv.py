import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import rotate as rt
import math
from math import e, sqrt, pi
import collections
import json


def hough_big_circles(image):
    circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,minDist=30,
                                param1=50,param2=16,minRadius=30,maxRadius=40)

    if len(circles[0])>2:
        for i in range(0,len(circles[0])):
            try:
                if circles[0][i][1] < 1400:
                    circles = np.delete(circles[0],i,0)
                    circles = np.expand_dims(circles, axis=0)
                    #print circles
            except:
                pass
                #print 'from debug', circles
    # 253,242, 230, 216, 195
    #try:
    if circles[0][0][0] > circles[0][1][0]:
        circles[0][0][0], circles[0][1][0] = circles[0][1][0], circles[0][0][0]
        circles[0][0][1], circles[0][1][1] = circles[0][1][1], circles[0][0][1]
        circles[0][0][2], circles[0][1][2] = circles[0][1][2], circles[0][0][2]
    #except:
        #pass
        #print 'from debug:',circles
    return circles

def create_look_up(circles):
    # ensure at least some circles were found
    if circles is not None:
        #print 'true'
    	# convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    	# loop over the (x, y) coordinates and radius of the circles
        look_up = {}
        for (x, y, r) in circles:
    		# draw the circle in the output image, then draw a rectangle
    		# corresponding to the center of the circle
            if(y>=60-7 and y<=60+7):
                pass
                #cv2.circle(image, (x, y), r, (255, 0, 0), 4)
            elif(x<=90+7):
                pass
                #cv2.circle(image, (x, y), r, (255, 0, 0), 4)
            elif(x<=418+7 and x>=313+7):
                pass
                #cv2.circle(image, (x, y), r, (255, 0, 0), 4)
            elif(x<=747+7 and x>=644+7):
                #cv2.circle(image, (x, y), r, (255, 0, 0), 4)
                pass
            elif( (x>90 and x<=145) or (x>420 and x<476) or (x>750 and x<800) ): # A answer
                #cv2.circle(image, (x, y), r, (0, 0, 255), 4)
                look_up[(x,y)] = 'A'
            elif( (x>150 and x<190) or (x>480 and x<518) or (x>810 and x<850)): #B answer
                #cv2.circle(image, (x, y), r, (255, 0, 255), 4)
                look_up[(x,y)] = 'B'
            elif( (x>195 and x<230) or (x>525 and x<560) or (x>855 and x<890)): #c answer
                #cv2.circle(image, (x, y), r, (255, 255, 0), 4)
                look_up[(x,y)] = 'C'
            elif( (x>235 and x<290 ) or (x>565 and x<620) or (x>895)): #D answer
                #cv2.circle(image, (x, y), r, (0, 255, 255), 4)
                look_up[(x,y)] = 'D'
            else:
                pass
                #cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    		#cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        return look_up


def draw_circles(image, circles):
    # ensure at least some circles were found
    if circles is not None:
        #print 'true'
    	# convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    	# loop over the (x, y) coordinates and radius of the circles
        look_up = {}
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
                look_up[(x,y)] = 'A'
            elif( (x>150 and x<190 ) or (x>480 and x<518) or (x>810 and x<850)): #B answer
                cv2.circle(image, (x, y), r, (255, 0, 255), 4)
                look_up[(x,y)] = 'B'
            elif( (x>195 and x<230 ) or (x>525 and x<560) or (x>855 and x<890)): #c answer
                cv2.circle(image, (x, y), r, (255, 255, 0), 4)
                look_up[(x,y)] = 'C'
            elif( (x>235 and x<290 ) or (x>565 and x<620) or (x>895)): #D answer
                cv2.circle(image, (x, y), r, (0, 255, 255), 4)
                look_up[(x,y)] = 'D'
            else:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    		cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        return image

def remove_unwanted_circles(circles):
    # ensure at least some circles were found
    if circles is not None:
        #print 'true'
    	# convert the (x, y) coordinates and radius of the circles to integers
        #circles = np.round(circles[0, :]).astype("int")
    	# loop over the (x, y) coordinates and radius of the circles
        new_circles = []

        for i in range(0,len(circles[0])):
            x = circles[0][i][0]
            y = circles[0][i][1]
    		# draw the circle in the output image, then draw a rectangle
    		# corresponding to the center of the circle
            #print 'type(circles):', type(circles)
            #print circles
            if(y>=60-10 and y<=60+10):
                continue
            elif(x<=90+15):
                continue
            elif(x<=418+15 and x>=313-10):
                continue
            elif(x<=747+15 and x>=644-10):
                continue
            else:
                new_circles.append(circles[0][i])

        new_circles = np.array(new_circles)
        new_circles = np.expand_dims(new_circles, axis=0)
        return new_circles


def get_top_circles(circles, threshold):
    # Return list of coordinates of circles above horizontal threshold
    top_circles = []
    for i in range(0,len(circles[0])):
        if circles[0][i][1] <= threshold :
            top_circles.append(circles[0][i])

    top_circles.sort(key=lambda top_circles:top_circles[:][0])
    return top_circles

def track_answer(circles):
    x_0 = 90
    x_1 = 315
    upper_bound = 80
    lower_bound = upper_bound + 40
    look_up = create_look_up(circles)
    answer = {}
    for i in range(0,15):
        for j in look_up:
            if j[0] < x_1 and j[0] > x_0:
                if j[1] > upper_bound and j[1] < lower_bound:
                    answer['Q'+str(i+1)] = look_up[j]
                    #answer.append(look_up[j])
        upper_bound = upper_bound + 40
        lower_bound = lower_bound + 40

    x_0 = 425
    x_1 = 650
    upper_bound = 80
    lower_bound = upper_bound + 40
    for i in range(15,30):
        for j in look_up:
            if j[0] < x_1 and j[0] > x_0:
                if j[1] > upper_bound and j[1] < lower_bound:
                    answer['Q'+str(i+1)] = look_up[j]
                    #answer.append(look_up[j])
        upper_bound = upper_bound + 40
        lower_bound = lower_bound + 40

    x_0 = 750
    x_1 = 950
    upper_bound = 80
    lower_bound = upper_bound + 40
    for i in range(30,45):
        for j in look_up:
            if j[0] < x_1 and j[0] > x_0:
                if j[1] > upper_bound and j[1] < lower_bound:
                    answer['Q'+str(i+1)] = look_up[j]
                    #answer.append(look_up[j])
        upper_bound = upper_bound + 40
        lower_bound = lower_bound + 40

    return answer





folder_path = 'MCQ/train/'
answers = {}

for img_index, img_name in enumerate(os.listdir(folder_path)):
    img_path = folder_path + img_name

    if img_index == 5000000000000:
        break

    print 'Image_number: ',img_index

    # detect circles in the image
    image = cv2.imread(img_path)
    output = image.copy()
    image = cv2.medianBlur(image,5)
    image = cv2.medianBlur(image,5)
    image = cv2.medianBlur(image,5)
    image = cv2.GaussianBlur(image,(5,5),0)
    edges_2 = cv2.Canny(image,100,110)



    # Big circles
    # 30 40 param2=18 param1=50
    circles = hough_big_circles(edges_2) #1 before rotation
    a = circles[0][1][1] - circles[0][0][1]
    b = circles[0][1][0] - circles[0][0][0]
    theta = math.atan(a/b)
    theta = math.degrees(theta)
    # Rotation process
    image = rt.rotate_bound(image,-1 * theta)
    output = rt.rotate_bound(output,-1 * theta)
    edges_2 = rt.rotate_bound(edges_2,-1 * theta)

    edges_2 = cv2.Canny(image,100,110)
    circles = hough_big_circles(edges_2) #2 after rotation

    # Translation process
    rows, cols, _ = image.shape
    c = (200) - (circles[0][0][0])
    d = (1350) - (circles[0][0][1])
    #print 'd:', d
    #print 'c:', c
    M = np.float32([[1,0,(c)/2],[0,1,(d)/2]])

    image = cv2.warpAffine(image,M,(cols,rows))
    output = cv2.warpAffine(image,M,(cols,rows))
    edges_2 = cv2.warpAffine(image,M,(cols,rows))

    cv2.circle(output, (200, 1350 ), 20, (0, 255, 255), 7)

    # Crop images
    output = output[525:1250, 80:1075]
    image =  image[525+100:1250+100, 80:1075]
    edges_2 = edges_2[525:1250, 80:1075]

    edges_2 = cv2.Canny(edges_2,100,110)

    ############################################################################
    # small circles:
    circles = cv2.HoughCircles(edges_2,cv2.HOUGH_GRADIENT,1,minDist=35,
                                param1=110,param2=5,minRadius=5,maxRadius=10)
    ############################################################################

    top_circles = get_top_circles(circles, 55)


    a = top_circles[-1][1] - top_circles[0][1]
    b = top_circles[-1][0] - top_circles[0][0]
    theta = math.atan(a/b)
    theta = math.degrees(theta)
    #print 'theta:', theta

    #Rotation process
    image = rt.rotate_bound(image,-1 * theta)
    output = rt.rotate_bound(output,-1 * theta)
    edges_2 = rt.rotate_bound(edges_2,-1 * theta)

    edges_2 = cv2.GaussianBlur(edges_2,(5,5),0)
    edges_2 = cv2.Canny(edges_2,100,110)
    edges_2 = cv2.medianBlur(edges_2,1)



    d = 0
    circles = cv2.HoughCircles(edges_2,cv2.HOUGH_GRADIENT,1,minDist=35,
                                param1=110,param2=4,minRadius=6,maxRadius=10)

    while(True): # Translation process

        top_circles = get_top_circles(circles,55)
        if len(top_circles) == 0:
            top_circles = get_top_circles(circles,60 + d)

        rows, cols, _ = output.shape
        #print 'top_circles:', len(top_circles)
        if(len(top_circles) == 0):
            #print 'len_circles:', 0
            break
        sum = 0
        for i in range(0,len(top_circles)):
            sum = sum + top_circles[0][1]
        average_y = sum / len(top_circles)
        d = (60) - (average_y)
        m = 10
        s = 7
        x = d

        #print 'd:', d

        if(img_index == -1 or True and False ):
            output_2 = output.copy()
            cv2.line(output_2,(20,60),(930,60),(255,0,0),5)
            output_2 = draw_circles(output_2,circles)
            plt.imshow(output_2)
            plt.title('Before Image'), plt.xticks([]), plt.yticks([])
            plt.show()


        if(abs(d)<3):
            break
        #print 'c:', c

        d_2 = e**(-0.5*(float(d-m)/s)**2) * d
        #print 'd_2 before:', d_2
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
        #print 'd_2:', d_2


        M = np.float32([[1,0,0],[0,1,(d_2)/2]])

        image = cv2.warpAffine(image,M,(cols,rows))
        output = cv2.warpAffine(image,M,(cols,rows))
        edges_2 = cv2.warpAffine(image,M,(cols,rows))
        edges_2 = cv2.Canny(output,100,110)

        for i in range(0,circles.shape[1]): # Translation of detected circles
            circles[0][i][1] = circles[0][i][1] + d_2

        if(img_index == -1 or True and False ):
            output_2 = output.copy()
            cv2.line(output_2,(20,60),(930,60),(255,0,0),5)
            output_2 = draw_circles(output_2,circles)
            plt.imshow(output_2)
            plt.title('After Image'), plt.xticks([]), plt.yticks([])
            plt.show()



    output = np.full(output.shape,255) # White image

    sum = 0
    count = 0
    tmp = []
    for i in range(0,circles.shape[1]): #Normalizing reference_verticle_line
        if circles[0][i][0] < 95:
            sum = sum + circles[0][i][0]
            count = count + 1
            tmp.append(circles[0][i][0])
    average = sum / count
    average = int(average)
    x = average + 30

    #print 'sum:', sum
    #print 'average:', average
    #print 'number_of_left_circles:', index

    cv2.line(output,(x,10),(x,720),(0,0,0),5)
    cv2.line(output,(x+225,10),(x+225,720),(0,0,0),5)
    cv2.line(output,(x+555,10),(x+555,720),(0,0,0),5)

    # middle 2 verticle line
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

    # Top Horizontal line
    cv2.line(output,(20,60),(930,60),(0,0,0),5)

    delta_1 = 60
    dellta_2 = 40
    for i in range(0,14): #Drawing the horizontal lines
        cv2.line(output,(20,(60+delta_1) + i * dellta_2),(930,(60+delta_1) + i * dellta_2 ),(0,0,0),3)


    #print 'len_circles_before:',circles.shape[1]
    circles = remove_unwanted_circles(circles)
    #print 'len_circles_after:',circles.shape[1]
    output = draw_circles(output,circles) # draw on the white image



    """
    output_2 = output.copy()
    cv2.line(output_2,(20,60),(930,60),(255,0,0),5)
    output_2 = draw_circles(output_2,circles)
    plt.imshow(output_2)
    plt.title('After Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    """

    vis = np.concatenate((image, output), axis=1) #Concatenate before and after image
    #vis = output
    cv2.imwrite('out/'+ str(img_name) +'.png',vis)

    answer = track_answer(circles)
    answers[img_name] = answer

    """
    for i in range(0,45):
        try:
            pass
            #print 'Q'+str(i+1)+':',answer['Q'+str(i+1)]
        except:
            pass
    """

with open('answers.txt', 'w') as outfile:
    json.dump(answers, outfile)
