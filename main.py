import cv2
import numpy as np


car_cas = cv2.CascadeClassifier('cars.xml')
car2_cas = cv2.CascadeClassifier('cars2.xml')
sign_cas = cv2.CascadeClassifier('sign.xml')
right_cas = cv2.CascadeClassifier('right-sign.xml')
left_cas = cv2.CascadeClassifier('left-sign.xml')
trafic_cas = cv2.CascadeClassifier('traffic_light.xml')
stop_cas = cv2.CascadeClassifier('stop_sign.xml')
pedestrain_cas = cv2.CascadeClassifier('pedestrian.xml')


#from lanes import *

def display_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1, y1), (x2,y2), (255, 0, 0), 10)
	return line_image	



def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
	[(100, height), (500, height), (320,200)]
	])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, polygons, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image


cap = cv2.VideoCapture(0)



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    lane_image = np.copy(frame)
    gray =cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny_image =cv2.Canny(blur, 50, 150)
    cropped_image = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    #averaged_lines = average_slop_intercept(lane_image, lines)
    line_image = display_lines(lane_image, lines)

    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    # Detects cars of different sizes in the input image 
    cars = car_cas.detectMultiScale(gray, 1.3, 5) 

    for (x,y,w,h) in cars: 
        # To draw a rectangle in a car 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(frame, "car", (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2) 
        # roi_gray = gray[y:y+h, x:x+w] 
        # roi_color = frame[y:y+h, x:x+w] 

        # Detects car of different sizes in the input image 
    cars2 = car2_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in car 
    for (ex,ey,ew,eh) in cars2: 
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        cv2.putText(frame, "car", (ex,ey-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
	
    signs = sign_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in sign 
    for (ax,ay,aw,ah) in signs: 
        cv2.rectangle(frame,(ax,ay),(ax+aw,ay+ah),(0,127,255),2)
        cv2.putText(frame, "sign", (ax,ay-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    right = right_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in sign 
    for (bx,by,bw,bh) in right: 
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(0,127,127),2)
        cv2.putText(frame, "right", (bx,by-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    left = left_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in left sign 
    for (cx,cy,cw,ch) in left: 
        cv2.rectangle(frame,(cx,cy),(cx+cw,cy+ch),(0,127,127),2)
        cv2.putText(frame, "left", (cx,cy-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    trafic = trafic_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in trafic 
    for (dx,dy,dw,dh) in trafic: 
        cv2.rectangle(frame,(dx,dy),(dx+dw,dy+dh),(0,127,127),2)
        cv2.putText(frame, "trafic", (dx,dy-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    stop = stop_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in stop sign 
    for (dx,dy,dw,dh) in stop: 
        cv2.rectangle(frame,(dx,dy),(dx+dw,dy+dh),(0,127,127),2)
        cv2.putText(frame, "stop", (dx,dy-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    pedestrain = pedestrain_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in pedestrain 
    for (fx,fy,fw,fh) in pedestrain: 
        cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,127,127),2)
        cv2.putText(frame, "pedestrain", (fx,fy-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    
    # Display an image in a window 
    cv2.imshow('frame',frame) 


    # cv2.imshow("line",line_image)
    cv2.imshow("linea",combo_image)
    # cv2.imshow('frame',frame)
    # cv2.imshow('result', blur)
    # cv2.imshow('canny', canny_image)
    # cv2.imshow('croped', cropped_image)

    #plt.imshow(canny)
    #plt.show()
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
