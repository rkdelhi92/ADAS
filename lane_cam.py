import cv2
import numpy as np
import matplotlib.pyplot as plt

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
