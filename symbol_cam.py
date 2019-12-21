import cv2

car_cas = cv2.CascadeClassifier('cars.xml')
car2_cas = cv2.CascadeClassifier('cars2.xml')
sign_cas = cv2.CascadeClassifier('sign.xml')
right_cas = cv2.CascadeClassifier('right-sign.xml')
left_cas = cv2.CascadeClassifier('left-sign.xml')
trafic_cas = cv2.CascadeClassifier('traffic_light.xml')
stop_cas = cv2.CascadeClassifier('stop_sign.xml')
pedestrain_cas = cv2.CascadeClassifier('pedestrian.xml')
# capture frames from a camera 
cap = cv2.VideoCapture(0) 

# loop runs if capturing has been initialized. =
while 1: 

    # reads frames from a camera 
    ret, img = cap.read() 

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Detects cars of different sizes in the input image 
    cars = car_cas.detectMultiScale(gray, 1.3, 5) 

    for (x,y,w,h) in cars: 
        # To draw a rectangle in a car 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(img, "car", (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2) 
        # roi_gray = gray[y:y+h, x:x+w] 
        # roi_color = img[y:y+h, x:x+w] 

        # Detects car of different sizes in the input image 
    cars2 = car2_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in car 
    for (ex,ey,ew,eh) in cars2: 
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        cv2.putText(img, "car", (ex,ey-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
	
    signs = sign_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in sign 
    for (ax,ay,aw,ah) in signs: 
        cv2.rectangle(img,(ax,ay),(ax+aw,ay+ah),(0,127,255),2)
        cv2.putText(img, "sign", (ax,ay-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    right = right_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in sign 
    for (bx,by,bw,bh) in right: 
        cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(0,127,127),2)
        cv2.putText(img, "right", (bx,by-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    left = left_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in left sign 
    for (cx,cy,cw,ch) in left: 
        cv2.rectangle(img,(cx,cy),(cx+cw,cy+ch),(0,127,127),2)
        cv2.putText(img, "left", (cx,cy-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    trafic = trafic_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in trafic 
    for (dx,dy,dw,dh) in trafic: 
        cv2.rectangle(img,(dx,dy),(dx+dw,dy+dh),(0,127,127),2)
        cv2.putText(img, "trafic", (dx,dy-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    stop = stop_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in stop sign 
    for (dx,dy,dw,dh) in stop: 
        cv2.rectangle(img,(dx,dy),(dx+dw,dy+dh),(0,127,127),2)
        cv2.putText(img, "stop", (dx,dy-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    pedestrain = pedestrain_cas.detectMultiScale(gray, 1.3, 5) 

        #To draw a rectangle in pedestrain 
    for (fx,fy,fw,fh) in pedestrain: 
        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(0,127,127),2)
        cv2.putText(img, "pedestrain", (fx,fy-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)

    
    # Display an image in a window 
    cv2.imshow('img',img) 

    # Wait for Esc key to stop 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 
