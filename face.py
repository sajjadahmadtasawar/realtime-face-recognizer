import cv2

#trained data for face recognition
face_trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#webcam / custom video
webcam = cv2.VideoCapture(0)

while True:
    #cutting webcam video to frames
    success,frame = webcam.read()

    #changing frames to gray scale
    gray_scaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #getting the face coordinated
    face_coordinates = face_trained_data.detectMultiScale(gray_scaled_frame)
    
    #drawing recangle around the face
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,266,0),2)

    #showing the webcam video
    cv2.imshow('video face recognizer',frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


webcam.release()