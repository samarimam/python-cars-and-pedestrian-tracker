import cv2

# IMPORTANT NOTE USE WEBCAM INSTAED OF VIDEO FOR USING YOUR LAPTOP OR PHONE FRONT CAMERA AND TRACKING CARS AND PEDESTRIANS 

# img_file = 'image.jpg'
video = cv2.VideoCapture('videofinal.webm')

# our pre-trained car classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(car_tracker_file)

pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

    # display the image with faces spotted
    cv2.imshow('Samar car and pedestrian detector', frame)

    # wait for a key
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break
video.release()
# create opencv image
# img = cv2.imread(img_file)

# # convert to grayscale
# black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # create car classsifier
# car_tracker = cv2.CascadeClassifier(classifier_file)

# # detect cars
# cars = car_tracker.detectMultiScale(black_n_white)

# # draw rectangles
# for (x, y, w, h) in cars:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)


# # display the image with faces spotted
# cv2.imshow('Samar car detector', img)

# # wait for a key
# cv2.waitKey()

# print("success")