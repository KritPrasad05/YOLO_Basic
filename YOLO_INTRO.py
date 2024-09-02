from ultralytics import YOLO 
import cv2
import cvzone
import math

model = YOLO("yolov8l.pt")

className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat","dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"]

#To detect object from image files
#Uncomment the below code for image detection
'''
results = model("",show=True)#Insert the image file location in the Quates
cv2.waitKey(0)
'''

#To detect object from Webcam or Videos
#Uncomment the below code for webcam or video file detection
'''
try:
    #Uncomment if to detect object from webcam
    #cap = cv2.VideoCapture(0)
    #Uncomment if to detect object from video file
    #cap = cv2.VideoCapture(r"")#Insert the file location inside the Quotes
    
    cap.set(3,1080)
    cap.set(4,720)
    while True:
        success, img = cap.read()

        if not success or img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Failed to capture a valid image.")
            break
        img = cv2.flip(img,1)
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                #bounding box Dimensions
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                #Simple bounding box
                #cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
    
                #boundry box with corner 
                w, h = x2-x1,y2-y1
                cvzone.cornerRect(img,bbox=(x1,y1,w,h),l=15)
    
                #set up the confidence text and the classification
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{className[cls]}{conf}', (max(0,x1),max(35,y1)),scale= 1,thickness=1)
                
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

finally:
    # Release the camera and destroy all OpenCV windows when done
    if cap:
        cap.release()
    cv2.destroyAllWindows()
'''