import math
import cv2
import facial_recognition
import object_recognition
import text_scrutinizer
import speech_recognition_file
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

filename = "clicked_img.jpg"

#-----------------------------------------------------------
from ultralytics import YOLO
model_track = YOLO('yolov8n.pt')

with open('coco.names', 'r') as file:
    lines = file.readlines()

classes = []

for line in lines:
    class_name = line.strip()
    classes.append(class_name)

def midpoint(coords):
    cx = (int(coords[0]) + int(coords[2]))/2
    cy = (int(coords[1]) + int(coords[2]))/2
    return int(cx),int(cy)

center_coords = (325,225)

def click_image(filename):
    print("Clicking Image")
    counter = 0
    while True:
        if counter >= 1:
            break
        else:
            pass

        ret,frames = cam.read()
        if ret is True:
            pass
        else:
            print("Cam failed to load")

        cv2.imwrite(filename,frames)
        counter += 1
    print("Clicked Image")
#-----------------------------------------------------------

while True:
    features = ["Facial Recognition","Initiate Voice Assistant","Text Scrutinizer","Object Tracking","Object Recognition"]
    print(features)
    input_line1 = input("Select ID: ")
    try:
        input_line1 = int(input_line1)
    except ValueError as e:
        break
    
    if input_line1 == 4:
        print("Object Tracking It is..")
        
        while True:
            xcord = 0
            ycord = 0
            ret, frames = cam.read()
            if ret == False:
                print("Cam failed to load")
            else:
                pass
            results = model_track(frames,stream=True)

            cv2.line(frames, (325,0), (325,600), (0,255,247), 3)  
            cv2.line(frames, (0,225), (650,225), (0,255,247), 3)
            cv2.circle(frames, (325,225), 80, (0,255,0), 2)

            for item in results:
                objects = item.boxes
                for object in objects:
                    object_class = int(object.cls[0])
                    conf_value = float(math.ceil((object.conf[0]*100)) / 100)
                    if object_class == 0 and conf_value > 0.50:
                        x1,y1,x2,y2 = object.xyxy[0]
                        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                        coords = [x1,y1,x2,y2]
                        cv2.rectangle(frames, (x1,y1),(x2,y2),(255,0,0),3)

                        #conf_value = str(math.ceil((object.conf[0]*100)) / 100)
                        cv2.putText(frames, f'{conf_value} {classes[int(object.cls[0])]}', (x1,y1),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

                        xcord,ycord = midpoint(coords)
                        cv2.circle(frames,(xcord,ycord),10,(0,0,255),7)
                        print(f"Confidence Value: {conf_value} and {type(conf_value)}")
                    else:
                        print(f"Class ID: {int(object.cls[0])}")
                        print(f"Confidence Value: {conf_value} and {type(conf_value)}")


                cv2.imshow("frames",frames)
                if cv2.waitKey(1) == ord('q'):
                    break

                if xcord == 0 and ycord == 0:
                    continue
                if xcord > 405:
                    print("Turn Left")
                    continue
                if xcord < 225:
                    print("Turn Right")
                    continue
                if ycord > 305:
                    print("Look down")
                    continue
                if ycord < 145:
                    print("Look up")
                    continue

    elif input_line1 == 1:
        click_image(filename)
        matches = facial_recognition.check_facial_matches(filename)
        if len(matches) == 0:
            print("No matches found")
        else:
            for match in matches:
                if match == None:
                    print("No face detected")
                else:
                    print(f"Match found with {match}")

    elif input_line1 == 5:
        click_image(filename)
        all_objects = object_recognition.recognize_images(filename)
        for object in all_objects:
            print(object)

    elif input_line1 == 3:
        click_image(filename)
        result = text_scrutinizer.scrutinize_image("assets/image1.jpg")
        print(result)