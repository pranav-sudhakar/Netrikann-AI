import math
import cv2
import tts
import facial_recognition
import object_recognition
import text_scrutinizer
import speech_transcription
import wikipedia_nk as wiki
import audio_player
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

filename = "clicked_img.jpg"

#-----------------------------------------------------------
from ultralytics import YOLO
model_track = YOLO('yolov8n.pt')

with open('coco.names', 'r') as file:
    lines = file.readlines()

classes = []
print(classes)

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
            audio_player.play_clip("assets/cam_issue.mp3")

        cv2.imwrite(filename,frames)
        counter += 1
    print("Clicked Image")
#-----------------------------------------------------------

requested_class_id = 0
object_requested = None
while True:
    features = ["Facial Recognition","Initiate Voice Assistant","Text Scrutinizer","Object Tracking","Object Recognition"]
    print(features)
    input_line1 = input("Select ID: ")
    try:
        input_line1 = int(input_line1)
    except ValueError as e:
        break
    
    if input_line1 == 4:
        audio_player.play_clip("assets/clips/find_what.mp3")
        while True:
            usr_response = speech_transcription.transcribe_audio()
            if usr_response == False:
                audio_player.play_clip("assets/clips/come_again.mp3")
                continue
            else:
                object_requested = str(usr_response.lower())
                for index, label in enumerate(classes):
                    if str(label.lower()) == object_requested:
                        requested_class_id = index
                        break
                break
        print(requested_class_id,object_requested)

        while True:
            xcord = 0
            ycord = 0
            ret, frames = cam.read()
            if ret == False:
                audio_player.play_clip("assets/clips/cam_issue.mp3")
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
                    if object_class == 0:
                        pass
                    else:
                        object_class += 1
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
                    audio_player.play_clip("assets/clips/turn_left.mp3")
                    continue
                if xcord < 225:
                    audio_player.play_clip("assets/clips/turn_right.mp3")
                    continue
                if ycord > 305:
                    audio_player.play_clip("assets/clips/look_down.mp3")
                    continue
                if ycord < 145:
                    audio_player.play_clip("assets/clips/look_up.mp3")                    
                    continue

    elif input_line1 == 1:
        click_image(filename)
        matches = facial_recognition.check_facial_matches(filename)
        if len(matches) == 0:
            audio_player.play_clip("assets/clips/no_face_recog.mp3")
        else:
            for match in matches:
                if match == None:
                    audio_player.play_clip("assets/clips/no_face.mp3")
                else:
                    tts.speak(f"Match found with {match}")

    elif input_line1 == 5:
        click_image(filename)
        all_objects = object_recognition.recognize_images(filename)
        if len(all_objects) == 0:
            audio_player.play_clip("assets/clips/no_objects.mp3")
            continue
        else:
            audio_player.play_clip("assets/clips/following_objects.mp3")
        for object in all_objects:
            tts.speak(object)

    elif input_line1 == 3:
        click_image(filename)
        result = text_scrutinizer.scrutinize_image("assets/images/image1.jpg")
        print(result)
        audio_player.play_clip("assets/clips/more_info.mp3")
        while True:
            usr_response = speech_transcription.transcribe_audio()
            if usr_response == False:
                audio_player.play_clip("assets/clips/come_again.mp3")
                continue
            else:
                usr_response = str(usr_response)
                if "Yes" in usr_response or "yes" in usr_response :
                    wiki_result = wiki.search_wiki(result)
                    tts.speak(wiki_result)
                    break
                else:
                    audio_player.play_clip("assets/clips/okay.mp3")
                    break