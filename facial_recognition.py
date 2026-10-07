import cv2
import face_recognition

existing_encodings = []
encoded_labels = ["Elon Musk","Sundar Pichai","David Goggins"]

print("Encoding Image 1...")
img = cv2.imread("assets/images/elon musk.webp")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings (rgb_img)[0]  
existing_encodings.append(img_encoding)

print("Encoding Image 2...")
img0 = cv2.imread("assets/images/sundar_pichai.webp")
rgb_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
img_encoding0 = face_recognition.face_encodings (rgb_img0)[0]  
existing_encodings.append(img_encoding0)

print("Encoding Image 3...")
img1 = cv2.imread("assets/images/goggins.webp")
rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img_encoding1 = face_recognition.face_encodings (rgb_img1)[0]  
existing_encodings.append(img_encoding1)

def check_facial_matches(clicked_filaname):
    return_item = []

    clicked_filaname = str(clicked_filaname)
    img = cv2.imread(clicked_filaname)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        img_encoding = face_recognition.face_encodings(rgb_img)[0]

        for index, encoding in enumerate(existing_encodings):
            result = face_recognition.compare_faces([img_encoding],encoding)
            result = str(result[0])
            if result == "True":
                return_item.append(encoded_labels[index]) 
            else:
                continue
    except IndexError as e:
        return_item.append(None)
    
    return return_item