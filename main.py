import sys
import os
from pickletools import read_bytes1
import eel
import numpy as np
import base64
from io import BytesIO
import cv2

from PIL import Image, ImageDraw
import subprocess
import mediapipe as mp
from deepface import DeepFace
import face_recognition
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
tessdata_dir_config =  "C:/Program Files (x86)/Tesseract-OCR/tessdata"


eel.init('web')


@eel.expose
def to_python(pic_url):


    image = pic_url  # raw data with base64 encoding
    
    imgString = base64.b64decode(image)
    nparr = np.frombuffer(imgString,dtype = np.uint8) 
    image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    cv2.imwrite('img/new.jpg', image)


    
  
   
    get_face_img = face_recognition.load_image_file('img/new.jpg')
    get_face_location = face_recognition.face_locations(get_face_img)

    print(get_face_location)

    pil_img1 = Image.fromarray(get_face_img)
    draw1 = ImageDraw.Draw(pil_img1)
    

    for (top, right, bottom, left) in get_face_location:

        draw1.rectangle(((left,top), (right,bottom)), outline=(255,255,0), width = 4)


     

        pil_img1.save("C:/Users/tsvet/OneDrive/Документы/diplom_new/web/image-2.jpg")
    
    if True:
        Result_dict = DeepFace.analyze(img_path = "img/new.jpg", actions = ['age', 'gender', 'race', 'emotion'])
        Age = Result_dict.get("age")
        Gender = Result_dict.get("gender")
        Race = Result_dict.get("dominant_race")
        Emotion = Result_dict.get("dominant_emotion")

        print(Result_dict)
        print (Age)
        print (Gender)
        
        eel.age_js(Age, Gender, Race, Emotion)
    else:
        pass

    


@eel.expose
def to_python_2(pic_url_2):

    image_2 = pic_url_2
    
    # Декодируем из b64
    imgString = base64.b64decode(image_2)
    nparr_2 = np.frombuffer(imgString,dtype = np.uint8) 
    image_2 = cv2.imdecode(nparr_2,cv2.IMREAD_COLOR)
    cv2.imwrite('img/new_2.jpg', image_2)
      
    text = pytesseract.image_to_string(Image.open('img/new_2.jpg'), lang='rus')  
    f1 = open('passport.txt', 'w')
    f1.write(text)

   
    get_face_img_2 = face_recognition.load_image_file('img/new_2.jpg')
    get_face_location_2 = face_recognition.face_locations(get_face_img_2)
    print(get_face_location_2)
    

    pil_img_2 = Image.fromarray(get_face_img_2)
    draw_2 = ImageDraw.Draw(pil_img_2)
  
    for (top, right, bottom, left) in get_face_location_2:

        draw_2.rectangle(((left,top), (right,bottom)), outline=(255,0,0), width = 4)

         

        pil_img_2.save("C:/Users/tsvet/OneDrive/Документы/diplom_new/web/image-3.jpg")



@eel.expose
def compare_faces(a , b):

    img1 = face_recognition.load_image_file(a)
    img1_encodings = face_recognition.face_encodings(img1)
    print(img1_encodings)
    
    img2 = face_recognition.load_image_file(b)
    img2_encodings = face_recognition.face_encodings(img2)
    

    result = face_recognition.compare_faces (img1_encodings[0], img2_encodings)

    if result[0]:
        eel.js_alarm()
       

    else:
        eel.js_alarm_2()
        
@eel.expose
def file_reading():
    
    subprocess.call(['notepad.exe', 'demo.txt'])
   

    
@eel.expose  
def analyze_python(text):
    print (text)
 
    



@eel.expose
def video_capture():

    
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing_styles = mp.solutions.drawing_styles


   
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
           
            continue

            
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

       
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
  
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break
    
        
    
    cap.release() 
    cv2.destroyAllWindows()   
    


@eel.expose
def face_recognition_cap():


  
    video_capture = cv2.VideoCapture(0)

    Nikita_image = face_recognition.load_image_file("Nikita Tsvetkov.jpg")
    Nikita_face_encoding = face_recognition.face_encodings(Nikita_image)[0]

    Irina_image = face_recognition.load_image_file("Irina.jpg")
    Irina_face_encoding = face_recognition.face_encodings(Irina_image)[0]


    known_face_encodings = [
        
        Nikita_face_encoding,
        Irina_face_encoding

    ]
    known_face_names = [
        "Nikita_Tsvetkov",
        "Irina Tsvetkova"
    ]

  
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
      
        ret, frame = video_capture.read()

 
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
          
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
               
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


       
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

           
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

       
        cv2.imshow('Video', frame)

        
        if cv2.waitKey(5) & 0xFF == 27:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
    video_capture.release()
    cv2.destroyAllWindows()


@eel.expose
def to_python_3(pic_url_3):

    image_3 = pic_url_3
    
    # Декодируем из b64
    imgString = base64.b64decode(image_3)
    nparr_3 = np.frombuffer(imgString,dtype = np.uint8) 
    image_3 = cv2.imdecode(nparr_3,cv2.IMREAD_COLOR)
    cv2.imwrite('img/new_3.jpg', image_3)
   
    if not os.path.exists("dataset"):
        print ("error")
        sys.exit()

    known_encodings = []


    my_img = face_recognition.load_image_file("img/new_3.jpg")
    my_enc = face_recognition.face_encodings(my_img) # вычисляю энкодинги лица из датасета

  

    images = os.listdir("dataset")

    print(images)
    
    
    

    for(i, image) in enumerate(images):
        print(f"[+] processing img {i + 1}/{len(images)}")

        face_img = face_recognition.load_image_file(f"dataset/{image}")
        face_enc = face_recognition.face_encodings(face_img) # вычисляю энкодинги лица из датасета
        
        result = face_recognition.compare_faces(my_enc, face_enc[0])

        print(result)

        if result[0]:
            
            print("same person!")
            image_name = str(i)+'.jpg'
            pil_img1 = Image.fromarray(face_img)
            pil_img1.save("C:/Users/tsvet/OneDrive/Документы/diplom_new/new_photos/"+image_name)
        
        else:
            pass
        

@eel.expose
def show_text_python():

    subprocess.call(['notepad.exe', 'passport.txt'])



eel.start('index.html', size = (1185, 700))



