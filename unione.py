import cv2
import numpy as np
import math
import mediapipe as mp
import tensorflow as tf
#tf.keras.models.load_model

# funzione per dimensionare correctly
def resize(image,x,y):

  DESIRED_HEIGHT = x
  DESIRED_WIDTH = y
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

  return img

# inizializzo mediapipe per hands
mp_hands = mp.solutions.hands                                                    #è un ogeetto contenente la mano
hands = mp_hands.Hands(max_num_hands = 1, min_detection_confidence = 0.7)        # così configuriamo il modello della mano (piglia una mano per frame)
mp_draw = mp.solutions.drawing_utils                                             #disegna i keypoints (non dobbiamo farlo a mano noi)

# inizializzo mediapipe per pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# inizializzo tf
model = tf.keras.models.load_model('/home/fausto/MediaPipe/mp_env/hand-gesture-recognition-code/mp_hand_gesture')

# carichiamo i nomi dei gest presenti nel modello
f = open('/home/fausto/MediaPipe/mp_env/hand-gesture-recognition-code/gesture.names', 'r')
class_names = f.read().split('\n')
f.close()


def check_gesto(frame, class_names, c_n):
  
  x,y,c = frame.shape

  # invertiamo da bgr a rgb
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # otteniamo i landmark
  result = hands.process(image)                                     # mettiamo in formato rgb perchè mediapipe works con rgb

  class_name = ''

  if(result.multi_hand_landmarks):                                  # se c'è una mano e becca i landmarks allora famo i check da fare
        landmarks = []
        for handslms in result.multi_hand_landmarks :
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            #finally disegniamo i landmark sul frame
            mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)


        # usiamo il modello per predirre il gesto fatto dalla mano
        prediction = model.predict([landmarks])                         #prende la lista di landmarks e restituisce un array contenente la predizione per ogni landmark
        #print(prediction)   
        classID = np.argmax(prediction)                                 #restituisce l'indice del valore max nella lista
        class_name = class_names[classID]                     

        #giriamo il frame in modo che sia a specchio
        frame = cv2.flip(frame,1)

        # e scriviamo la predizione sul frame
        cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)   
        cv2.imshow("output", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            return False

        if class_name == c_n:
            return True
        else: 
            return False
        
  else:                                                           #se la mano non ci sta famo solo vedere il video
     cv2.imshow("output", cv2.flip(frame,1))
     if cv2.waitKey(5) & 0xFF == 27:
        return False
    
     return False


progress = 0                              # serve per dopo

cap = cv2.VideoCapture(0)                                           #0 è l'ID della videocamera (usando quella del computer, se vuoi usarne una different devi cambiare quel value)
while cap.isOpened():                                               
    success, frame = cap.read()
    
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    
    
    x,y,c = frame.shape

    # prima di prendere lo scheletro della persona aspettiamo 2 gesti di riconoscimento: 'peace', 'thumb up'
    check = False
    
    if(progress == 0):                                            # ancora non sono stati riconosciuti gesti
        check = check_gesto(frame, class_names, 'peace')
        if check:
            spunta = cv2.imread("/home/fausto/MediaPipe/mp_env/images/spunta_gialla.png", cv2.IMREAD_COLOR)
            cv2.imshow("output", resize(spunta,x,y))
            cv2.waitKey(0)
            progress = 1
        

    elif(progress == 1):                                          # è stato riconosciuto 'peace' deve essere riconosciuto 'thumb up'
        check = check_gesto(frame, class_names,'thumbs up')
        if check:
            spunta = cv2.imread("/home/fausto/MediaPipe/mp_env/images/spunta_verde.jpg", cv2.IMREAD_COLOR)
            cv2.imshow("output", resize(spunta,x,y))
            cv2.waitKey(0)
            progress = 2
        

    elif(progress == 2):     
        # gesti riconosciuti annamo a pigliare lo scheletro
        land_nose = [0,0]                                          #serve per checkare di stare seguendo la pesrona giusta
        first_frame= True                                      
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        if results.pose_landmarks != None:
            land1= [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x , results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y ]

            if first_frame:                                   #al primo frame skippo 
                land_nose = land1
                first_frame = False

            else:

                diff1 = land1[0] - land_nose[0]                 #calcolo la differenza tra le nose coordinates x e y dei 2 frame successivi
                diff2 = land1[1] - land_nose[1] 


                if(-0.07< diff1 < 0.07 and -0.07< diff2 < 0.07 ):  #e se questa differenza è eccessiva vuol dire che mi sono perso la persona --> torno a chiedere il gesto di riconoscimento
                    print(
                    f'Nose coordinates:\n'
                    f'frame precedente : {land_nose},\n'
                    f'frame corrente : {land1},\n '
                    f'diff tra frame : {[diff1 , diff2]}')
                else:
                    print('\n \n DAMN ME LO SONO PERSO \n \n')
                    print(
                    f'Nose coordinates:\n'
                    f'frame precedente : {land_nose},\n'
                    f'frame corrente : {land1},\n '
                    f'diff tra frame : {[diff1 , diff2]}'
                    )
                    progress = 0
                    continue

        else:
            progress = 0
            continue
            

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('output', cv2.flip(image, 1))
    
        if cv2.waitKey(5) & 0xFF == 27:
            break
        

cap.release()
cv2.destroyAllWindows()


