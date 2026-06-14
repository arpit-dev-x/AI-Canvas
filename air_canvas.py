import os
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
model_path=os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

options=HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)

detector=HandLandmarker.create_from_options(options)

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

canvas=None
prev_x, prev_y=0,0

while True:
    ret, frame=cap.read()
    frame=cv2.flip(frame, 1) # To get the mirror image horizontally(to make the visual look same oriented)

    if canvas is None:
        canvas=np.zeros_like(frame) # zeros_like func is used to create a  blank image of the size of webcam

    # Convert the color from BGR to RGB
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to mediapipe
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result=detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h, w, _=frame.shape

            # index-finger tip=landmark 8
            x=int(hand[8].x*w)
            y=int(hand[8].y*h)
            
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            if prev_x==0 and prev_y==0:
                prev_x, prev_y=x,y
            
            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 5)
            prev_x, prev_y=x,y
    else:
        prev_x, prev_y=0,0
    
    # Merge canvas
    frame=cv2.add(frame, canvas)

    cv2.imshow("Air Draw", frame)

    key=cv2.waitKey(1)
    if key==ord("c"):
        canvas=np.zeros_like(frame)
    if key==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()