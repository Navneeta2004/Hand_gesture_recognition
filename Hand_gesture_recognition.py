import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import collections

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
try:
    model = load_model('mp_hand_gesture')
    with open('gesture.names', 'r') as f:
        classNames = f.read().split('\n')
except FileNotFoundError:
    print("Error: Model or gesture names file not found.")
    exit()

# Initialize the webcam
cap = cv2.VideoCapture(0)

landmarks = []
gesture_history = collections.deque(maxlen=10)
confidence_threshold = 0.8

while True:
    _, frame = cap.read()

    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    className = 'Unknown'

    landmarks = []
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    if landmarks:
        prediction = model.predict([landmarks])
        if np.max(prediction) > confidence_threshold:
            classID = np.argmax(prediction)
            className = classNames[classID]
        else:
            className = 'Unknown'

    gesture_history.append(className)

    most_common_gesture = collections.Counter(gesture_history).most_common(1)
    if most_common_gesture:
        className = most_common_gesture[0][0]

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
