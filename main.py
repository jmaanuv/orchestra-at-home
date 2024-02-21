from contextlib import redirect_stderr

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # holistic model
mp_drawing = mp.solutions.drawing_utils  # holistic drawing

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # converting image to another color scheme so that the media pipe can do its work cause it uses that format
    # image is no longer writable
    image.flags.writeable = False
    # make prediction
    results = model.process(image)
    # image is writable
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(240,240,240), thickness=2, circle_radius=2))
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


cap = cv2.VideoCapture(0)
# set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        # read feed
        ret, frame = cap.read()

        # make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # draw landmarks
        draw_landmarks(image, results)

        # show to screen
        cv2.imshow('ok', image)
        # if key pressed is q then close the camera feed and release the camera resource
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     return np.concatenate([pose, lh, rh])

# print(extract_keypoints(results))
#
# # Path for exported data, numpy arrays
# DATA_PATH = os.path.join('MP_Data')
# # gives error
#
# # Actions that we try to detect
# actions = np.array(['start'])
#
# # Thirty videos worth of data
# no_sequences = 30
#
# # Videos are going to be 30 frames in length
# sequence_length = 30
#
# # Folder start
# start_folder = 30
#
# for action in actions:
#     dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
#     for sequence in range(1, no_sequences + 1):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
#         except:
#             pass