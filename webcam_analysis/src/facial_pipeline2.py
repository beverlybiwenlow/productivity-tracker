# imports
import cv2
import dlib
import numpy as np
from imutils import face_utils
from statistics import mode
import math
import time
import csv

from keras.models import load_model
from scipy.spatial import distance as dist

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

face_landmark_path = './shape_predictor_68_face_landmarks.dat'

# parameters

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])


# Camera internals
size = (720, 1280, 3)

focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# function to get head pose


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# function to check if eyes open


def get_eyes(shape, rgb_image):

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 3

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(rgb_image, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(rgb_image, [rightEyeHull], -1, (0, 255, 0), 1)

    if ear < EYE_AR_THRESH:
        # eyes closed
        return 0.
    else:
        # eyes open
        return 1.


def main():
    showEmotions = True
    showFaceDirection = True
    viz = False

    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    # parameters for loading data and images
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    RATE = 10
    emotions_weight = [5, 5, 5, 5, 5, 5, 1]
    cur_emotions = np.zeros(7)
    cur_gaze = np.zeros(3)
    cur_eyes = 0.

    gaze_csv = open('webcam_data.csv', 'w')
    gaze_csv.write('timestamp,gaze_loc,emotion,eyes_open,eyes_aspect_ratio\n')
    i = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if i == 0:
            print('saving!')
            gaze_val = np.argmax(cur_gaze)
            emotions_val = np.argmax(cur_emotions)
            eyes_val = cur_eyes / RATE >= 0.5
            gaze_csv.write('{},{},{},{},{}\n'.format(
                int(time.time()), gaze_val, emotions_val, eyes_val, cur_eyes / RATE))
            cur_eyes = 0.
            cur_emotions = np.zeros(7)
            cur_gaze = np.zeros(3)
        i = (i + 1) % RATE
        if ret:
            face_rects = detector(frame, 0)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if showFaceDirection:
                # get facial directions
                if len(face_rects) > 0:
                    # get gaze
                    shape = predictor(gray_image, face_rects[0])
                    shape = face_utils.shape_to_np(shape)
                    image_points = np.float32([shape[30], shape[8], shape[36], shape[45], shape[48],
                                               shape[54]])
                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        model_points, image_points, camera_matrix, dist_coeffs)
                    nose_end_point2D, jacobian = cv2.projectPoints(
                        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    for p in image_points:
                        cv2.circle(rgb_image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                    p1 = (int(image_points[0][0]), int(image_points[0][1]))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                    dim_0 = (p1[0] - p2[0]) / size[0]
                    dim_1 = (p1[1] - p2[1]) / size[1]
                    normed_dist = math.sqrt(dim_0**2 + dim_1**2)
                    threshold1 = 0.3
                    threshold2 = 0.2
                    if normed_dist > threshold1:
                        cur_gaze[0] += 1
                        if viz:
                            cv2.line(rgb_image, p1, p2, (0, 255, 0), 10)
                    elif normed_dist > threshold2:
                        cur_gaze[1] += 1
                        if viz:
                            cv2.line(rgb_image, p1, p2, (0, 255, 255), 5)
                    else:
                        cur_gaze[2] += 1
                        if viz:
                            cv2.line(rgb_image, p1, p2, (255, 0, 0), 2)

                    # get eyes
                    cur_eyes += get_eyes(shape, rgb_image)

                else:
                    cur_gaze[0] += 1

            # get emotions
            if showEmotions:
                faces = detect_faces(face_detection, gray_image)
                if len(faces) == 0:
                    cur_emotions[6] += 1
                for face_coordinates in faces[:1]:
                    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                    gray_face = gray_image[y1:y2, x1:x2]
                    try:
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        continue

                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)
                    emotion_prediction = emotion_classifier.predict(gray_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    cur_emotions[emotion_label_arg] += emotions_weight[emotion_label_arg]
                    emotion_text = emotion_labels[emotion_label_arg]
                    emotion_window.append(emotion_text)

                    if len(emotion_window) > frame_window:
                        emotion_window.pop(0)
                    try:
                        emotion_mode = mode(emotion_window)
                    except:
                        continue

                    if emotion_text == 'angry':
                        color = emotion_probability * np.asarray((255, 0, 0))
                    elif emotion_text == 'sad':
                        color = emotion_probability * np.asarray((0, 0, 255))
                    elif emotion_text == 'happy':
                        color = emotion_probability * np.asarray((255, 255, 0))
                    elif emotion_text == 'surprise':
                        color = emotion_probability * np.asarray((0, 255, 255))
                    else:
                        color = emotion_probability * np.asarray((0, 255, 0))

                    color = color.astype(int)
                    color = color.tolist()

                    if viz:
                        draw_bounding_box(face_coordinates, rgb_image, color)
                        draw_text(face_coordinates, rgb_image, emotion_mode,
                                  color, 0, -45, 1, 1)

            if viz:
                cv2.imshow("demo", rgb_image)
            # cv2.waitKey(50)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
