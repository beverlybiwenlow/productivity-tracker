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


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def main():
    showEmotions = True
    showFaceDirection = True
    viz = True

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
    emotions_weight = [2, 2, 2, 2, 2, 2, 1]
    cur_emotions = np.zeros(7)
    cur_gaze = np.zeros(3)

    gaze_csv = open('webcam_data.csv', 'w')

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if i == 0:
            print('saving!')
            gaze_rate = np.argmax(cur_gaze)
            emotions_rate = np.argmax(cur_emotions)
            gaze_csv.write('{},{},{}\n'.format(int(time.time()), gaze_rate, emotions_rate))
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
                    shape = predictor(frame, face_rects[0])
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
                        cur_gaze[2] += 1
                        if viz:
                            cv2.line(rgb_image, p1, p2, (0, 255, 0), 10)
                    elif normed_dist > threshold2:
                        cur_gaze[1] += 1
                        if viz:
                            cv2.line(rgb_image, p1, p2, (0, 255, 255), 5)
                    else:
                        cur_gaze[0] += 1
                        if viz:
                            cv2.line(rgb_image, p1, p2, (255, 0, 0), 2)
                else:
                    cur_gaze[2] += 1

            # get emotions
            if showEmotions:
                faces = detect_faces(face_detection, gray_image)
                if len(faces) == 0:
                    cur_emotions[6] += 1
                for face_coordinates in faces:
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
            cv2.waitKey(50)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
