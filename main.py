import numpy as np
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from pygame import mixer
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

model2 = keras.models.load_model("modelf.h5")
mixer.init()
mixer.music.load("alarm.mp3")
mixer.music.set_volume(0.7)

model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner

])

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion


# To calculate eye aspect ratio
def eye_aspect_ratio(eye):
    vertical1 = dist.euclidean(eye[1], eye[5])
    vertical2 = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


# To play alarm if drowsy
def play_music(flag):
    global cur_status
    if (flag == 1 and cur_status == 0):
        print('playing')
        cur_status = 1
        mixer.music.play()
    elif flag == 0 and cur_status == 1:
        print('stop')
        cur_status = 0
        mixer.music.stop()


# To check if yawning
def check_yawn(landmarks):
    upper_lips_mean = 0
    lower_lips_mean = 0
    for i in range(50, 53):
        (x, y) = landmarks[i]
        upper_lips_mean += y
    for i in range(61, 64):
        (x, y) = landmarks[i]
        upper_lips_mean += y
    for i in range(56, 59):
        (x, y) = landmarks[i]
        lower_lips_mean += y
    for i in range(65, 68):
        (x, y) = landmarks[i]
        lower_lips_mean += y
    lip_distance = upper_lips_mean / 6 - lower_lips_mean / 6
    lip_distance = lip_distance * (-1)
    print(lip_distance)
    if lip_distance > 22:
        return (1, lip_distance)
    else:
        return (0, lip_distance)


# To check if drowsy
def check_status(landmarks):
    left_eye = []
    right_eye = []
    for i in range(36, 42):
        left_eye.append(landmarks[i])
    for i in range(42, 48):
        right_eye.append(landmarks[i])

    if (eye_aspect_ratio(left_eye) < 0.25 or eye_aspect_ratio(right_eye) < 0.25):
        return 1
    else:
        return 0


def check_eyes(frame, landmarks):
    img = frame[landmarks[38][1] - 5:landmarks[41][1] + 5, landmarks[36][0] - 5:landmarks[39][0] + 5]
    cv2.imshow('fra', img)
    cv2.waitKey(1)
    x = cv2.resize(img, (24, 24))
    x = np.expand_dims(x, axis=0)
    ans = model2.predict(x)
    ans = np.argmax(ans)
    return abs(1 - ans)


def cal_angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


def check_orientation(camera_matrix, img_points, im):
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(im, p1, p2, (255, 0, 0), 2)
    print(p1, p2)
    cv2.imshow('ori', im)
    cv2.waitKey(1)
    v1 = []
    v2 = []
    v3 = []
    v1.append(p2[0] - p1[0])
    v1.append(p2[1] - p1[1])
    v2.append(0)
    v2.append(p1[0])
    v3.append(p1[1])
    v3.append(0)
    vertical_angle = cal_angle(v1, v2)
    horizontal_angle = cal_angle(v1, v3)
    print('horizontal angle:' + str(cal_angle(v1, v2)))
    print('vertical angle:' + str(cal_angle(v1, v3)))
    if (horizontal_angle >= 3 or horizontal_angle < 0.1 or vertical_angle >= 3 or vertical_angle <= 1):
        return 1
    else:
        return 0


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
eye_closed_count = 0
eye_open_count = 0
cur_status = 0
status = 'Active'
yawn = 0
ear = 0
lip_distance = 0
sleeping = 0
yawn_count = 0

while (True):
    ret, frame = cap.read()

    # gray scale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    eye_detection = frame.copy()
    lip_detection = frame.copy()
    face_detection = frame.copy()

    print('$$$$$', len(faces))

    for face in faces:
        left_coordinate = face.left()
        top_coordinate = face.top()
        right_coordinate = face.right()
        bottom_coordinate = face.bottom()
        cv2.rectangle(face_detection, (left_coordinate, top_coordinate), (right_coordinate, bottom_coordinate),
                      (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        for i in range(0, 68):
            (x, y) = landmarks[i]
            cv2.circle(eye_detection, (x, y), 1, (0, 255, 0), -1)
        for i in range(36, 48):
            (x, y) = landmarks[i]
            cv2.circle(eye_detection, (x, y), 1, (0, 255, 0), -1)

        for i in range(48, 67):
            (x, y) = landmarks[i]
            cv2.circle(lip_detection, (x, y), 1, (0, 255, 0), -1)
        eye = []
        for i in range(36, 48):
            eye.append(landmarks[i])

        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],  # Chin
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corne
            landmarks[48],  # Left Mouth corner
            landmarks[54]  # Right mouth corner
        ], dtype="double")
        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        ear = eye_aspect_ratio(eye)
        sleeping = check_status(landmarks)
        yawn, lip_distance = check_yawn(landmarks)
        orientation = check_orientation(camera_matrix, image_points, frame)

        # if ear is greater than 0.25 and person is not yawning
        if (sleeping == 0 and yawn == 0 and orientation == 0):
            if eye_open_count >= 4:
                eye_open_count += 1
                eye_closed_count = 0
                status = 'Active'
            else:
                eye_open_count += 1

        # if eye is closed for sufficient time(7 frames) or person is yawning
        else:
            if (eye_closed_count >= 7):
                eye_closed_count += 1
                eye_open_count = 0
                status = 'sleepy'
            else:
                eye_closed_count += 1

    # no faces detected
    if (status == 'Active'):
        print('active')
        play_music(0)
    else:
        print('sleep')
        play_music(1)

    print(cur_status)
    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    face_detection = cv2.putText(face_detection, status, (00, 185), font, 1, color, 2, cv2.LINE_AA, False)
    eye_detection = cv2.putText(eye_detection, str(ear), (00, 185), font, 1, color, 2, cv2.LINE_AA, False)
    lip_detection = cv2.putText(lip_detection, str(lip_distance), (00, 185), font, 1, color, 2, cv2.LINE_AA, False)

    cv2.imshow('face_detection', face_detection)
    cv2.imshow('lip_detection', lip_detection)
    cv2.imshow('eye_detection', eye_detection)

    key = cv2.waitKey(1)
    if (key == 113):
        play_music(0)
        while (cv2.waitKey(1) != 113):
            print('System Paused')

    if (len(faces) == 0):
        if (eye_closed_count >= 7):
            eye_closed_count += 1
            eye_open_count = 0
            status = 'sleepy'
        else:
            eye_closed_count += 1
cap.release()
cv2.destroyAllWindows()
