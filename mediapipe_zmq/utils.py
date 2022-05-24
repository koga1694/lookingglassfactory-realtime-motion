import copy
import argparse
from unicodedata import category
import tensorflow as tf
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque

categories= ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def get_args():
    parser = argparse.ArgumentParser()
    
    # 카메라 설정
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # face detection
    parser.add_argument("--model_selection", type=int, default=0)
    parser.add_argument("--min_face_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)

    # holistic
    parser.add_argument('--unuse_smooth_landmarks', action='store_true')
    parser.add_argument('--enable_segmentation', action='store_true')
    parser.add_argument('--smooth_segmentation', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_mesh_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--segmentation_score_th",
                        help='segmentation_score_threshold',
                        type=float,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')
    # parser.add_argument('--plot_world_landmark', action='store_true')
    args = parser.parse_args()

    return args

def emotion_recognition(d_results, debug_image, mobile_VIT):
    H, W, _ = debug_image.shape

    if d_results.detections:
        faces = []
        for detection in d_results.detections:
            box = detection.location_data.relative_bounding_box

            x = int(box.xmin * W)
            y = int(box.ymin * H)
            w = int(box.width * W)
            h = int(box.height * H)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, W)
            y2 = min(y + h, H)

            face = debug_image[y1:y2,x1:x2]
           
            faces.append(face)

        x = recognition_preprocessing(faces)

        emotion = mobile_VIT.predict(x).argmax()
        result = categories[emotion]

        print(result)

        return result

def face_mesh(face_landmarks, cap_width, cap_height):
    face_point = {}
    if face_landmarks is not None:
        for idx, landmark in enumerate(face_landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * cap_width), cap_width - 1)
            landmark_y = min(int(landmark.y * cap_height), cap_height - 1)
            landmark_z = landmark.z

            face_point[f'{idx}'] = [landmark_x, landmark_y, landmark_z]

        return face_point

def left_hand(left_hand_landmark, cap_width, cap_height):
    left_hand_point = {}
    if left_hand_landmark is not None:
        for idx, landmark in enumerate(left_hand_landmark.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_x = min(int(landmark.x * cap_width), cap_width - 1)             
            landmark_y = min(int(landmark.y * cap_height), cap_height - 1)
            landmark_z = landmark.z

            left_hand_point[f'{idx}'] = [landmark_x, landmark_y, landmark_z]
        
        return left_hand_point

def right_hand(right_hand_landmark, cap_width, cap_height):
    right_hand_point = {}
    if right_hand_landmark is not None:
        for idx, landmark in enumerate(right_hand_landmark.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_x = min(int(landmark.x * cap_width), cap_width - 1)             
            landmark_y = min(int(landmark.y * cap_height), cap_height - 1)
            landmark_z = landmark.z

            right_hand_point[f'{idx}'] = [landmark_x, landmark_y, landmark_z]

        return right_hand_point

def pose_landmark(pose_landmarks, cap_width, cap_height):
    pose_point = {}
    if pose_landmarks is not None:
        for idx, landmark in enumerate(pose_landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * cap_width), cap_width - 1)             
            landmark_y = min(int(landmark.y * cap_height), cap_height - 1)
            landmark_z = landmark.z

            pose_point[f'{idx}'] = [landmark_x, landmark_y, landmark_z]
        
        return pose_point








def resize_face(face):
    x = tf.convert_to_tensor(face)
    return tf.image.resize(x, (256,256))

def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x