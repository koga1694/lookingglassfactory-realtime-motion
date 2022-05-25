import copy
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque
import json
from mediapipe_zmq.utils import get_args
from mediapipe_zmq.utils import emotion_recognition
from mediapipe_zmq.utils import face_mesh
from mediapipe_zmq.utils import left_hand
from mediapipe_zmq.utils import right_hand
from mediapipe_zmq.utils import pose_landmark
import zmq
import time
import tensorflow as tf


#
class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
#


def main():
    # --- 설정 ---

    # ZMQ bind
    context = zmq.Context()
    sockets = context.socket(zmq.PUB)
    sockets.bind('tcp://127.0.0.1:10100')

    # --- json
    dicts = {
    'face_mesh' : {},
    'left_hand' : {},
    'right_hand' : {},
    'pose' : {},
    'emotion' : {}
}

    # --- 인수값 불러오기
    args = get_args()
    
    # 카메라
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    
    # face detection
    model_selection = args.model_selection
    min_face_detection_confidence = args.min_face_detection_confidence

    # holistic
    smooth_landmarks = not args.unuse_smooth_landmarks
    enable_segmentation = args.enable_segmentation
    smooth_segmentation = args.smooth_segmentation
    model_complexity = args.model_complexity
    min_mesh_detection_confidence = args.min_mesh_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    segmentation_score_th = args.segmentation_score_th

    # plot_world_landmark = args.plot_world_landmark
    use_brect = args.use_brect

    # --- 카메라 설정
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)


    # --- mobile_VIT 모델
    mobile_VIT=tf.keras.models.load_model('C:\workplace\zmq\mobile_VIT.h5')

    # --- mediapipe 모델
    
    # face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_face_detection_confidence,
    )

    # holistic model
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        smooth_segmentation=smooth_segmentation,
        min_detection_confidence=min_mesh_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # --- 프레임 체크
    global CvFpsCalc
    a_cvFpsCalc = CvFpsCalc(buffer_len=10)
    # --- 시작 ---
    while True:
        display_fps = a_cvFpsCalc.get()

        # 카메라 On
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  
        # debug_image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        


        # face detection result
        d_results = face_detection.process(image)

        emotion = emotion_recognition(d_results, debug_image, mobile_VIT)
        dicts['emotion'] = [emotion]
        

        # holistic result
        image.flags.writeable = False
        h_results = holistic.process(image)
        image.flags.writeable = True

        # face mesh
        face_landmarks = h_results.face_landmarks
        face = face_mesh(face_landmarks, cap_width, cap_height)
        dicts['face_mesh'] = face

        # hand
        left_hand_landmark = h_results.left_hand_landmarks
        right_hand_landmark = h_results.right_hand_landmarks

        left = left_hand(left_hand_landmark, cap_width, cap_height)
        right = right_hand(right_hand_landmark, cap_width, cap_height)

        dicts['left_hand'] = left
        dicts['right_hand'] = right

        # pose
        pose_landmarks = h_results.pose_landmarks
        pose = pose_landmark(pose_landmarks, cap_width, cap_height)
        dicts['pose'] = pose

        dicts_json = json.dumps(dicts)
        sockets.send_json(dicts_json)


        cv.imshow('MediaPipe Holistic', debug_image)
        if cv.waitKey(5) & 0xFF == 27:
            print(dicts_json)
            print(display_fps)

            break


if __name__ == '__main__':
    main()
    print('완료')
