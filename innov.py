import cv2
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
 
def get_ear(eye_points):
    """Calculates the Eye Aspect Ration (EAR)."""
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])   
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

def blink_detection(landmarks):
    left_ear_lanmarks = [landmarks[i] for i in LEFT_EYE]
    right_ear_landmarks = [landmarks[i] for i in RIGHT_EYE]
    
    left_ear = get_ear(np.array(left_ear_lanmarks))
    right_ear = get_ear(np.array(right_ear_landmarks))
    
    EAR_THRESHOLD = 0.2
    
    return left_ear < EAR_THRESHOLD or right_ear < EAR_THRESHOLD

prev_x_pos = None
def head_shake_detection(landmarks):
    global prev_x_pos
    nose_tip_x = landmarks[1].x
    SHAKE_THRESHOLD = 0.05
    if prev_x_pos is not None:
        x_change = nose_tip_x - prev_x_pos
        if abs(x_change) > SHAKE_THRESHOLD:
            prev_x_pos = nose_tip_x
            return True
    prev_x_pos = nose_tip_x
    return False

INNER_TOP_LIP = 13
INNER_BOTTOM_LIP = 14

def open_mouth_detection(landmarks):
    top_lip = landmarks[INNER_TOP_LIP]
    bottom_lip = landmarks[INNER_BOTTOM_LIP]
    mouth_height = abs(top_lip.y - bottom_lip.y)
    MOUTH_OPEN_THRESHOLD = 0.03
    return mouth_height > MOUTH_OPEN_THRESHOLD
        
        