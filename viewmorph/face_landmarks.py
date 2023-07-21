import os

import cv2
import dlib
import numpy as np

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
path = os.path.join("./checkpoints/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(path)


def get_landmarks(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = detector(gray)
    # Get the landmarks for each face
    landmarks = [predictor(gray, face) for face in faces]
    return landmarks


def extract_points(landmarks):
    points = []
    for landmark in landmarks:
        for i in range(0, 68):
            point = [landmark.part(i).x, landmark.part(i).y]
            points.append(point)
    return np.array(points)


# Function to detect keypoints and extract descriptors using ORB
def detect_and_extract_ORB(image, num_keypoints=500):
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=num_keypoints)

    # Detect keypoints
    keypoints = orb.detect(image, None)

    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)

    # Convert keypoints to numpy array
    points = np.array([kp.pt for kp in keypoints])

    return points, descriptors


# Function to detect keypoints and extract descriptors using AKAZE
def detect_and_extract_AKAZE(image):
    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create()

    # Detect keypoints
    keypoints = akaze.detect(image, None)

    # Compute descriptors
    keypoints, descriptors = akaze.compute(image, keypoints)

    # Convert keypoints to numpy array
    points = np.array([kp.pt for kp in keypoints])

    return points, descriptors
