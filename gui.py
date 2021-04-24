
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2



import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer



class VideoTransformer(VideoTransformerBase):
    def transform(self, video_capture):

        EAR_THRESHOLD = 0.3
        
        EAR_CONSEC_FRAMES = 30
        
        
        COUNTER = 0
        
        
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        
        def eye_aspect_ratio(eye):
            A = distance.euclidean(eye[1], eye[5])
            B = distance.euclidean(eye[2], eye[4])
            C = distance.euclidean(eye[0], eye[3])
        
            x = (A+B) / (2*C)
            return x
        
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        
        while(True):
            
            ret, frame = video_capture.read()
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
            faces = detector(gray, 0)
        
        
            face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        
            for (x,y,w,h) in face_rectangle:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        
            for face in faces:
        
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
        
        
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
        
        
                leftEyeAspectRatio = eye_aspect_ratio(leftEye)
                rightEyeAspectRatio = eye_aspect_ratio(rightEye)
        
                eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        
        
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        
                if(eyeAspectRatio < EAR_THRESHOLD):
                    COUNTER += 1
                    
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                else:
                    COUNTER = 0
        
        

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)