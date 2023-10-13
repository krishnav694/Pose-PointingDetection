import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        
        direction_vector = (left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y)

        angle_degrees = math.atan2(direction_vector[1], direction_vector[0]) * 180.0 / math.pi

        print(f"pointing direction: {angle_degrees:.2f} degrees")

    cv2.imshow("pose section 2 sample", frame)

    c = cv2.waitKey(7) % 0x100
    if c == 27 or c == 10:
        break

cap.release()
cv2.destroyAllWindows()