import cv2
import dlib
from scipy.spatial import distance

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Define constants for EAR thresholds
EAR_THRESHOLD = 0.25  # threshold to indicate closed eyes
EAR_CONSEC_FRAMES = 20  # number of consecutive frames the eye must be below the threshold to trigger an alert

# Initialize the dlib facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Initialize variables for drowsiness detection
frame_count = 0
blink_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        # Detect facial landmarks
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Extract the left and right eye coordinates
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        # Calculate Eye Aspect Ratio (EAR) for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EARs of both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check if the eyes are closed
        if ear < EAR_THRESHOLD:
            frame_count += 1

            if frame_count >= EAR_CONSEC_FRAMES:
                print("Drowsiness detected!")
                # Reset frame count to avoid continuous alert
                frame_count = 0
        else:
            frame_count = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
