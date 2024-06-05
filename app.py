import cv2
import numpy as np

# Load pre-trained Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier('C:\\Users\\shubh\\OneDrive\\Desktop\\project\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\shubh\\OneDrive\\Desktop\\project\\haarcascade_eye.xml')

# Function to detect eyes in a frame
def detect_eyes(gray, frame):
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Function to calculate depth using StereoBM
def calculate_depth(img_left, img_right):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img_left, img_right)
    # Normalize the disparity map for better visualization
    min_disparity = disparity.min()
    max_disparity = disparity.max()
    normalized_disparity = (disparity - min_disparity) / (max_disparity - min_disparity)
    # Scale normalized disparity to get depth values
    depth = 1 / (normalized_disparity + 0.01)
    return depth

# Open the default camera (index 0 and 1 for two cameras)
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame from first camera
    ret1, frame1 = cap1.read()
    if not ret1:
        break
    
    # Capture frame-by-frame from second camera
    ret2, frame2 = cap2.read()
    if not ret2:
        break
    
    # Convert frames to grayscale for depth calculation
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate depth map using StereoBM
    depth_map = calculate_depth(gray1, gray2)
    
    # Detect faces in frame 1
    faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces1:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray1[y:y+h, x:x+w]
        frame1 = detect_eyes(roi_gray, frame1)
        # Calculate depth for each detected face
        depth = np.mean(depth_map[y:y+h, x:x+w])
        # Display depth as text on frame 1
        cv2.putText(frame1, f'Depth: {depth:.2f} cm', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Detect faces in frame 2
    faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces2:
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray2[y:y+h, x:x+w]
        frame2 = detect_eyes(roi_gray, frame2)
        # Calculate depth for each detected face
        depth = np.mean(depth_map[y:y+h, x:x+w])
        # Display depth as text on frame 2
        cv2.putText(frame2, f'Depth: {depth:.2f} cm', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frames side by side
    both_frames = np.hstack((frame1, frame2))
    cv2.imshow('Cameras Side by Side', both_frames)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap1.release()
cap2.release()
cv2.destroyAllWindows()
