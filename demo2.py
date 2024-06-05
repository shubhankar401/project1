import cv2
import dlib
import numpy as np
import time
from imutils import face_utils

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\shubh\\OneDrive\\Desktop\\project\\shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects = detector(gray, 0)

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)  # Convert dlib's rectangle to OpenCV's bounding box format

        # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Draw landmarks on the image
        for (lx, ly) in shape:
            cv2.circle(image, (lx, ly), 1, (0, 255, 0), -1)

        # Get the 2D coordinates of the selected facial landmarks
        image_points = np.array([
            shape[33],  # Nose tip
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left Mouth corner
            shape[54]   # Right mouth corner
        ], dtype="double")

        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Camera internals
        focal_length = image.shape[1]
        center = (image.shape[1] / 2, image.shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # SolvePnP
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        # Project a 3D point (0, 0, 1000.0) onto the image plane
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0, 0, 1000.0)]), rot_vec, trans_vec, camera_matrix, dist_coeffs)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x_angle = angles[0] * 360
        y_angle = -angles[1] * 360  # Negate the y-angle to correct the direction for the flipped image
        z_angle = angles[2] * 360

        # Determine the text to display based on the y angle
        if y_angle < -5:
            if x_angle < -5:
                text = "Looking left and Down"
            elif x_angle > 5:
                text = "Looking left and Up"
            else:
                text = "Looking left"
        elif y_angle > 5:
            if x_angle < -5:
                text = "Looking right and Down"
            elif x_angle > 5:
                text = "Looking right and Up"
            else:
                text = "Looking right"
        else:
            text = "Forward"

        # Draw nose direction
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(image, p1, p2, (255, 0, 0), 3)

        # Calculate position for the text
        text_x = x
        text_y = y - 10 if y - 10 > 20 else y + h + 20  # Adjust to place the text above or below the face

        # Add the text on the image
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, "x: " + str(np.round(x_angle, 2)), (text_x, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image, "y: " + str(np.round(y_angle, 2)), (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image, "z: " + str(np.round(z_angle, 2)), (text_x, text_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()