import cv2

def split_camera_feed():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get the width and height of the frame
        height, width, _ = frame.shape

        # Calculate the split point for 80:20 ratio
        split_point = int(width * 0.9)

        # Split the frame into two parts
        left_frame = frame[:, :split_point]
        right_frame = frame[:, :split_point]

        # Display the resulting frames
        # cv2.imshow('Left Frame (90%)', left_frame)
        # cv2.imshow('Right Frame (90%)', right_frame)

        # Press 'q' to exit the loop
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        return left_frame, right_frame

    # When everything is done, release the capture and close windows
    cap.release()
    

if __name__ == "__main__":
    split_camera_feed()
