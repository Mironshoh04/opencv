import cv2
import numpy as np

# Initialize video capture
video_path = 'same_instance.mp4'
# video_path = 'istockphoto-1187482501-640_adpp_is.mp4'
# video_path = 'istockphoto-1248544042-640_adpp_is.mp4'

cap = cv2.VideoCapture(video_path)  # Replace with your video file or camera

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Select a bounding box for tracking (manually or using object detection)
bbox = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

bbox = (782, 251, 83, 343)    # For same_instance.mp4
# bbox = (1228, 642, 20, 17)    # For small_instance.mp4

# Initialize the tracker with the bounding box
x, y, w, h = bbox
roi_prev = prev_gray[y:y+h, x:x+w]

# Create a mask for drawing purposes
mask = np.zeros_like(first_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow for the ROI
    p0 = np.array([[x + w//2, y + h//2]], dtype=np.float32)  # Track center of bbox
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)

    # Check if the tracking is successful
    if st[0][0] == 1:
        dx, dy = p1[0][0] - p0[0][0], p1[0][1] - p0[0][1]
        x += dx
        y += dy

        # Update the bounding box
        bbox_new = (int(x), int(y), w, h)

        # Draw the updated bounding box
        cv2.rectangle(frame, (bbox_new[0], bbox_new[1]), 
                      (bbox_new[0] + bbox_new[2], bbox_new[1] + bbox_new[3]), (0, 255, 0), 2)

    # Show the result
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Optical Flow Object Tracking', frame)

    # Update previous frame for the next loop
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Average FPS:", fps)
cap.release()
cv2.destroyAllWindows()
