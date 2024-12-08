import cv2

# video file path
video_path = "small_instance.mp4"
# video_path = "same_instance.mp4"

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Cannot open video or camera.")
    exit()

# Initialize legacy library trackers
legacy_trackers = {
    'CSRT' : cv2.legacy.TrackerCSRT_create(),
    'MIL' : cv2.legacy.TrackerMIL_create(),
    'Boosting' : cv2.legacy.TrackerBoosting_create(), 
    'KCF' : cv2.legacy.TrackerKCF_create(), 
    'TLD' : cv2.legacy.TrackerTLD_create(), 
    'MedianFlow' : cv2.legacy.TrackerMedianFlow_create(), 
    'MOSSE' : cv2.legacy.TrackerMOSSE_create(), 
}

tracker = legacy_trackers['CSRT']

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read the first frame.")
    cap.release()
    exit()

# Select the ROI (Region of Interest) manually
bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)

# Initialize the tracker with the first frame and bounding box
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # Update the tracker with the current frame
    success, bbox = tracker.update(frame)

    if success:
        # If tracking is successful, draw the bounding box
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # If tracking fails
        cv2.putText(frame, "Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("CSRT Tracker", frame)

    # Break loop with 'q' key
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
