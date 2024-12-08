import cv2
import time

def main():
    # Initialize video capture (0 for webcam, or provide video file path)
    video_path = 'same_instance.mp4'
    # video_path = 'istockphoto-1248544042-640_adpp_is.mp4'
    # video_path = 'istockphoto-1187482501-640_adpp_is.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Select ROI (Region of Interest) manually
    roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    print("Selected ROI:", roi)
    roi = (782, 251, 83, 343)  # For same_instance.mp4
    # roi = (1228, 642, 20, 17)  # For small_instance.mp4

    # Extract the template (ROI)
    x, y, w, h = [int(v) for v in roi]
    template = frame[y:y+h, x:x+w]

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Perform template matching
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Draw a rectangle around the matched area
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Object Tracking", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Average FPS:", fps)
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
