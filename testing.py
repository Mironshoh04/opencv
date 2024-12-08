import cv2
import time
import numpy as np

# Initialize video paths
video_paths = {
    "T1": "same_instance.mp4",  # The path for T1 test case video
    "T2": "small_instance.mp4",    # The path for T2 test case video
}

# Define ground truth for evaluation (manually annotate or load from a file)
# Format: {frame_number: (x, y, w, h)}
ground_truth = {
    "T1": {1: (50, 50, 100, 100), 2: (52, 50, 98, 100)},  # Replace with actual ground truth
    "T2": {1: (30, 30, 20, 20), 2: (32, 30, 18, 18)},
}

# Define the legacy trackers
legacy_trackers = {
    'CSRT': cv2.legacy.TrackerCSRT_create,
    'MIL': cv2.legacy.TrackerMIL_create,
    'Boosting': cv2.legacy.TrackerBoosting_create,
    'KCF': cv2.legacy.TrackerKCF_create,
    'TLD': cv2.legacy.TrackerTLD_create,
    'MedianFlow': cv2.legacy.TrackerMedianFlow_create,
    'MOSSE': cv2.legacy.TrackerMOSSE_create,
}

results = {test: {} for test in video_paths.keys()}  # To store results for T1 and T2

# Iterate through T1 and T2 test cases
for test_case, video_path in video_paths.items():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video for {test_case} test case.")
        continue

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read the first frame for {test_case} test case.")
        cap.release()
        continue

    # Select ROI manually for the first frame
    bbox = cv2.selectROI(f"Select ROI for {test_case}", frame, fromCenter=False, showCrosshair=True)

    for tracker_name, tracker_creator in legacy_trackers.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning
        tracker = tracker_creator()
        tracker.init(frame, bbox)

        print(f"Testing {tracker_name} on {test_case}...")
        total_iou = 0
        frame_count = 0
        total_time = 0
        success_count = 0

        t1 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Measure processing time for the tracker update
            start_time = time.time()
            success, pred_bbox = tracker.update(frame)
            end_time = time.time()

            processing_time = end_time - start_time
            total_time += processing_time

            if success:
                success_count += 1
                x, y, w, h = map(int, pred_bbox)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate IoU with ground truth if available
                if frame_count + 1 in ground_truth[test_case]:
                    gt_bbox = ground_truth[test_case][frame_count + 1]
                    xi1 = max(x, gt_bbox[0])
                    yi1 = max(y, gt_bbox[1])
                    xi2 = min(x + w, gt_bbox[0] + gt_bbox[2])
                    yi2 = min(y + h, gt_bbox[1] + gt_bbox[3])
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

                    gt_area = gt_bbox[2] * gt_bbox[3]
                    pred_area = w * h
                    union_area = gt_area + pred_area - inter_area

                    iou = inter_area / union_area if union_area > 0 else 0
                    total_iou += iou

            frame_count += 1

            # Show the frame (optional)
            # cv2.imshow(f"{tracker_name} on {test_case}", frame)
            # if cv2.waitKey(20) & 0xFF == ord('q'):
            #     break

        t2 = time.time()
        tt = t2 - t1

        # Store results
        avg_iou = total_iou / frame_count if frame_count > 0 else 0
        avg_time = total_time / frame_count if frame_count > 0 else 0
        fps = frame_count / total_time if total_time > 0 else 0
        fps2 = frame_count / tt if tt > 0 else 0
        results[test_case][tracker_name] = {
            "Accuracy (IoU)": avg_iou,
            "Total Time (s)": total_time,
            "fps": fps,
            "fps2": fps2,
            "Processing Time (s)": avg_time,
            "Frames Processed": frame_count,
            "Success Rate": success_count / frame_count if frame_count > 0 else 0,
        }

        print("Accuracy (IoU) ", results[test_case][tracker_name]["Accuracy (IoU)"])
        print("Total Time (s) ", results[test_case][tracker_name]["Total Time (s)"])
        print("fps ", results[test_case][tracker_name]["fps"])
        print("fps2 ", results[test_case][tracker_name]["fps2"])
        print("Processing Time (s) ", results[test_case][tracker_name]["Processing Time (s)"])
        print("Frames Processed ", results[test_case][tracker_name]["Frames Processed"])
        print("Success Rate ", results[test_case][tracker_name]["Success Rate"])

    cap.release()

# cv2.destroyAllWindows()

# # Print results
# for test_case, tracker_results in results.items():
#     print(f"\nResults for {test_case} test case:")
#     for tracker_name, metrics in tracker_results.items():
#         print(f"{tracker_name}: {metrics}")
