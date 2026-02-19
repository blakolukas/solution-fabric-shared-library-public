import json
import time

import cv2

import tasks  # Import to trigger task registration
from core.logging import init_logging
from core.workflow import Workflow


def main():
    init_logging(level="DEBUG", log_file="fabricflow.log")

    with open("workflows/yolo_object_detection.json", "r") as f:
        workflow_definition = json.load(f)

    workflow = Workflow.from_definition(workflow_definition)
    workflow_context = None

    # Create window in main thread
    window_name = "YOLO v11 Object Detection (SDK)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print(f"Starting YOLO v11 object detection workflow.")
    print(f"Window '{window_name}' created. Press 'q' in the window to quit.")
    print("Press 's' to save current frame.")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open camera 0")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # FPS calculation variables
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 1.0  # Update FPS every second
    last_fps_update = start_time

    # Processing time tracking
    total_processing_time = 0.0
    processed_frames = 0

    try:
        iteration = 0
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to capture frame")
                break

            # Measure processing time
            process_start = time.time()

            # Run the workflow with the captured frame
            if workflow_context is None:
                result = workflow.run_once({"inputs": {"image": frame}})  # Pass captured frame directly
            else:
                # Update the context with new input image
                workflow_context["inputs"]["image"] = frame
                result = workflow.run_once(context=workflow_context)

            # Get the workflow context from the workflow's internal state
            # Context is maintained by the workflow instance, not returned in outputs
            workflow_context = workflow.get_last_context()
            processing_time = time.time() - process_start

            # Track processing statistics
            total_processing_time += processing_time
            processed_frames += 1

            # Get the annotated image and detections from workflow output
            annotated_image = result.get("annotated_image")
            detections = result.get("detections", [])

            if annotated_image is not None:
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - last_fps_update

                if elapsed >= fps_update_interval:
                    fps = frame_count / elapsed
                    frame_count = 0
                    last_fps_update = current_time

                # Draw performance stats on the annotated image
                display_image = annotated_image.copy()
                avg_processing_time = total_processing_time / max(1, processed_frames)

                # Draw background for stats
                overlay = display_image.copy()
                cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)

                # Draw stats text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (255, 255, 255)
                thickness = 1

                cv2.putText(display_image, f"FPS: {fps:.1f}", (20, 30), font, font_scale, color, thickness)
                cv2.putText(
                    display_image,
                    f"Processing: {processing_time*1000:.1f}ms",
                    (20, 50),
                    font,
                    font_scale,
                    color,
                    thickness,
                )
                cv2.putText(
                    display_image,
                    f"Avg Processing: {avg_processing_time*1000:.1f}ms",
                    (20, 70),
                    font,
                    font_scale,
                    color,
                    thickness,
                )
                cv2.putText(
                    display_image,
                    f"Detections: {len(detections)}",
                    (20, 90),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness,
                )

                # Display image in main thread (CRITICAL for OpenCV stability)
                cv2.imshow(window_name, display_image)

                # Check for key press (must be in same thread as imshow)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("'q' pressed. Exiting workflow...")
                    break
                elif key == ord("s"):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    original_path = f"camera_object_detection_original_{timestamp}.jpg"
                    annotated_path = f"camera_object_detection_annotated_{timestamp}.jpg"
                    cv2.imwrite(original_path, frame)
                    cv2.imwrite(annotated_path, annotated_image)
                    print(f"✓ Saved: {original_path}, {annotated_path}")

            iteration += 1
            if iteration % 30 == 0:
                avg_time = total_processing_time / max(1, processed_frames)
                print(
                    f"Processed {iteration} frames... (FPS: {fps:.1f}, Avg: {avg_time*1000:.1f}ms, Detections: {len(detections)})"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Print final statistics
        if processed_frames > 0:
            avg_time = total_processing_time / processed_frames
            print(f"\nFinal Statistics:")
            print(f"Total frames processed: {processed_frames}")
            print(f"Average processing time: {avg_time*1000:.1f}ms")
            if avg_time > 0:
                print(f"Theoretical max FPS: {1/avg_time:.1f}")


if __name__ == "__main__":
    main()
