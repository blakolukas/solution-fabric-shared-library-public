"""
Background blur SDK sample using direct workflow execution with FCN-ResNet50 model.

This sample demonstrates:
1. Direct workflow execution using FabricFlow SDK
2. Real-time camera capture and processing
3. Background blur with FCN-ResNet50 model
4. Live display with performance metrics

This is the SDK counterpart to camera_blur_client.py (API version).
"""

import json
import time

import cv2

import tasks  # Import to trigger task registration
from core.logging import init_logging
from core.workflow import Workflow


def main():
    init_logging(level="DEBUG", log_file="fabricflow.log")

    # Load the background blur workflow
    with open("workflows/camera_background_blur.json", "r") as f:
        workflow_definition = json.load(f)

    workflow = Workflow.from_definition(workflow_definition)
    workflow_context = None

    # Create window in main thread
    window_name = "FCN-ResNet50 Background Blur (SDK)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    # Create windows for side-by-side display
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("FCN-ResNet50 Background Blur", cv2.WINDOW_AUTOSIZE)

    # Position windows side by side
    cv2.moveWindow("Original", 100, 100)
    cv2.moveWindow("FCN-ResNet50 Background Blur", 750, 100)

    print(f"Starting FCN-ResNet50 background blur SDK workflow.")
    print(f"Windows created. Press 'q' in any window to quit.")
    print("Press 's' to save current frame.")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open camera 0")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

            # Get the processed image from workflow output
            blurred_image = result.get("blurred_image")

            if blurred_image is not None:
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - last_fps_update

                if elapsed >= fps_update_interval:
                    fps = frame_count / elapsed
                    frame_count = 0
                    last_fps_update = current_time

                # Calculate average processing time
                avg_processing_time = total_processing_time / processed_frames

                # Draw stats on original frame
                original_with_stats = frame.copy()
                draw_stats(original_with_stats, fps, processing_time * 1000, avg_processing_time * 1000)

                # Draw stats on processed frame
                blurred_with_stats = blurred_image.copy()
                draw_stats(blurred_with_stats, fps, processing_time * 1000, avg_processing_time * 1000)

                # Display images in separate windows
                cv2.imshow("Original", original_with_stats)
                cv2.imshow("FCN-ResNet50 Background Blur", blurred_with_stats)

                # Check for key press (must be in same thread as imshow)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("'q' pressed. Exiting workflow...")
                    break
                elif key == ord("s"):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    original_path = f"fcn_resnet50_background_blur_sdk_original_{timestamp}.jpg"
                    processed_path = f"fcn_resnet50_background_blur_sdk_processed_{timestamp}.jpg"
                    cv2.imwrite(original_path, frame)
                    cv2.imwrite(processed_path, blurred_image)
                    print(f"✓ Saved: {original_path}, {processed_path}")

            iteration += 1
            if iteration % 30 == 0:
                avg_time = total_processing_time / processed_frames if processed_frames > 0 else 0
                print(f"Processed {iteration} frames... (FPS: {fps:.1f}, Avg: {avg_time*1000:.1f}ms)")

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
            print("\n" + "=" * 60)
            print("SESSION STATISTICS")
            print("=" * 60)
            print(f"Total frames processed: {processed_frames}")
            print(f"Average processing time: {avg_time*1000:.1f}ms")
            if avg_time > 0:
                print(f"Theoretical max FPS: {1/avg_time:.1f}")
            print("=" * 60)


def draw_stats(frame, fps, processing_time_ms, avg_processing_time_ms):
    """Draw performance statistics on frame."""
    # Draw background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), font, font_scale, color, thickness)
    cv2.putText(frame, f"Processing: {processing_time_ms:.1f}ms", (20, 60), font, font_scale, color, thickness)
    cv2.putText(frame, f"Avg Processing: {avg_processing_time_ms:.1f}ms", (20, 85), font, font_scale, color, thickness)
    cv2.putText(frame, f"FCN-ResNet50 SDK Mode", (20, 110), font, font_scale, (0, 255, 0), thickness)


if __name__ == "__main__":
    main()
