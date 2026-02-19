"""
Real-time camera background blur client using FabricFlow API with FCN-ResNet50 model.

This client demonstrates:
1. Real-time camera capture
2. Frame processing via existing FabricFlow camera_background_blur workflow instance
3. Live display of processed frames
4. Using public session with shared workflow instances

Prerequisites:
- FabricFlow server running: python fabricflow.py
- Workflow instance created: fabric workflow instantiate camera_background_blur

Usage:
- python camera_background_blur.py                    # Auto-discover existing instance
- python camera_background_blur.py <instance_guid>    # Use specific instance
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from fabricflow.config import get_server_url
from utils.fabricflow_client import FabricFlowClient


@dataclass
class FrameResult:
    """Container for frame processing results."""

    frame_id: int
    original_frame: np.ndarray
    processed_frame: Optional[np.ndarray] = None
    processing_time: float = 0.0
    error: Optional[str] = None


class CameraBlurClient:
    """Real-time camera client with background blur processing using existing workflow instance."""

    def __init__(self, fabricflow_url: str = None, instance_guid: str = None):
        self.fabricflow_url = fabricflow_url or get_server_url()
        self.client = None
        self.instance_guid = instance_guid  # Use provided instance or discover existing

        # Threading and queues
        self.frame_queue = queue.Queue(maxsize=2)  # Limit buffering
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.running = False

        # Statistics
        self.frame_count = 0
        self.processed_count = 0
        self.total_processing_time = 0.0
        self.last_fps_time = time.time()
        self.fps = 0.0

    def setup_workflow(self):
        """Find and use existing FCN-ResNet50 background blur workflow instance."""
        print("Finding existing FCN-ResNet50 workflow instance...")
        try:
            if self.instance_guid:
                # Verify the provided instance exists and is active
                status = self.client.get_instance_status(self.instance_guid)
                if status["status"] in ["idle", "running"]:
                    print(f"✓ Using provided FCN-ResNet50 workflow instance: {self.instance_guid}")
                    return True
                else:
                    print(f"✗ Provided instance {self.instance_guid} is not available (status: {status['status']})")
                    self.instance_guid = None

            if not self.instance_guid:
                # Find existing camera_background_blur instance
                instances = self.client.list_instances(workflow_name="camera_background_blur")
                available_instances = [
                    inst for inst in instances.get("instances", []) if inst["status"] in ["idle", "running"]
                ]

                if available_instances:
                    # Use the first available instance
                    self.instance_guid = available_instances[0]["instance_guid"]
                    print(f"✓ Found existing FCN-ResNet50 workflow instance: {self.instance_guid}")
                    print(
                        f"  Status: {available_instances[0]['status']}, TTL remaining: {available_instances[0]['ttl_remaining_seconds']}s"
                    )
                    return True
                else:
                    print("No existing camera_background_blur workflow instances found")
                    print("Creating new workflow instance...")
                    # Create a new workflow instance
                    instance = self.client.instantiate_workflow("camera_background_blur", ttl_seconds=3600)
                    self.instance_guid = instance["instance_guid"]
                    print(f"✓ Created new FCN-ResNet50 workflow instance: {self.instance_guid}")
                    return True
        except Exception as e:
            print(f"✗ Failed to setup FCN-ResNet50 workflow: {e}")
            return False

    def cleanup_workflow(self):
        """Clean up resources (but keep the shared workflow instance running)."""
        # Don't terminate the shared instance - other processes might be using it
        print("✓ Resources cleaned up (workflow instance kept running for reuse)")

    def process_frame_worker(self):
        """Worker thread for frame processing."""
        while self.running:
            try:
                # Get frame from queue (with timeout)
                frame_result = self.frame_queue.get(timeout=0.1)

                # Process the frame
                start_time = time.time()
                try:
                    # Send frame to FabricFlow for processing
                    result = self.client.execute_workflow(
                        self.instance_guid,
                        inputs={
                            "image": frame_result.original_frame,  # NumPy array - automatically uploaded!
                        },
                        mode="sync",
                        output_format="multipart",  # Get binary response
                    )

                    processing_time = time.time() - start_time

                    # Extract processed frame
                    if result and "blurred_image" in result:
                        processed_frame = result["blurred_image"]
                        if isinstance(processed_frame, np.ndarray):
                            frame_result.processed_frame = processed_frame
                            frame_result.processing_time = processing_time
                            self.processed_count += 1
                            self.total_processing_time += processing_time
                        else:
                            frame_result.error = f"Invalid result type: {type(processed_frame)}"
                    else:
                        error_msg = result.get("error", "Unknown error") if result else "No result"
                        frame_result.error = f"Processing failed: {error_msg}"

                except Exception as e:
                    frame_result.error = f"Processing error: {str(e)}"
                    frame_result.processing_time = time.time() - start_time

                # Put result back
                self.result_queue.put(frame_result)
                self.frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                break

    def start_processing(self):
        """Start the background processing thread."""
        if not self.instance_guid:
            raise RuntimeError("Workflow not set up. Call setup_workflow() first.")

        self.running = True
        self.processing_thread = threading.Thread(target=self.process_frame_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("✓ Background processing started")

    def stop_processing(self):
        """Stop the background processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("✓ Background processing stopped")

    def submit_frame(self, frame: np.ndarray) -> bool:
        """Submit a frame for processing. Returns True if submitted, False if queue full."""
        try:
            frame_result = FrameResult(frame_id=self.frame_count, original_frame=frame.copy())
            self.frame_queue.put_nowait(frame_result)
            self.frame_count += 1
            return True
        except queue.Full:
            return False  # Skip frame if queue is full

    def get_result(self) -> Optional[FrameResult]:
        """Get the next processed frame result."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            # Calculate FPS based on processed frames
            if self.processed_count > 0:
                self.fps = self.processed_count / (current_time - self.last_fps_time + 1e-6)
            self.last_fps_time = current_time
            self.processed_count = 0

    def draw_stats(self, frame: np.ndarray, processing_time: float = 0.0):
        """Draw performance statistics on frame."""
        # Calculate average processing time
        avg_processing_time = self.total_processing_time / max(1, self.processed_count)

        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 35), font, font_scale, color, thickness)
        cv2.putText(frame, f"Processing: {processing_time*1000:.1f}ms", (20, 60), font, font_scale, color, thickness)
        cv2.putText(
            frame, f"Avg Processing: {avg_processing_time*1000:.1f}ms", (20, 85), font, font_scale, color, thickness
        )
        cv2.putText(frame, f"FCN-ResNet50 Model", (20, 110), font, font_scale, (0, 255, 0), thickness)

    def run_camera(self, camera_id: int = 0):
        """Run the camera client."""
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"✗ Cannot open camera {camera_id}")
            return False

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"✓ Camera {camera_id} opened")
        print("Press 'q' to quit, 's' to save current frame")

        # Create windows
        cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("FCN-ResNet50 Background Blur", cv2.WINDOW_AUTOSIZE)

        # Position windows side by side
        cv2.moveWindow("Original", 100, 100)
        cv2.moveWindow("FCN-ResNet50 Background Blur", 750, 100)

        last_frame = None
        last_processing_time = 0.0

        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("✗ Failed to capture frame")
                    break

                # Submit frame for processing (if queue not full)
                submitted = self.submit_frame(frame)

                # Get processed result (if available)
                result = self.get_result()
                if result:
                    if result.processed_frame is not None:
                        last_frame = result.processed_frame
                        last_processing_time = result.processing_time
                    elif result.error:
                        print(f"Processing error: {result.error}")

                # Update FPS
                self.update_fps()

                # Draw stats on original frame
                frame_with_stats = frame.copy()
                self.draw_stats(frame_with_stats, last_processing_time)

                # Display frames
                cv2.imshow("Original", frame_with_stats)

                if last_frame is not None:
                    # Draw stats on processed frame too
                    processed_with_stats = last_frame.copy()
                    self.draw_stats(processed_with_stats, last_processing_time)
                    cv2.imshow("FCN-ResNet50 Background Blur", processed_with_stats)
                else:
                    # Show placeholder
                    placeholder = np.zeros_like(frame)
                    cv2.putText(
                        placeholder, "Processing...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                    )
                    cv2.imshow("FCN-ResNet50 Background Blur", placeholder)

                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and last_frame is not None:
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    original_path = f"fcn_resnet50_camera_original_{timestamp}.jpg"
                    processed_path = f"fcn_resnet50_camera_processed_{timestamp}.jpg"
                    cv2.imwrite(original_path, frame)
                    cv2.imwrite(processed_path, last_frame)
                    print(f"✓ Saved: {original_path}, {processed_path}")

        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Camera released")

        return True


def main(instance_guid: str = None):
    """Main function to run the camera blur client."""
    print("=" * 80)
    print("FabricFlow Real-Time Camera Background Blur with FCN-ResNet50")
    print("=" * 80)
    print("This client captures camera frames and processes them in real-time")
    print("using an existing FabricFlow camera_background_blur workflow instance.")
    print("")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("=" * 80)

    # Check if FabricFlow service is running
    server_url = get_server_url()
    try:
        import requests

        response = requests.get(f"{server_url}/health", timeout=2)
        if response.status_code != 200:
            print("✗ FabricFlow service not responding correctly")
            print("  Make sure to run: python fabricflow.py")
            return
    except:
        print(f"✗ Cannot connect to FabricFlow service at {server_url}")
        print("  Make sure to run: python fabricflow.py")
        return

    print("✓ FabricFlow service is running")

    # Create custom client configured for public session (no private session needed)
    with FabricFlowClient(base_url=server_url, use_private_session=False) as fabricflow_client:
        client = CameraBlurClient(instance_guid=instance_guid)
        client.client = fabricflow_client

        try:
            # Setup workflow
            if not client.setup_workflow():
                return

            # Start background processing
            client.start_processing()

            # Run camera
            success = client.run_camera()

            if success:
                print("\n" + "=" * 80)
                print("SESSION STATISTICS")
                print("=" * 80)
                print(f"Total frames captured: {client.frame_count}")
                print(f"Total frames processed: {client.processed_count}")
                if client.total_processing_time > 0:
                    avg_time = client.total_processing_time / max(1, client.processed_count)
                    print(f"Average processing time: {avg_time*1000:.1f}ms")
                    print(f"Theoretical max FPS: {1/avg_time:.1f}")
                print("=" * 80)

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Cleanup
            client.stop_processing()
            client.cleanup_workflow()


if __name__ == "__main__":
    import sys

    # Allow specifying instance GUID as command line argument
    instance_guid = None
    if len(sys.argv) > 1:
        instance_guid = sys.argv[1]
        print(f"Using specified instance GUID: {instance_guid}")

    main(instance_guid)
