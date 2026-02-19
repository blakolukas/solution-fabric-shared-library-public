"""
Stable Diffusion Client using FabricFlow API.

This client demonstrates:
1. Text-to-image generation via FabricFlow API
2. Batch image generation with configurable parameters
3. Progress monitoring for long-running generation tasks
4. Image download and display
"""

import time
from typing import Any, Dict, Optional

from fabricflow.config import get_server_url
from utils.fabricflow_client import FabricFlowClient


class StableDiffusionClient:
    """Stable Diffusion client using FabricFlow API."""

    def __init__(self, fabricflow_url: str = None):
        self.fabricflow_url = fabricflow_url or get_server_url()
        self.client = FabricFlowClient(base_url=self.fabricflow_url)
        self.instance_guid = None

    def setup_workflow(self):
        """Create and configure the Stable Diffusion workflow instance."""
        print("Setting up FabricFlow Stable Diffusion workflow instance...")
        try:
            # Create workflow instance
            instance = self.client.instantiate_workflow(
                "stable_diffusion",
                ttl_seconds=3600,  # 1 hour for generation tasks
                metadata={"purpose": "image_generation", "client": "api_client"},
            )
            self.instance_guid = instance["instance_guid"]
            print(f"✓ Workflow instance created: {self.instance_guid}")
            return True
        except Exception as e:
            print(f"✗ Failed to setup workflow: {e}")
            return False

    def cleanup_workflow(self):
        """Clean up the workflow instance."""
        if self.instance_guid:
            try:
                self.client.terminate_instance(self.instance_guid)
                self.client.terminate_session()
                print("✓ Workflow instance cleaned up")
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")

    def generate_images(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate images with Stable Diffusion.

        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with execution results or None if failed
        """
        # Default parameters
        default_params = {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "device": "cuda",  # Use "cpu" if CUDA is not available
            "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "seed": 42,  # For reproducibility
            "num_images_per_prompt": 4,  # Total number of images to generate
            "batch_size": 2,  # Generate 2 images at a time
            "output_dir": "output/",
            "output_name": "generated_image",
            "output_format": "png",
        }

        # Merge with user parameters
        params = {**default_params, **kwargs}
        params["prompt"] = prompt

        print("Starting Stable Diffusion image generation...")
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Model: {params['model_id']}")
        print(f"Steps: {params['num_inference_steps']}")
        print(f"Generating {params['num_images_per_prompt']} images in batches of {params['batch_size']}")
        print("=" * 80)

        try:
            # Execute workflow
            start_time = time.time()
            result = self.client.execute_workflow(
                self.instance_guid,
                inputs=params,
                mode="sync",  # Synchronous execution for image generation
                output_format="json",
            )

            generation_time = time.time() - start_time

            if "error" in result:
                print(f"✗ Generation failed: {result['error']}")
                return None

            print(f"\n✓ Generation completed in {generation_time:.1f} seconds!")

            # Extract results
            saved_paths = result.get("saved_image_paths", [])
            if saved_paths:
                print(f"Generated {len(saved_paths)} images:")
                for path in saved_paths:
                    print(f"  - {path}")

                return {"saved_image_paths": saved_paths, "generation_time": generation_time, "parameters": params}
            else:
                print("✗ No images were generated")
                return None

        except Exception as e:
            print(f"✗ Error during generation: {e}")
            return None

    def generate_images_async(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate images asynchronously (for large batches).

        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional generation parameters

        Returns:
            Execution ID for monitoring progress or None if failed
        """
        # Default parameters for async generation
        default_params = {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "device": "cuda",
            "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "seed": None,  # Random seed for variety
            "num_images_per_prompt": 16,  # Larger batch for async
            "batch_size": 4,
            "output_dir": "output/",
            "output_name": "generated_image",
            "output_format": "png",
        }

        params = {**default_params, **kwargs}
        params["prompt"] = prompt

        print("Starting asynchronous Stable Diffusion generation...")
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Generating {params['num_images_per_prompt']} images in batches of {params['batch_size']}")
        print("=" * 80)

        try:
            result = self.client.execute_workflow(
                self.instance_guid, inputs=params, mode="async", output_format="json"  # Asynchronous execution
            )

            if "execution_id" in result:
                execution_id = result["execution_id"]
                print(f"✓ Generation started with execution ID: {execution_id}")
                return execution_id
            else:
                print(f"✗ Failed to start generation: {result}")
                return None

        except Exception as e:
            print(f"✗ Error starting generation: {e}")
            return None

    def monitor_generation(self, execution_id: str, check_interval: int = 10):
        """
        Monitor asynchronous image generation progress.

        Args:
            execution_id: Execution ID from generate_images_async()
            check_interval: Interval in seconds between status checks
        """
        print(f"\nMonitoring execution {execution_id}...")
        print("Press Ctrl+C to stop monitoring (generation will continue)")
        print("=" * 60)

        try:
            last_status = None
            start_time = time.time()

            while True:
                try:
                    # Get current status
                    status_response = self.client.get_execution_status(execution_id)
                    current_status = status_response.get("status", "unknown")

                    # Print status update if changed
                    if current_status != last_status:
                        elapsed = time.time() - start_time
                        print(f"[{elapsed:.0f}s] Status: {current_status}")
                        last_status = current_status

                    # Check if completed
                    if current_status in ["completed", "failed", "cancelled"]:
                        elapsed = time.time() - start_time
                        print(f"\n✓ Generation {current_status} after {elapsed:.0f} seconds")

                        if current_status == "completed":
                            return self.get_generation_results(execution_id)
                        else:
                            print(f"✗ Generation {current_status}")
                            return None

                    # Wait before next check
                    time.sleep(check_interval)

                except KeyboardInterrupt:
                    print(f"\n✓ Stopped monitoring. Generation {execution_id} continues running.")
                    print(f"  Check status later with: client.get_execution_status('{execution_id}')")
                    return None

                except Exception as e:
                    print(f"✗ Error checking status: {e}")
                    time.sleep(check_interval)

        except Exception as e:
            print(f"✗ Error in monitoring: {e}")
            return None

    def get_generation_results(self, execution_id: str):
        """Get the results of completed image generation."""
        try:
            print("Retrieving generation results...")
            results = self.client.get_execution_results(execution_id)

            if "error" in results:
                print(f"✗ Generation failed: {results['error']}")
                return None

            saved_paths = results.get("saved_image_paths", [])
            if saved_paths:
                print("=" * 60)
                print("IMAGE GENERATION COMPLETED!")
                print("=" * 60)
                print(f"Generated {len(saved_paths)} images:")
                for path in saved_paths:
                    print(f"  - {path}")
                return results
            else:
                print("✗ No images were generated")
                return None

        except Exception as e:
            print(f"✗ Error retrieving results: {e}")
            return None


def main():
    """Main function to run Stable Diffusion generation."""
    print("=" * 80)
    print("FabricFlow Stable Diffusion Client")
    print("=" * 80)
    print("This client generates images from text prompts using")
    print("Stable Diffusion via FabricFlow API.")
    print("=" * 80)

    # Check if FabricFlow service is running
    server_url = get_server_url()
    try:
        import requests

        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code != 200:
            print("✗ FabricFlow service not responding correctly")
            print("  Make sure to run: python fabricflow.py")
            return
    except:
        print(f"✗ Cannot connect to FabricFlow service at {server_url}")
        print("  Make sure to run: python fabricflow.py")
        return

    print("✓ FabricFlow service is running")

    # Create client
    client = StableDiffusionClient()

    try:
        # Setup workflow
        if not client.setup_workflow():
            return

        # Example prompts
        example_prompts = [
            "a beautiful landscape with mountains and a lake, sunset, highly detailed, 8k",
            "a cute robot sitting in a garden, digital art, vibrant colors",
            "abstract geometric patterns, colorful, modern art style",
            "a cozy coffee shop interior, warm lighting, photorealistic",
        ]

        print("\nExample prompts:")
        for i, prompt in enumerate(example_prompts, 1):
            print(f"  {i}. {prompt}")

        print("\nOptions:")
        print("  1. Quick generation (4 images, synchronous)")
        print("  2. Batch generation (16 images, asynchronous)")
        print("  3. Custom prompt")

        choice = input("\nChoose an option [1-3]: ").strip()

        if choice == "1":
            # Quick generation
            prompt_choice = input("Select example prompt [1-4] or enter custom: ").strip()
            try:
                prompt_idx = int(prompt_choice) - 1
                if 0 <= prompt_idx < len(example_prompts):
                    prompt = example_prompts[prompt_idx]
                else:
                    raise ValueError()
            except ValueError:
                prompt = prompt_choice

            print(f"\nGenerating 4 images for: {prompt}")
            result = client.generate_images(prompt=prompt, num_images_per_prompt=4, batch_size=2)

            if result:
                print("\n✓ Generation completed!")
                print(f"Time taken: {result['generation_time']:.1f} seconds")

        elif choice == "2":
            # Batch generation
            prompt_choice = input("Select example prompt [1-4] or enter custom: ").strip()
            try:
                prompt_idx = int(prompt_choice) - 1
                if 0 <= prompt_idx < len(example_prompts):
                    prompt = example_prompts[prompt_idx]
                else:
                    raise ValueError()
            except ValueError:
                prompt = prompt_choice

            print(f"\nStarting batch generation for: {prompt}")
            execution_id = client.generate_images_async(prompt=prompt, num_images_per_prompt=16, batch_size=4)

            if execution_id:
                client.monitor_generation(execution_id)

        elif choice == "3":
            # Custom prompt and parameters
            prompt = input("Enter your prompt: ").strip()
            if not prompt:
                print("✗ Empty prompt")
                return

            print(f"\nGenerating images for: {prompt}")

            # Ask for basic parameters
            try:
                num_images = int(input("Number of images [4]: ") or "4")
                steps = int(input("Inference steps [50]: ") or "50")
            except ValueError:
                num_images, steps = 4, 50

            result = client.generate_images(
                prompt=prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=steps,
                batch_size=min(2, num_images),
            )

            if result:
                print("\n✓ Generation completed!")
                print(f"Time taken: {result['generation_time']:.1f} seconds")

        else:
            print("✗ Invalid choice")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        client.cleanup_workflow()


if __name__ == "__main__":
    main()
