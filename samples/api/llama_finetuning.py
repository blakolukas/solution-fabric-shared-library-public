"""
LLaMA Fine-tuning Client using FabricFlow API.

This client demonstrates:
1. LLaMA 3 model fine-tuning via FabricFlow API
2. Workflow execution with progress monitoring
3. Asynchronous execution with status checking
4. Result retrieval upon completion
"""

import time
from typing import Optional

from fabricflow.config import get_server_url
from utils.fabricflow_client import FabricFlowClient


class LlamaFineTuningClient:
    """LLaMA fine-tuning client using FabricFlow API."""

    def __init__(self, fabricflow_url: str = None):
        self.fabricflow_url = fabricflow_url or get_server_url()
        self.client = FabricFlowClient(base_url=self.fabricflow_url)
        self.instance_guid = None

    def setup_workflow(self):
        """Create and configure the LLaMA fine-tuning workflow instance."""
        print("Setting up FabricFlow LLaMA fine-tuning workflow instance...")
        try:
            # Create workflow instance with longer TTL for training
            instance = self.client.instantiate_workflow(
                "llama_finetuning",
                ttl_seconds=14400,  # 4 hours for fine-tuning
                metadata={"purpose": "llama_finetuning", "client": "api_client"},
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

    def start_finetuning(self, config: Optional[dict] = None):
        """
        Start LLaMA fine-tuning with optional configuration override.

        Args:
            config: Optional configuration dictionary to override defaults.
                   If None, uses the configuration defined in the workflow JSON.

        Returns:
            Execution ID for monitoring progress
        """
        print("Starting LLaMA 3 fine-tuning workflow...")
        print("=" * 80)

        # Use default configuration from workflow if none provided
        inputs = config or {}

        try:
            # Start asynchronous execution (fine-tuning takes a long time)
            result = self.client.execute_workflow(
                self.instance_guid, inputs=inputs, mode="async", output_format="json"  # Asynchronous execution
            )

            if "execution_id" in result:
                execution_id = result["execution_id"]
                print(f"✓ Fine-tuning started with execution ID: {execution_id}")
                print("  This process may take several hours depending on dataset size...")
                return execution_id
            else:
                print(f"✗ Failed to start execution: {result}")
                return None

        except Exception as e:
            print(f"✗ Error starting fine-tuning: {e}")
            return None

    def monitor_progress(self, execution_id: str, check_interval: int = 30):
        """
        Monitor fine-tuning progress.

        Args:
            execution_id: Execution ID returned from start_finetuning()
            check_interval: Interval in seconds between status checks
        """
        print(f"\nMonitoring execution {execution_id}...")
        print("Press Ctrl+C to stop monitoring (execution will continue)")
        print("=" * 80)

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

                        # Print additional details if available
                        if "details" in status_response:
                            details = status_response["details"]
                            if isinstance(details, dict):
                                for key, value in details.items():
                                    print(f"  {key}: {value}")

                        last_status = current_status

                    # Check if completed
                    if current_status in ["completed", "failed", "cancelled"]:
                        elapsed = time.time() - start_time
                        print(f"\n✓ Execution {current_status} after {elapsed:.0f} seconds")

                        if current_status == "completed":
                            return self.get_results(execution_id)
                        else:
                            print(f"✗ Execution {current_status}")
                            return None

                    # Wait before next check
                    time.sleep(check_interval)

                except KeyboardInterrupt:
                    print(f"\n✓ Stopped monitoring. Execution {execution_id} continues running.")
                    print(f"  Check status later with: client.get_execution_status('{execution_id}')")
                    print(f"  Get results with: client.get_execution_results('{execution_id}')")
                    return None

                except Exception as e:
                    print(f"✗ Error checking status: {e}")
                    time.sleep(check_interval)

        except Exception as e:
            print(f"✗ Error in monitoring: {e}")
            return None

    def get_results(self, execution_id: str):
        """Get the results of a completed fine-tuning execution."""
        try:
            print("Retrieving fine-tuning results...")
            results = self.client.get_execution_results(execution_id)

            if "error" in results:
                print(f"✗ Execution failed: {results['error']}")
                return None

            print("=" * 80)
            print("FINE-TUNING COMPLETED!")
            print("=" * 80)

            # Extract key results
            saved_model_path = results.get("saved_model_path")
            training_result = results.get("training_result", {})

            if saved_model_path:
                print(f"Model saved to: {saved_model_path}")

            if training_result:
                if hasattr(training_result, "training_loss"):
                    print(f"Final training loss: {training_result.training_loss:.4f}")
                elif isinstance(training_result, dict) and "training_loss" in training_result:
                    print(f"Final training loss: {training_result['training_loss']:.4f}")

            # Print all results for reference
            print("\nDetailed Results:")
            for key, value in results.items():
                if key not in ["saved_model_path", "training_result"]:
                    print(f"  {key}: {value}")

            return results

        except Exception as e:
            print(f"✗ Error retrieving results: {e}")
            return None

    def run_finetuning(self, config: Optional[dict] = None, monitor: bool = True):
        """
        Complete fine-tuning workflow: setup, start, monitor, and get results.

        Args:
            config: Optional configuration override
            monitor: Whether to monitor progress automatically

        Returns:
            Fine-tuning results or None if failed
        """
        # Setup workflow
        if not self.setup_workflow():
            return None

        try:
            # Start fine-tuning
            execution_id = self.start_finetuning(config)
            if not execution_id:
                return None

            # Monitor progress if requested
            if monitor:
                return self.monitor_progress(execution_id)
            else:
                print(f"Fine-tuning started. Monitor with execution ID: {execution_id}")
                return {"execution_id": execution_id}

        finally:
            # Note: Don't cleanup workflow automatically for long-running tasks
            # User should call cleanup_workflow() manually when done
            pass


def main():
    """Main function to run LLaMA fine-tuning."""
    print("=" * 80)
    print("FabricFlow LLaMA Fine-tuning Client")
    print("=" * 80)
    print("This client fine-tunes meta-llama/Meta-Llama-3-8B-Instruct using")
    print("the HuggingFaceH4/ultrachat_200k dataset via FabricFlow API.")
    print("")
    print("Note: Fine-tuning requires significant compute resources and time.")
    print("      Make sure your system has sufficient GPU memory and storage.")
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
    client = LlamaFineTuningClient()

    try:
        # Optional: Override default configuration
        custom_config = {
            # Uncomment and modify these to override defaults:
            # "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            # "dataset_name": "HuggingFaceH4/ultrachat_200k",
            # "dataset_split": "train_sft[:500]",  # Smaller dataset for testing
            # "num_train_epochs": 1,
            # "per_device_train_batch_size": 1,
            # "learning_rate": 1e-4,
        }

        # Ask user for confirmation
        print("\nStarting fine-tuning with default configuration...")
        print("This will:")
        print("  - Download meta-llama/Meta-Llama-3-8B-Instruct model")
        print("  - Use HuggingFaceH4/ultrachat_200k dataset (first 1000 samples)")
        print("  - Train for 1 epoch with LoRA fine-tuning")
        print("  - Save the fine-tuned model to ./llama_finetuned_final")
        print()

        confirm = input("Proceed with fine-tuning? [y/N]: ").strip().lower()
        if confirm != "y":
            print("✓ Cancelled by user")
            return

        # Run fine-tuning
        result = client.run_finetuning(
            config=None, monitor=True  # Use default config from workflow  # Monitor progress automatically
        )

        if result:
            print("\n✓ Fine-tuning completed successfully!")
        else:
            print("\n✗ Fine-tuning failed or was cancelled")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup (optional - you might want to keep the instance for multiple runs)
        print("\nCleaning up...")
        client.cleanup_workflow()


if __name__ == "__main__":
    main()
