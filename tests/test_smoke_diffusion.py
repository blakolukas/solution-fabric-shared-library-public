"""Smoke tests for tasks/diffusion/ — verify importability and basic execution."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# load_stable_diffusion_pipeline
# ---------------------------------------------------------------------------
class TestLoadStableDiffusionPipelineSmoke:
    """Smoke tests for tasks.diffusion.load_stable_diffusion_pipeline."""

    def test_importable(self):
        from tasks.diffusion.load_stable_diffusion_pipeline import (
            load_stable_diffusion_pipeline,
        )

        assert hasattr(load_stable_diffusion_pipeline, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"

        mock_diffusers = MagicMock()
        mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipeline

        with patch.dict(
            sys.modules, {"torch": mock_torch, "diffusers": mock_diffusers}
        ):
            from tasks.diffusion.load_stable_diffusion_pipeline import (
                load_stable_diffusion_pipeline,
            )

            result = load_stable_diffusion_pipeline.__wrapped_function__(
                model_id="runwayml/stable-diffusion-v1-5", device="cpu"
            )

        mock_diffusers.StableDiffusionPipeline.from_pretrained.assert_called_once()
        assert result is mock_pipeline


# ---------------------------------------------------------------------------
# generate_image_from_prompt
# ---------------------------------------------------------------------------
class TestGenerateImageFromPromptSmoke:
    """Smoke tests for tasks.diffusion.generate_image_from_prompt."""

    def test_importable(self):
        from tasks.diffusion.generate_image_from_prompt import generate_image_from_prompt

        assert hasattr(generate_image_from_prompt, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        from PIL import Image

        import numpy as np

        from tasks.diffusion.generate_image_from_prompt import generate_image_from_prompt

        fake_image = Image.fromarray(
            np.zeros((512, 512, 3), dtype=np.uint8)
        )
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.images = [fake_image]
        mock_pipeline.device = "cpu"

        mock_torch = MagicMock()
        mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = generate_image_from_prompt.__wrapped_function__(
                pipeline=mock_pipeline,
                prompt="A sunset over mountains",
                num_images_per_prompt=1,
                batch_size=1,
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is fake_image

    @pytest.mark.unit
    def test_multiple_images(self):
        from PIL import Image

        import numpy as np

        from tasks.diffusion.generate_image_from_prompt import generate_image_from_prompt

        fake_images = [
            Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            for _ in range(2)
        ]
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.images = fake_images
        mock_pipeline.device = "cpu"

        mock_torch = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = generate_image_from_prompt.__wrapped_function__(
                pipeline=mock_pipeline,
                prompt="A forest path",
                num_images_per_prompt=2,
                batch_size=2,
            )

        assert len(result) == 2

