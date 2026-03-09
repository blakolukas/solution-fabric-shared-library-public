"""Smoke tests for tasks/huggingface/ — verify importability and basic execution."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# download_hf_model
# ---------------------------------------------------------------------------
class TestDownloadHfModelSmoke:
    """Smoke tests for tasks.huggingface.download_hf_model."""

    def test_importable(self):
        from tasks.huggingface.download_hf_model import download_hf_model

        assert hasattr(download_hf_model, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self, tmp_path):
        fake_cached_path = str(tmp_path / "model.gguf")
        mock_hf_hub = MagicMock()
        mock_hf_hub.hf_hub_download.return_value = fake_cached_path

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
            from tasks.huggingface.download_hf_model import download_hf_model

            result = download_hf_model.__wrapped_function__(
                repository_id="TheBloke/Llama-2-7B-GGUF",
                model_name="llama-2-7b.Q4_K_M.gguf",
            )
        assert result == fake_cached_path


# ---------------------------------------------------------------------------
# huggingface_login
# ---------------------------------------------------------------------------
class TestHuggingfaceLoginSmoke:
    """Smoke tests for tasks.huggingface.huggingface_login."""

    def test_importable(self):
        from tasks.huggingface.huggingface_login import huggingface_login

        assert hasattr(huggingface_login, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        mock_hf_hub = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
            from tasks.huggingface.huggingface_login import huggingface_login

            result = huggingface_login.__wrapped_function__(token="hf_fake_token")

        assert result == "logged_in"
        mock_hf_hub.login.assert_called_once_with(
            token="hf_fake_token",
            add_to_git_credential=False,
            new_session=True,
        )

    @pytest.mark.unit
    def test_env_variable_fallback(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")
        mock_hf_hub = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
            from tasks.huggingface.huggingface_login import huggingface_login

            result = huggingface_login.__wrapped_function__()

        assert result == "logged_in"
        mock_hf_hub.login.assert_called_once()
        call_kwargs = mock_hf_hub.login.call_args[1]
        assert call_kwargs["token"] == "hf_env_token"

    @pytest.mark.unit
    def test_no_token_raises(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        mock_hf_hub = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
            from tasks.huggingface.huggingface_login import huggingface_login

            with pytest.raises(ValueError, match="No HuggingFace token"):
                huggingface_login.__wrapped_function__()

