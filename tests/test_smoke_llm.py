"""Smoke tests for tasks/llm/ — verify importability and basic execution."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------
class TestBuildMessagesSmoke:
    """Smoke tests for tasks.llm.build_messages."""

    def test_importable(self):
        from tasks.llm.build_messages import build_messages

        assert hasattr(build_messages, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.llm.build_messages import build_messages

        result = build_messages.__wrapped_function__(user_message="Hello")
        assert isinstance(result, list)
        assert result[-1] == {"role": "user", "content": "Hello"}

    @pytest.mark.unit
    def test_with_system_message(self):
        from tasks.llm.build_messages import build_messages

        result = build_messages.__wrapped_function__(
            user_message="Hello", system_message="You are helpful."
        )
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[-1]["role"] == "user"

    @pytest.mark.unit
    def test_with_history(self):
        from tasks.llm.build_messages import build_messages

        history = [{"role": "assistant", "content": "Hi!"}]
        result = build_messages.__wrapped_function__(
            user_message="How are you?", history=history
        )
        assert len(result) == 2
        assert result[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# decode_llm_output
# ---------------------------------------------------------------------------
class TestDecodeLlmOutputSmoke:
    """Smoke tests for tasks.llm.decode_llm_output."""

    def test_importable(self):
        from tasks.llm.decode_llm_output import decode_llm_output

        assert hasattr(decode_llm_output, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.llm.decode_llm_output import decode_llm_output

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Hello world"
        output_ids = [[101, 7592, 2088, 102]]

        result = decode_llm_output.__wrapped_function__(
            output_ids=output_ids, tokenizer=mock_tokenizer
        )
        assert result == "Hello world"
        mock_tokenizer.decode.assert_called_once_with(
            output_ids[0], skip_special_tokens=True
        )


# ---------------------------------------------------------------------------
# tokenize_prompt
# ---------------------------------------------------------------------------
class TestTokenizePromptSmoke:
    """Smoke tests for tasks.llm.tokenize_prompt."""

    def test_importable(self):
        from tasks.llm.tokenize_prompt import tokenize_prompt

        assert hasattr(tokenize_prompt, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.llm.tokenize_prompt import tokenize_prompt

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [101, 7592], "attention_mask": [1, 1]}

        result = tokenize_prompt.__wrapped_function__(
            prompt="Hello", tokenizer=mock_tokenizer
        )
        mock_tokenizer.assert_called_once_with(
            "Hello",
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        assert "input_ids" in result


# ---------------------------------------------------------------------------
# load_gguf_model
# ---------------------------------------------------------------------------
class TestLoadGgufModelSmoke:
    """Smoke tests for tasks.llm.load_gguf_model."""

    def test_importable(self):
        from tasks.llm.load_gguf_model import load_gguf_model

        assert hasattr(load_gguf_model, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self, tmp_path):
        import sys

        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake model")
        mock_llama_instance = MagicMock()
        mock_llama_cls = MagicMock(return_value=mock_llama_instance)
        mock_llama_cpp = MagicMock()
        mock_llama_cpp.Llama = mock_llama_cls

        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            from tasks.llm.load_gguf_model import load_gguf_model

            result = load_gguf_model.__wrapped_function__(model_path=str(model_file))

        mock_llama_cls.assert_called_once()
        # Result is wrapped in ThreadSafeModel
        assert hasattr(result, "_model")

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path):
        import sys

        mock_llama_cpp = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            from tasks.llm.load_gguf_model import load_gguf_model

            with pytest.raises(FileNotFoundError):
                load_gguf_model.__wrapped_function__(
                    model_path=str(tmp_path / "nonexistent.gguf")
                )


# ---------------------------------------------------------------------------
# llm_generate
# ---------------------------------------------------------------------------
class TestLlmGenerateSmoke:
    """Smoke tests for tasks.llm.llm_generate."""

    def test_importable(self):
        from tasks.llm.llm_generate import llm_generate

        assert hasattr(llm_generate, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.llm.llm_generate import llm_generate

        mock_model = MagicMock()
        mock_model.return_value = {
            "choices": [{"text": "response text"}],
            "usage": {"total_tokens": 15},
        }

        text, tokens = llm_generate.__wrapped_function__(
            model=mock_model, prompt="Hello world"
        )
        assert text == "response text"
        assert tokens == 15

    @pytest.mark.unit
    def test_empty_prompt_returns_empty(self):
        from tasks.llm.llm_generate import llm_generate

        mock_model = MagicMock()
        text, tokens = llm_generate.__wrapped_function__(
            model=mock_model, prompt=""
        )
        assert text == ""
        assert tokens == 0
        mock_model.assert_not_called()


# ---------------------------------------------------------------------------
# llm_inference
# ---------------------------------------------------------------------------
class TestLlmInferenceSmoke:
    """Smoke tests for tasks.llm.llm_inference."""

    def test_importable(self):
        from tasks.llm.llm_inference import llm_inference

        assert hasattr(llm_inference, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.llm.llm_inference import llm_inference

        mock_model = MagicMock()
        mock_model.return_value = {"choices": [{"text": "The answer"}]}

        result = llm_inference.__wrapped_function__(
            model=mock_model, prompt="What is 2+2?"
        )
        assert result == "The answer"

    @pytest.mark.unit
    def test_empty_prompt_returns_empty(self):
        from tasks.llm.llm_inference import llm_inference

        mock_model = MagicMock()
        result = llm_inference.__wrapped_function__(model=mock_model, prompt="")
        assert result == ""
        mock_model.assert_not_called()

    @pytest.mark.unit
    def test_string_output(self):
        from tasks.llm.llm_inference import llm_inference

        mock_model = MagicMock()
        mock_model.return_value = "plain string output"

        result = llm_inference.__wrapped_function__(
            model=mock_model, prompt="test prompt"
        )
        assert result == "plain string output"


# ---------------------------------------------------------------------------
# run_llm_inference
# ---------------------------------------------------------------------------
class TestRunLlmInferenceSmoke:
    """Smoke tests for tasks.llm.run_llm_inference."""

    def test_importable(self):
        from tasks.llm.run_llm_inference import run_llm_inference

        assert hasattr(run_llm_inference, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        import sys

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2

        # Create fake tensor-like inputs
        mock_input_ids = MagicMock()
        mock_attention_mask = MagicMock()
        mock_input_ids.to.return_value = mock_input_ids
        mock_attention_mask.to.return_value = mock_attention_mask
        tokenized = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
        }

        expected_output = MagicMock()
        mock_model.generate.return_value = expected_output

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from tasks.llm.run_llm_inference import run_llm_inference

            result = run_llm_inference.__wrapped_function__(
                tokenized=tokenized,
                model=mock_model,
                tokenizer=mock_tokenizer,
            )

        mock_model.generate.assert_called_once()
        assert result is expected_output


# ---------------------------------------------------------------------------
# chat_completion
# ---------------------------------------------------------------------------
class TestChatCompletionSmoke:
    """Smoke tests for tasks.llm.chat_completion."""

    def test_importable(self):
        from tasks.llm.chat_completion import chat_completion

        assert hasattr(chat_completion, "__wrapped_function__")

    @pytest.mark.unit
    def test_with_create_chat_completion(self):
        from tasks.llm.chat_completion import chat_completion

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "  42  "}}]
        }
        messages = [{"role": "user", "content": "What is 6x7?"}]

        result = chat_completion.__wrapped_function__(
            model=mock_model, messages=messages
        )
        assert result == "42"

    @pytest.mark.unit
    def test_fallback_without_chat_method(self):
        from tasks.llm.chat_completion import chat_completion

        mock_model = MagicMock(spec=[])  # no create_chat_completion attribute
        mock_model.return_value = {
            "choices": [{"text": " fallback response "}]
        }
        messages = [{"role": "user", "content": "Hello"}]

        result = chat_completion.__wrapped_function__(
            model=mock_model, messages=messages
        )
        assert result == "fallback response"

    @pytest.mark.unit
    def test_empty_messages_returns_empty(self):
        from tasks.llm.chat_completion import chat_completion

        mock_model = MagicMock()
        result = chat_completion.__wrapped_function__(model=mock_model, messages=[])
        assert result == ""


# ---------------------------------------------------------------------------
# configure_llama_lora
# ---------------------------------------------------------------------------
class TestConfigureLlamaLoraSmoke:
    """Smoke tests for tasks.llm.configure_llama_lora."""

    def test_importable(self):
        from tasks.llm.configure_llama_lora import configure_llama_lora

        assert hasattr(configure_llama_lora, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        import sys

        mock_lora_config = MagicMock()
        mock_task_type = MagicMock()
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"
        mock_peft = MagicMock()
        mock_peft.LoraConfig.return_value = mock_lora_config
        mock_peft.TaskType = mock_task_type

        with patch.dict(sys.modules, {"peft": mock_peft}):
            from tasks.llm.configure_llama_lora import configure_llama_lora

            result = configure_llama_lora.__wrapped_function__()

        assert result is mock_lora_config
        mock_peft.LoraConfig.assert_called_once()

    @pytest.mark.unit
    def test_custom_rank_with_mock(self):
        import sys

        mock_lora_config = MagicMock()
        mock_task_type = MagicMock()
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"
        mock_peft = MagicMock()
        mock_peft.LoraConfig.return_value = mock_lora_config
        mock_peft.TaskType = mock_task_type

        with patch.dict(sys.modules, {"peft": mock_peft}):
            from tasks.llm.configure_llama_lora import configure_llama_lora

            result = configure_llama_lora.__wrapped_function__(r=8, lora_alpha=16)

        assert result is mock_lora_config
        call_kwargs = mock_peft.LoraConfig.call_args[1]
        assert call_kwargs["r"] == 8
        assert call_kwargs["lora_alpha"] == 16


# ---------------------------------------------------------------------------
# configure_llama_training_arguments
# ---------------------------------------------------------------------------
class TestConfigureLlamaTrainingArgumentsSmoke:
    """Smoke tests for tasks.llm.configure_llama_training_arguments."""

    def test_importable(self):
        from tasks.llm.configure_llama_training_arguments import (
            configure_llama_training_arguments,
        )

        assert hasattr(configure_llama_training_arguments, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self, tmp_path):
        import sys

        mock_training_args = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.TrainingArguments.return_value = mock_training_args

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            from tasks.llm.configure_llama_training_arguments import (
                configure_llama_training_arguments,
            )

            result = configure_llama_training_arguments.__wrapped_function__(
                output_dir=str(tmp_path)
            )

        assert result is mock_training_args
        mock_transformers.TrainingArguments.assert_called_once()


# ---------------------------------------------------------------------------
# execute_training
# ---------------------------------------------------------------------------
class TestExecuteTrainingSmoke:
    """Smoke tests for tasks.llm.execute_training."""

    def test_importable(self):
        from tasks.llm.execute_training import execute_training

        assert hasattr(execute_training, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.llm.execute_training import execute_training

        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.42
        mock_trained_model = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_train_result
        mock_trainer.model = mock_trained_model

        training_result, trained_model = execute_training.__wrapped_function__(
            trainer=mock_trainer
        )

        mock_trainer.train.assert_called_once()
        assert training_result is mock_train_result
        assert trained_model is mock_trained_model


# ---------------------------------------------------------------------------
# initialize_llama_trainer
# ---------------------------------------------------------------------------
class TestInitializeLlamaTrainerSmoke:
    """Smoke tests for tasks.llm.initialize_llama_trainer."""

    def test_importable(self):
        from tasks.llm.initialize_llama_trainer import initialize_llama_trainer

        assert hasattr(initialize_llama_trainer, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        import sys

        # Create fake parameter objects with numel() so the division doesn't blow up
        mock_param_trainable = MagicMock()
        mock_param_trainable.requires_grad = True
        mock_param_trainable.numel.return_value = 100
        mock_param_frozen = MagicMock()
        mock_param_frozen.requires_grad = False
        mock_param_frozen.numel.return_value = 900

        mock_peft_model = MagicMock()
        mock_peft_model.parameters.return_value = [mock_param_trainable, mock_param_frozen]
        mock_trainer_instance = MagicMock()

        mock_peft = MagicMock()
        mock_peft.get_peft_model.return_value = mock_peft_model

        mock_data_collator = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.DataCollatorForLanguageModeling.return_value = mock_data_collator
        mock_transformers.Trainer.return_value = mock_trainer_instance

        mock_model = MagicMock()
        mock_lora_config = MagicMock()
        mock_training_args = MagicMock()
        mock_training_args.gradient_checkpointing = False
        mock_processed_dataset = MagicMock()
        mock_tokenizer = MagicMock()

        with patch.dict(sys.modules, {"peft": mock_peft, "transformers": mock_transformers}):
            from tasks.llm.initialize_llama_trainer import initialize_llama_trainer

            result = initialize_llama_trainer.__wrapped_function__(
                model=mock_model,
                lora_config=mock_lora_config,
                training_args=mock_training_args,
                processed_dataset=mock_processed_dataset,
                tokenizer=mock_tokenizer,
            )

        assert result is mock_trainer_instance
        mock_peft.get_peft_model.assert_called_once_with(mock_model, mock_lora_config)


# ---------------------------------------------------------------------------
# load_llama_model
# ---------------------------------------------------------------------------
class TestLoadLlamaModelSmoke:
    """Smoke tests for tasks.llm.load_llama_model."""

    def test_importable(self):
        from tasks.llm.load_llama_model import load_llama_model

        assert hasattr(load_llama_model, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        import sys

        mock_model_instance = MagicMock()
        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM = mock_auto_model
        mock_transformers.BitsAndBytesConfig = MagicMock()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            from tasks.llm.load_llama_model import load_llama_model

            result = load_llama_model.__wrapped_function__(
                model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                device_map="cpu",
            )

        assert result is mock_model_instance
        mock_auto_model.from_pretrained.assert_called_once()


# ---------------------------------------------------------------------------
# load_llama_tokenizer
# ---------------------------------------------------------------------------
class TestLoadLlamaTokenizerSmoke:
    """Smoke tests for tasks.llm.load_llama_tokenizer."""

    def test_importable(self):
        from tasks.llm.load_llama_tokenizer import load_llama_tokenizer

        assert hasattr(load_llama_tokenizer, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        import sys

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            from tasks.llm.load_llama_tokenizer import load_llama_tokenizer

            result = load_llama_tokenizer.__wrapped_function__(
                model_name="meta-llama/Meta-Llama-3-8B-Instruct"
            )

        assert result is mock_tokenizer
        mock_auto_tokenizer.from_pretrained.assert_called_once()

    @pytest.mark.unit
    def test_padding_side_set(self):
        import sys

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_auto_tokenizer

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            from tasks.llm.load_llama_tokenizer import load_llama_tokenizer

            result = load_llama_tokenizer.__wrapped_function__(
                model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                padding_side="left",
            )

        assert result.padding_side == "left"


# ---------------------------------------------------------------------------
# load_training_dataset
# ---------------------------------------------------------------------------
class TestLoadTrainingDatasetSmoke:
    """Smoke tests for tasks.llm.load_training_dataset."""

    def test_importable(self):
        from tasks.llm.load_training_dataset import load_training_dataset

        assert hasattr(load_training_dataset, "__wrapped_function__")

    @pytest.mark.unit
    def test_load_by_name_with_mock(self):
        import sys

        mock_dataset = MagicMock()
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_dataset

        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            from tasks.llm.load_training_dataset import load_training_dataset

            result = load_training_dataset.__wrapped_function__(dataset_name="squad")

        assert result is mock_dataset
        mock_datasets.load_dataset.assert_called_once_with(
            "squad", split="train", streaming=False, trust_remote_code=False
        )

    @pytest.mark.unit
    def test_load_by_path_with_mock(self, tmp_path):
        import sys

        mock_dataset = MagicMock()
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_dataset

        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            from tasks.llm.load_training_dataset import load_training_dataset

            result = load_training_dataset.__wrapped_function__(
                dataset_path=str(tmp_path)
            )

        assert result is mock_dataset

    @pytest.mark.unit
    def test_no_source_raises(self):
        import sys

        mock_datasets = MagicMock()
        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            from tasks.llm.load_training_dataset import load_training_dataset

            with pytest.raises(ValueError, match="Either dataset_name or dataset_path"):
                load_training_dataset.__wrapped_function__()


# ---------------------------------------------------------------------------
# prepare_llama_dataset_for_training
# ---------------------------------------------------------------------------
class TestPrepareLlamaDatasetForTrainingSmoke:
    """Smoke tests for tasks.llm.prepare_llama_dataset_for_training."""

    def test_importable(self):
        from tasks.llm.prepare_llama_dataset_for_training import (
            prepare_llama_dataset_for_training,
        )

        assert hasattr(prepare_llama_dataset_for_training, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        from tasks.llm.prepare_llama_dataset_for_training import (
            prepare_llama_dataset_for_training,
        )

        mock_processed = MagicMock()
        mock_processed.__len__ = MagicMock(return_value=10)

        mock_formatted = MagicMock()
        mock_formatted.column_names = ["messages"]
        mock_formatted.map.return_value = mock_processed

        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_formatted

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted text"
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }

        result = prepare_llama_dataset_for_training.__wrapped_function__(
            dataset=mock_dataset,
            tokenizer=mock_tokenizer,
            max_length=512,
            num_proc=1,
        )

        assert result is mock_processed
        mock_dataset.map.assert_called_once()


# ---------------------------------------------------------------------------
# save_llama_model
# ---------------------------------------------------------------------------
class TestSaveLlamaModelSmoke:
    """Smoke tests for tasks.llm.save_llama_model."""

    def test_importable(self):
        from tasks.llm.save_llama_model import save_llama_model

        assert hasattr(save_llama_model, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self, tmp_path):
        from tasks.llm.save_llama_model import save_llama_model

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        output_dir = str(tmp_path / "model_output")

        result = save_llama_model.__wrapped_function__(
            trained_model=mock_model,
            tokenizer=mock_tokenizer,
            output_dir=output_dir,
        )

        assert result == output_dir
        mock_model.save_pretrained.assert_called_once_with(output_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(output_dir)

    @pytest.mark.unit
    def test_creates_output_directory(self, tmp_path):
        from tasks.llm.save_llama_model import save_llama_model

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        output_dir = str(tmp_path / "nested" / "model_dir")

        save_llama_model.__wrapped_function__(
            trained_model=mock_model,
            tokenizer=mock_tokenizer,
            output_dir=output_dir,
        )

        import os
        assert os.path.isdir(output_dir)
