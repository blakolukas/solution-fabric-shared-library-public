"""Smoke tests for tasks/array/ — verify importability and basic execution."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# batch_array
# ---------------------------------------------------------------------------
class TestBatchArraySmoke:
    """Smoke tests for tasks.array.batch_array."""

    def test_importable(self):
        from tasks.array.batch_array import batch_array

        assert hasattr(batch_array, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.array.batch_array import batch_array

        result = batch_array.__wrapped_function__(np.array([1, 2, 3]))
        assert result.shape == (1, 3)

    @pytest.mark.unit
    def test_custom_axis(self):
        from tasks.array.batch_array import batch_array

        result = batch_array.__wrapped_function__(np.array([1, 2, 3]), axis=1)
        assert result.shape == (3, 1)


# ---------------------------------------------------------------------------
# convert_array_dtype
# ---------------------------------------------------------------------------
class TestConvertArrayDtypeSmoke:
    """Smoke tests for tasks.array.convert_array_dtype."""

    def test_importable(self):
        from tasks.array.convert_array_dtype import convert_array_dtype

        assert hasattr(convert_array_dtype, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.array.convert_array_dtype import convert_array_dtype

        result = convert_array_dtype.__wrapped_function__(np.array([1, 2, 3]), "float32")
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# convert_channel_format
# ---------------------------------------------------------------------------
class TestConvertChannelFormatSmoke:
    """Smoke tests for tasks.array.convert_between_channel_first_or_last."""

    def test_importable(self):
        from tasks.array.convert_between_channel_first_or_last import convert_channel_format

        assert hasattr(convert_channel_format, "__wrapped_function__")

    @pytest.mark.unit
    def test_channel_first(self):
        from tasks.array.convert_between_channel_first_or_last import convert_channel_format

        arr = np.ones((3, 4, 5))  # (H, W, C) where C=5
        result = convert_channel_format.__wrapped_function__(arr, "channel_first")
        assert result.shape == (5, 3, 4)

    @pytest.mark.unit
    def test_channel_last(self):
        from tasks.array.convert_between_channel_first_or_last import convert_channel_format

        arr = np.ones((5, 3, 4))  # (C, H, W)
        result = convert_channel_format.__wrapped_function__(arr, "channel_last")
        assert result.shape == (3, 4, 5)


# ---------------------------------------------------------------------------
# ensure_array_contiguous
# ---------------------------------------------------------------------------
class TestEnsureArrayContiguousSmoke:
    """Smoke tests for tasks.array.ensure_array_contiguous."""

    def test_importable(self):
        from tasks.array.ensure_array_contiguous import ensure_array_contiguous

        assert hasattr(ensure_array_contiguous, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.array.ensure_array_contiguous import ensure_array_contiguous

        arr = np.array([1, 2, 3])
        result = ensure_array_contiguous.__wrapped_function__(arr)
        assert result.flags.c_contiguous

    @pytest.mark.unit
    def test_non_contiguous_input(self):
        from tasks.array.ensure_array_contiguous import ensure_array_contiguous

        arr = np.asfortranarray(np.ones((3, 4)))
        assert not arr.flags.c_contiguous
        result = ensure_array_contiguous.__wrapped_function__(arr)
        assert result.flags.c_contiguous


# ---------------------------------------------------------------------------
# extract_array_hw
# ---------------------------------------------------------------------------
class TestExtractArrayHwSmoke:
    """Smoke tests for tasks.array.extract_array_hw."""

    def test_importable(self):
        from tasks.array.extract_array_hw import extract_array_hw

        assert hasattr(extract_array_hw, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.array.extract_array_hw import extract_array_hw

        result = extract_array_hw.__wrapped_function__(np.ones((100, 200, 3)))
        assert result == (100, 200)


# ---------------------------------------------------------------------------
# extract_first_element
# ---------------------------------------------------------------------------
class TestExtractFirstElementSmoke:
    """Smoke tests for tasks.array.extract_first_element."""

    def test_importable(self):
        from tasks.array.extract_first_element import extract_first_element

        assert hasattr(extract_first_element, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.array.extract_first_element import extract_first_element

        result = extract_first_element.__wrapped_function__([42, 2, 3])
        assert result == 42

    @pytest.mark.unit
    def test_numpy_passthrough(self):
        from tasks.array.extract_first_element import extract_first_element

        arr = np.array([10, 20, 30])
        result = extract_first_element.__wrapped_function__(arr)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# extract_hw_from_shape
# ---------------------------------------------------------------------------
class TestExtractHwFromShapeSmoke:
    """Smoke tests for tasks.array.extract_hw_from_shape."""

    def test_importable(self):
        from tasks.array.extract_hw_from_shape import extract_hw_from_shape

        assert hasattr(extract_hw_from_shape, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.array.extract_hw_from_shape import extract_hw_from_shape

        result = extract_hw_from_shape.__wrapped_function__((1, 3, 224, 224))
        assert result == (224, 224)


# ---------------------------------------------------------------------------
# unbatch_array
# ---------------------------------------------------------------------------
class TestUnbatchArraySmoke:
    """Smoke tests for tasks.array.unbatch_array."""

    def test_importable(self):
        from tasks.array.unbatch_array import unbatch_array

        assert hasattr(unbatch_array, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.array.unbatch_array import unbatch_array

        result = unbatch_array.__wrapped_function__(np.array([[1, 2, 3]]))
        assert result.shape == (3,)
