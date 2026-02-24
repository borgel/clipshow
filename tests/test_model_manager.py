"""TDD tests for ModelManager (clipshow/detection/models.py).

Tests written BEFORE implementation (red phase). These define the expected
interface and behavior for ModelManager — the shared ONNX model download,
cache, and session loading infrastructure.

All tests mock network calls — no real downloads in CI.
Tests will initially fail (red); clipshow-hph implements to make them pass (green).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import clipshow.detection.models as models_mod
from clipshow.detection.models import MODEL_REGISTRY, ModelManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ort():
    """Ensure clipshow.detection.models.ort is a mock module (for CI without onnxruntime)."""
    mock = MagicMock()
    mock.get_available_providers.return_value = ["CPUExecutionProvider"]
    with patch.object(models_mod, "ort", mock):
        yield mock


@pytest.fixture
def tmp_cache(tmp_path):
    """Temp directory used as model cache for isolation."""
    return tmp_path / "models"


@pytest.fixture
def manager(tmp_cache):
    """ModelManager with cache_dir pointed at a temp directory."""
    return ModelManager(cache_dir=tmp_cache)


def _first_model():
    """Return (name, metadata) for the first model in the registry."""
    name = next(iter(MODEL_REGISTRY))
    return name, MODEL_REGISTRY[name]


def _populate_cache(cache_dir: Path, filename: str, data: bytes = b"fake-model-data"):
    """Write a fake model file into the cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    path.write_bytes(data)
    return path


def _fake_download(url, dest, reporthook=None):
    """Side-effect for urlretrieve that writes fake data."""
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(b"downloaded-onnx-bytes")
    if reporthook:
        reporthook(0, 8192, 100000)
        reporthook(1, 8192, 100000)


# ---------------------------------------------------------------------------
# MODEL_REGISTRY structure
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """MODEL_REGISTRY should contain metadata for all known models."""

    def test_registry_is_dict(self):
        assert isinstance(MODEL_REGISTRY, dict)

    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_entries_have_required_fields(self):
        required = {"file", "url", "size_mb"}
        for name, meta in MODEL_REGISTRY.items():
            missing = required - set(meta.keys())
            assert not missing, f"Model {name!r} missing fields: {missing}"

    def test_entries_have_valid_types(self):
        for name, meta in MODEL_REGISTRY.items():
            assert isinstance(meta["file"], str), f"{name}: file should be str"
            assert isinstance(meta["url"], str), f"{name}: url should be str"
            assert isinstance(
                meta["size_mb"], (int, float)
            ), f"{name}: size_mb should be numeric"

    def test_filenames_end_with_onnx(self):
        for name, meta in MODEL_REGISTRY.items():
            assert meta["file"].endswith(".onnx"), f"{name}: file should end with .onnx"


# ---------------------------------------------------------------------------
# ModelManager construction
# ---------------------------------------------------------------------------


class TestModelManagerInit:
    """ModelManager should be configurable for cache directory."""

    def test_default_cache_dir(self):
        mm = ModelManager()
        assert mm.cache_dir == Path.home() / ".clipshow" / "models"

    def test_custom_cache_dir(self, tmp_cache):
        mm = ModelManager(cache_dir=tmp_cache)
        assert mm.cache_dir == tmp_cache


# ---------------------------------------------------------------------------
# ensure_model — download / cache logic
# ---------------------------------------------------------------------------


class TestEnsureModel:
    """ensure_model should download if missing, skip if cached."""

    def test_unknown_model_raises_key_error(self, manager):
        """Requesting a model not in the registry should raise KeyError."""
        with pytest.raises(KeyError):
            manager.ensure_model("nonexistent-model-xyz")

    def test_cached_model_returns_path_immediately(self, manager, tmp_cache):
        """If the model file already exists in cache, return path without downloading."""
        name, meta = _first_model()
        cached_file = _populate_cache(tmp_cache, meta["file"])

        with patch("urllib.request.urlretrieve") as mock_dl:
            path = manager.ensure_model(name)

        assert path == cached_file
        mock_dl.assert_not_called()

    def test_missing_model_downloads_to_cache(self, manager, tmp_cache):
        """If model file is absent, ensure_model downloads it."""
        name, meta = _first_model()
        expected_path = tmp_cache / meta["file"]

        with patch(
            "urllib.request.urlretrieve", side_effect=_fake_download
        ) as mock_dl:
            path = manager.ensure_model(name)

        assert path == expected_path
        assert path.exists()
        mock_dl.assert_called_once()
        # URL should match registry
        assert mock_dl.call_args[0][0] == meta["url"]

    def test_creates_cache_dir_if_missing(self, manager, tmp_cache):
        """ensure_model should create cache_dir when it doesn't exist."""
        assert not tmp_cache.exists()
        name, _ = _first_model()

        with patch("urllib.request.urlretrieve", side_effect=_fake_download):
            manager.ensure_model(name)

        assert tmp_cache.is_dir()

    def test_progress_callback_invoked(self, manager, tmp_cache):
        """Progress callback should fire at least once during download."""
        name, _ = _first_model()
        progress_values = []

        with patch("urllib.request.urlretrieve", side_effect=_fake_download):
            manager.ensure_model(
                name, progress_cb=lambda p: progress_values.append(p)
            )

        assert len(progress_values) > 0, "Progress callback never called"

    def test_progress_values_between_zero_and_one(self, manager, tmp_cache):
        """Progress values should be in the range [0, 1]."""
        name, _ = _first_model()
        progress_values = []

        with patch("urllib.request.urlretrieve", side_effect=_fake_download):
            manager.ensure_model(
                name, progress_cb=lambda p: progress_values.append(p)
            )

        for v in progress_values:
            assert 0.0 <= v <= 1.0, f"Progress value {v} out of [0, 1] range"

    def test_download_failure_raises(self, manager):
        """Network failure during download should propagate as an exception."""
        name, _ = _first_model()

        with patch(
            "urllib.request.urlretrieve", side_effect=OSError("network error")
        ):
            with pytest.raises((OSError, RuntimeError)):
                manager.ensure_model(name)

    def test_corrupt_empty_file_redownloads(self, manager, tmp_cache):
        """An empty (0-byte) cached file should trigger re-download."""
        name, meta = _first_model()
        _populate_cache(tmp_cache, meta["file"], data=b"")  # empty = corrupt

        with patch(
            "urllib.request.urlretrieve", side_effect=_fake_download
        ) as mock_dl:
            path = manager.ensure_model(name)

        mock_dl.assert_called_once()
        assert path.stat().st_size > 0

    def test_returns_path_object(self, manager, tmp_cache):
        """ensure_model should return a pathlib.Path."""
        name, meta = _first_model()
        _populate_cache(tmp_cache, meta["file"])

        path = manager.ensure_model(name)
        assert isinstance(path, Path)


# ---------------------------------------------------------------------------
# _get_providers — execution provider detection
# ---------------------------------------------------------------------------


class TestGetProviders:
    """_get_providers should detect and prioritize execution providers."""

    def test_returns_list(self, manager, mock_ort):
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        providers = manager._get_providers()
        assert isinstance(providers, list)

    def test_cpu_always_present(self, manager, mock_ort):
        """CPUExecutionProvider should always be included when available."""
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        providers = manager._get_providers()
        assert "CPUExecutionProvider" in providers

    def test_cuda_preferred_over_cpu(self, manager, mock_ort):
        """CUDA should appear before CPU when both are available."""
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ]
        providers = manager._get_providers()

        cuda_idx = providers.index("CUDAExecutionProvider")
        cpu_idx = providers.index("CPUExecutionProvider")
        assert cuda_idx < cpu_idx

    def test_coreml_preferred_over_cpu(self, manager, mock_ort):
        """CoreML should appear before CPU when both are available."""
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CoreMLExecutionProvider",
        ]
        providers = manager._get_providers()

        coreml_idx = providers.index("CoreMLExecutionProvider")
        cpu_idx = providers.index("CPUExecutionProvider")
        assert coreml_idx < cpu_idx

    def test_cuda_preferred_over_coreml(self, manager, mock_ort):
        """Full priority order: CUDA > CoreML > CPU."""
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
        ]
        providers = manager._get_providers()

        assert providers.index("CUDAExecutionProvider") < providers.index(
            "CoreMLExecutionProvider"
        )
        assert providers.index("CoreMLExecutionProvider") < providers.index(
            "CPUExecutionProvider"
        )

    def test_unavailable_providers_excluded(self, manager, mock_ort):
        """Providers not reported by ORT should not appear in the list."""
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        providers = manager._get_providers()

        assert "CUDAExecutionProvider" not in providers
        assert "CoreMLExecutionProvider" not in providers


# ---------------------------------------------------------------------------
# load_session — ONNX Runtime session creation
# ---------------------------------------------------------------------------


class TestLoadSession:
    """load_session should create an InferenceSession with correct providers."""

    def test_returns_inference_session(self, manager, tmp_cache, mock_ort):
        """load_session should return an ort.InferenceSession."""
        name, meta = _first_model()
        _populate_cache(tmp_cache, meta["file"])

        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        with patch.object(
            manager, "_get_providers", return_value=["CPUExecutionProvider"]
        ):
            session = manager.load_session(name)

        assert session is mock_session

    def test_passes_providers_to_session(self, manager, tmp_cache, mock_ort):
        """load_session should pass detected providers to InferenceSession."""
        name, meta = _first_model()
        _populate_cache(tmp_cache, meta["file"])

        expected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with patch.object(
            manager, "_get_providers", return_value=expected_providers
        ):
            manager.load_session(name)

        call_kwargs = mock_ort.InferenceSession.call_args.kwargs
        assert call_kwargs.get("providers") == expected_providers

    def test_passes_model_path_to_session(self, manager, tmp_cache, mock_ort):
        """load_session should pass the correct model file path as string."""
        name, meta = _first_model()
        expected_path = _populate_cache(tmp_cache, meta["file"])

        with patch.object(
            manager, "_get_providers", return_value=["CPUExecutionProvider"]
        ):
            manager.load_session(name)

        # First positional arg should be the model path as a string
        call_args = mock_ort.InferenceSession.call_args.args
        assert call_args[0] == str(expected_path)

    def test_delegates_to_ensure_model(self, manager, mock_ort):
        """load_session should call ensure_model to get the model path."""
        name, _ = _first_model()

        with patch.object(
            manager, "ensure_model", return_value=Path("/fake/model.onnx")
        ) as mock_ensure:
            with patch.object(
                manager,
                "_get_providers",
                return_value=["CPUExecutionProvider"],
            ):
                manager.load_session(name)

        mock_ensure.assert_called_once_with(name)

    def test_unknown_model_raises(self, manager, mock_ort):
        """load_session with unknown model should raise KeyError (via ensure_model)."""
        with pytest.raises(KeyError):
            manager.load_session("nonexistent-model-xyz")


# ---------------------------------------------------------------------------
# Integration test — real (tiny) ONNX model
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration test loading a real (tiny) ONNX model file."""

    @pytest.fixture
    def tiny_onnx_path(self, tmp_path):
        """Create a minimal valid ONNX model (single Identity node)."""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "test_graph", [X], [Y])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        model.ir_version = 7

        path = tmp_path / "tiny_model.onnx"
        onnx.save(model, str(path))
        return path

    def test_load_real_onnx_session(self, tmp_path, tiny_onnx_path):
        """ModelManager can load a real ONNX file into a working InferenceSession."""
        ort = pytest.importorskip("onnxruntime")

        # Set up cache with our tiny model
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        import shutil

        cached = cache_dir / "tiny_model.onnx"
        shutil.copy(tiny_onnx_path, cached)

        mm = ModelManager(cache_dir=cache_dir)

        # Temporarily register our test model
        MODEL_REGISTRY["_test_tiny"] = {
            "file": "tiny_model.onnx",
            "url": "http://localhost/fake",
            "size_mb": 0,
        }

        try:
            session = mm.load_session("_test_tiny")
            assert isinstance(session, ort.InferenceSession)

            # Verify the session actually runs inference
            input_data = np.array([[1.0, 2.0]], dtype=np.float32)
            result = session.run(None, {"X": input_data})
            np.testing.assert_array_almost_equal(result[0], [[1.0, 2.0]])
        finally:
            MODEL_REGISTRY.pop("_test_tiny", None)

    def test_end_to_end_download_and_load(self, tmp_path, tiny_onnx_path):
        """Full flow: ensure_model downloads, load_session creates session."""
        ort = pytest.importorskip("onnxruntime")

        cache_dir = tmp_path / "e2e_cache"
        mm = ModelManager(cache_dir=cache_dir)

        MODEL_REGISTRY["_test_e2e"] = {
            "file": "e2e_model.onnx",
            "url": "http://localhost/fake/e2e_model.onnx",
            "size_mb": 0,
        }

        try:
            # Mock download to copy our real tiny model
            def fake_dl(url, dest, reporthook=None):
                import shutil

                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(tiny_onnx_path, dest)

            with patch("urllib.request.urlretrieve", side_effect=fake_dl):
                session = mm.load_session("_test_e2e")

            assert isinstance(session, ort.InferenceSession)
            input_data = np.array([[3.0, 4.0]], dtype=np.float32)
            result = session.run(None, {"X": input_data})
            np.testing.assert_array_almost_equal(result[0], [[3.0, 4.0]])
        finally:
            MODEL_REGISTRY.pop("_test_e2e", None)
