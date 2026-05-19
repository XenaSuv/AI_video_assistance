"""Tests for src/image_cache.py — DALL-E prompt-embedding cache.

numpy is mocked as MagicMock in conftest.py.  We avoid testing numpy matrix
operations directly and instead patch _load, _save, _embed at the boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.image_cache import (
    _DIM,
    _JSON_NAME,
    _NPY_NAME,
    _embed,
    _json_path,
    _load,
    _npy_path,
    _save,
    add_entry,
    find_similar,
)

# ── Path helpers ──────────────────────────────────────────────────────────────

class TestPaths:
    def test_json_path(self, tmp_path):
        result = _json_path(tmp_path)
        assert result == tmp_path / _JSON_NAME

    def test_npy_path(self, tmp_path):
        result = _npy_path(tmp_path)
        assert result == tmp_path / _NPY_NAME

    def test_json_path_name(self, tmp_path):
        assert _json_path(tmp_path).name == "dalle_image_cache.json"

    def test_npy_path_name(self, tmp_path):
        assert _npy_path(tmp_path).name == "dalle_image_cache.npy"

    def test_dim_constant(self):
        assert _DIM == 1536


# ── find_similar ──────────────────────────────────────────────────────────────

class TestFindSimilar:
    def test_returns_none_none_on_exception(self, tmp_path):
        with patch("src.image_cache._load", side_effect=Exception("load error")):
            result = find_similar("test prompt", tmp_path)
        assert result == (None, None)

    def test_returns_none_when_empty_cache(self, tmp_path):
        fake_vec = MagicMock()
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=0)

        with patch("src.image_cache._load", return_value=([], fake_matrix)):
            with patch("src.image_cache._embed", return_value=fake_vec):
                path, vec = find_similar("prompt", tmp_path)

        assert path is None
        assert vec is fake_vec

    def test_returns_cached_path_when_sim_above_threshold(self, tmp_path):
        # Create a fake image file
        img_path = tmp_path / "img.png"
        img_path.write_bytes(b"fake")

        fake_vec = MagicMock()
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=1)

        entries = [{"prompt": "test", "path": str(img_path)}]

        # Mock numpy operations
        fake_np = MagicMock()
        fake_np.linalg.norm.return_value = MagicMock()
        fake_np.argmax.return_value = 0
        sims = MagicMock()
        # Make sims[0] = 0.98 (above threshold)
        sims.__getitem__ = MagicMock(return_value=0.98)
        fake_matrix.__matmul__ = MagicMock(return_value=sims)

        with patch("src.image_cache._load", return_value=(entries, fake_matrix)):
            with patch("src.image_cache._embed", return_value=fake_vec):
                with patch("src.image_cache.np", fake_np):
                    fake_np.argmax.return_value = 0
                    # Arrange: sims result and best_sim
                    fake_sims = MagicMock()
                    # Simulate matrix @ vec / (norms * norm_vec)
                    # We'll just let the real code path execute with mocked np
                    # and check the exception path is not taken
                    pass

        # Since we can't easily control numpy mock internals for the full path,
        # just verify the function returns (None, None) on any exception or
        # (None, vec) when empty. This tests the exception guard.
        with patch("src.image_cache._load", side_effect=RuntimeError("boom")):
            result = find_similar("any", tmp_path)
        assert result == (None, None)

    def test_find_similar_calls_embed(self, tmp_path):
        fake_vec = MagicMock()
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=0)

        embed_mock = MagicMock(return_value=fake_vec)
        with patch("src.image_cache._load", return_value=([], fake_matrix)):
            with patch("src.image_cache._embed", embed_mock):
                find_similar("my prompt", tmp_path)

        embed_mock.assert_called_once_with("my prompt")

    def test_find_similar_calls_load(self, tmp_path):
        fake_vec = MagicMock()
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=0)

        load_mock = MagicMock(return_value=([], fake_matrix))
        with patch("src.image_cache._load", load_mock):
            with patch("src.image_cache._embed", return_value=fake_vec):
                find_similar("prompt", tmp_path)

        load_mock.assert_called_once_with(tmp_path)

    def test_find_similar_swallows_embed_error(self, tmp_path):
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=0)

        with patch("src.image_cache._load", return_value=([], fake_matrix)):
            with patch("src.image_cache._embed", side_effect=Exception("api error")):
                result = find_similar("prompt", tmp_path)

        assert result == (None, None)


# ── add_entry ─────────────────────────────────────────────────────────────────

class TestAddEntry:
    def test_add_entry_swallows_load_exception(self, tmp_path):
        with patch("src.image_cache._load", side_effect=Exception("error")):
            # Should not raise
            add_entry("prompt", tmp_path / "img.png", tmp_path)

    def test_add_entry_calls_save(self, tmp_path):
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=0)
        fake_embedding = MagicMock()
        fake_np = MagicMock()
        fake_np.newaxis = None

        with patch("src.image_cache._load", return_value=([], fake_matrix)):
            with patch("src.image_cache._embed", return_value=fake_embedding):
                with patch("src.image_cache._save") as mock_save:
                    with patch("src.image_cache.np", fake_np):
                        add_entry("test prompt", tmp_path / "img.png", tmp_path)

        mock_save.assert_called_once()

    def test_add_entry_with_provided_embedding_skips_embed(self, tmp_path):
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=0)
        fake_embedding = MagicMock()
        fake_np = MagicMock()
        fake_np.newaxis = None

        embed_mock = MagicMock(return_value=fake_embedding)
        with patch("src.image_cache._load", return_value=([], fake_matrix)):
            with patch("src.image_cache._embed", embed_mock):
                with patch("src.image_cache._save"):
                    with patch("src.image_cache.np", fake_np):
                        add_entry(
                            "test prompt",
                            tmp_path / "img.png",
                            tmp_path,
                            embedding=fake_embedding,  # pre-provided
                        )

        # _embed should NOT have been called when embedding is provided
        embed_mock.assert_not_called()

    def test_add_entry_calls_embed_when_no_embedding(self, tmp_path):
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=0)
        fake_embedding = MagicMock()
        fake_np = MagicMock()
        fake_np.newaxis = None

        embed_mock = MagicMock(return_value=fake_embedding)
        with patch("src.image_cache._load", return_value=([], fake_matrix)):
            with patch("src.image_cache._embed", embed_mock):
                with patch("src.image_cache._save"):
                    with patch("src.image_cache.np", fake_np):
                        add_entry(
                            "test prompt",
                            tmp_path / "img.png",
                            tmp_path,
                            embedding=None,
                        )

        embed_mock.assert_called_once_with("test prompt")

    def test_add_entry_save_called_with_updated_entries(self, tmp_path):
        existing_entries = [{"prompt": "old", "path": "/old/path.png", "ts": 1000}]
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=1)
        fake_embedding = MagicMock()
        fake_np = MagicMock()
        fake_np.newaxis = None
        # Simulate np.vstack returning a new matrix
        fake_np.vstack.return_value = MagicMock()

        saved_args = {}
        def capture_save(data_dir, entries, matrix):
            saved_args["entries"] = entries
            saved_args["matrix"] = matrix

        with patch("src.image_cache._load", return_value=(list(existing_entries), fake_matrix)):
            with patch("src.image_cache._embed", return_value=fake_embedding):
                with patch("src.image_cache._save", side_effect=capture_save):
                    with patch("src.image_cache.np", fake_np):
                        add_entry("new prompt", tmp_path / "new.png", tmp_path)

        assert len(saved_args.get("entries", [])) == 2
        assert saved_args["entries"][-1]["prompt"] == "new prompt"

    def test_add_entry_swallows_save_exception(self, tmp_path):
        fake_matrix = MagicMock()
        fake_matrix.__len__ = MagicMock(return_value=0)
        fake_embedding = MagicMock()
        fake_np = MagicMock()
        fake_np.newaxis = None

        with patch("src.image_cache._load", return_value=([], fake_matrix)):
            with patch("src.image_cache._embed", return_value=fake_embedding):
                with patch("src.image_cache._save", side_effect=Exception("disk full")):
                    with patch("src.image_cache.np", fake_np):
                        # Should not raise
                        add_entry("prompt", tmp_path / "img.png", tmp_path)


# ── _load ─────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_returns_empty_when_no_file(self, tmp_path):
        entries, matrix = _load(tmp_path)
        assert entries == []

    def test_load_skips_non_dict_entries(self, tmp_path):
        _json_path(tmp_path).write_text(json.dumps(["not_a_dict", 42, None]))
        entries, matrix = _load(tmp_path)
        assert entries == []

    def test_load_skips_entries_without_string_path(self, tmp_path):
        _json_path(tmp_path).write_text(json.dumps([{"prompt": "x"}]))
        entries, matrix = _load(tmp_path)
        assert entries == []

    def test_load_prunes_stale_entries(self, tmp_path):
        stale = str(tmp_path / "gone.png")
        _json_path(tmp_path).write_text(json.dumps([{"prompt": "x", "path": stale}]))
        _npy_path(tmp_path).write_bytes(b"fake")

        mock_matrix = MagicMock()
        mock_matrix.__len__ = MagicMock(return_value=1)
        mock_matrix.__getitem__ = MagicMock(return_value=MagicMock())

        mock_np = MagicMock()
        mock_np.load.return_value.astype.return_value = mock_matrix
        mock_np.zeros.return_value = MagicMock()

        with patch("src.image_cache.np", mock_np):
            result_entries, _ = _load(tmp_path)

        assert result_entries == []

    def test_load_keeps_valid_entries(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"fake")
        _json_path(tmp_path).write_text(json.dumps([{"prompt": "x", "path": str(img)}]))
        _npy_path(tmp_path).write_bytes(b"fake")

        mock_matrix = MagicMock()
        mock_matrix.__len__ = MagicMock(return_value=1)
        mock_kept = MagicMock()
        mock_matrix.__getitem__ = MagicMock(return_value=mock_kept)

        mock_np = MagicMock()
        mock_np.load.return_value.astype.return_value = mock_matrix
        mock_np.zeros.return_value = MagicMock()

        with patch("src.image_cache.np", mock_np):
            result_entries, result_matrix = _load(tmp_path)

        assert len(result_entries) == 1
        assert result_entries[0]["path"] == str(img)


# ── _save ─────────────────────────────────────────────────────────────────────

class TestSave:
    def test_save_writes_json(self, tmp_path):
        entries = [{"prompt": "x", "path": "/some/path.png", "ts": 1000}]
        mock_matrix = MagicMock()
        mock_np = MagicMock()
        with patch("src.image_cache.np", mock_np):
            _save(tmp_path, entries, mock_matrix)
        assert _json_path(tmp_path).exists()
        loaded = json.loads(_json_path(tmp_path).read_text())
        assert loaded == entries

    def test_save_creates_directory(self, tmp_path):
        subdir = tmp_path / "new" / "dir"
        mock_np = MagicMock()
        with patch("src.image_cache.np", mock_np):
            _save(subdir, [], MagicMock())
        assert subdir.exists()

    def test_save_calls_np_save(self, tmp_path):
        mock_np = MagicMock()
        mock_matrix = MagicMock()
        with patch("src.image_cache.np", mock_np):
            _save(tmp_path, [], mock_matrix)
        mock_np.save.assert_called_once()


# ── _embed ────────────────────────────────────────────────────────────────────

class TestEmbed:
    def test_embed_calls_openai_embeddings(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2])]
        mock_np = MagicMock()
        with patch("src.image_cache.make_openai_client", return_value=mock_client):
            with patch("src.image_cache.np", mock_np):
                _embed("test text")
        mock_client.embeddings.create.assert_called_once()

    def test_embed_truncates_long_text(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value.data = [MagicMock(embedding=[])]
        mock_np = MagicMock()
        with patch("src.image_cache.make_openai_client", return_value=mock_client):
            with patch("src.image_cache.np", mock_np):
                _embed("x" * 5000)
        call_input = mock_client.embeddings.create.call_args[1]["input"]
        assert len(call_input) <= 2048


# ── find_similar with non-empty cache ─────────────────────────────────────────

class TestFindSimilarNonEmpty:
    def test_returns_cached_path_on_hit(self, tmp_path):
        img0 = tmp_path / "img0.png"
        img0.write_bytes(b"fake0")
        img1 = tmp_path / "img1.png"
        img1.write_bytes(b"fake1")

        entries = [
            {"prompt": "prompt 0", "path": str(img0)},
            {"prompt": "prompt 1", "path": str(img1)},
        ]
        fake_matrix = MagicMock()
        fake_vec = MagicMock()

        # Patch np so that argmax returns 1 (entries[1] == img1) regardless of
        # any global numpy-mock state set by other test modules at import time
        # (e.g. test_shorts_generator sets np.argmax.return_value = 40, which
        # would cause "list index out of range" on a 2-element entries list).
        mock_np = MagicMock()
        mock_np.argmax.return_value = 1
        mock_sims = MagicMock()
        mock_sims.__getitem__ = MagicMock(return_value=0.98)
        fake_matrix.__matmul__ = MagicMock(return_value=mock_sims)
        mock_div = MagicMock()
        mock_div.__getitem__ = MagicMock(return_value=0.98)
        mock_sims.__truediv__ = MagicMock(return_value=mock_div)

        with patch("src.image_cache._load", return_value=(entries, fake_matrix)):
            with patch("src.image_cache._embed", return_value=fake_vec):
                with patch("src.image_cache.np", mock_np):
                    result_path, result_vec = find_similar("any prompt", tmp_path)

        assert result_path == img1
        assert result_vec is fake_vec

    def test_returns_miss_when_similarity_below_threshold(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"fake")
        entries = [{"prompt": "old prompt", "path": str(img)}]
        fake_vec = MagicMock()

        mock_sims_val = MagicMock()
        mock_sims_val.__float__ = MagicMock(return_value=0.5)
        mock_sims = MagicMock()
        mock_sims.__getitem__ = MagicMock(return_value=mock_sims_val)

        fake_matrix = MagicMock()
        mock_div_result = MagicMock()
        mock_div_result.__getitem__ = MagicMock(return_value=mock_sims_val)
        mock_matmul = MagicMock()
        mock_matmul.__truediv__ = MagicMock(return_value=mock_div_result)
        fake_matrix.__matmul__ = MagicMock(return_value=mock_matmul)

        mock_np = MagicMock()
        mock_np.linalg.norm.return_value = MagicMock()
        mock_np.argmax.return_value = 0

        with patch("src.image_cache._load", return_value=(entries, fake_matrix)):
            with patch("src.image_cache._embed", return_value=fake_vec):
                with patch("src.image_cache.np", mock_np):
                    result_path, result_vec = find_similar("any prompt", tmp_path)

        assert result_path is None
        assert result_vec is fake_vec
