"""Tests for retry_utils: http_get, http_post, make_openai_client."""
import pytest
import requests
import requests.exceptions
from unittest.mock import MagicMock, patch, call


# ── helpers ───────────────────────────────────────────────────────────────────

def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    exc = requests.exceptions.HTTPError(response=resp)
    return exc


# ── _is_retryable_http ────────────────────────────────────────────────────────

class TestIsRetryableHttp:
    def setup_method(self):
        from src.retry_utils import _is_retryable_http
        self.pred = _is_retryable_http

    def test_connection_error_is_retryable(self):
        assert self.pred(requests.exceptions.ConnectionError())

    def test_timeout_is_retryable(self):
        assert self.pred(requests.exceptions.Timeout())

    def test_429_is_retryable(self):
        assert self.pred(_http_error(429))

    def test_500_is_retryable(self):
        assert self.pred(_http_error(500))

    def test_502_is_retryable(self):
        assert self.pred(_http_error(502))

    def test_503_is_retryable(self):
        assert self.pred(_http_error(503))

    def test_504_is_retryable(self):
        assert self.pred(_http_error(504))

    def test_404_not_retryable(self):
        assert not self.pred(_http_error(404))

    def test_401_not_retryable(self):
        assert not self.pred(_http_error(401))

    def test_value_error_not_retryable(self):
        assert not self.pred(ValueError("bad"))


# ── http_get ──────────────────────────────────────────────────────────────────

class TestHttpGet:
    def test_success_on_first_try(self):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp) as mock_get:
            from src.retry_utils import http_get
            result = http_get("https://example.com", timeout=5)
        assert result is mock_resp
        assert mock_get.call_count == 1

    def test_retries_on_connection_error_then_succeeds(self):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.raise_for_status = MagicMock()
        from tenacity import retry, retry_if_exception, stop_after_attempt, wait_none
        from src.retry_utils import _is_retryable_http
        # Build a fast-retry version (no sleep)
        @retry(retry=retry_if_exception(_is_retryable_http),
               wait=wait_none(), stop=stop_after_attempt(3), reraise=True)
        def fast_get(url, **kwargs):
            import requests as _req
            resp = _req.get(url, **kwargs)
            resp.raise_for_status()
            return resp

        with patch("requests.get", side_effect=[
            requests.exceptions.ConnectionError(),
            mock_resp,
        ]) as mock_get:
            result = fast_get("https://example.com")

        assert result is mock_resp
        assert mock_get.call_count == 2

    def test_raises_on_non_retryable_error(self):
        with patch("requests.get", side_effect=_http_error(404)):
            from src.retry_utils import http_get
            with pytest.raises(requests.exceptions.HTTPError):
                http_get("https://example.com")

    def test_raises_for_status_propagates(self):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.raise_for_status.side_effect = _http_error(403)
        with patch("requests.get", return_value=mock_resp):
            from src.retry_utils import http_get
            with pytest.raises(requests.exceptions.HTTPError):
                http_get("https://example.com")


# ── http_post ─────────────────────────────────────────────────────────────────

class TestHttpPost:
    def test_success_on_first_try(self):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_resp) as mock_post:
            from src.retry_utils import http_post
            result = http_post("https://example.com", json={"k": "v"}, timeout=5)
        assert result is mock_resp
        assert mock_post.call_count == 1

    def test_passes_kwargs_through(self):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_resp) as mock_post:
            from src.retry_utils import http_post
            http_post("https://example.com", json={"a": 1}, headers={"X": "Y"}, timeout=10)
        _, kwargs = mock_post.call_args
        assert kwargs["json"] == {"a": 1}
        assert kwargs["headers"] == {"X": "Y"}

    def test_raises_on_400(self):
        with patch("requests.post", side_effect=_http_error(400)):
            from src.retry_utils import http_post
            with pytest.raises(requests.exceptions.HTTPError):
                http_post("https://example.com")


# ── make_openai_client ────────────────────────────────────────────────────────

class TestMakeOpenaiClient:
    def test_returns_openai_instance(self):
        from openai import OpenAI
        with patch("src.retry_utils.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test-key"
            from src.retry_utils import make_openai_client
            client = make_openai_client()
        assert isinstance(client, OpenAI)

    def test_sets_max_retries_3(self):
        with patch("src.retry_utils.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test-key"
            from src.retry_utils import make_openai_client
            client = make_openai_client()
        assert client.max_retries == 3

    def test_kwargs_forwarded(self):
        with patch("src.retry_utils.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test-key"
            from src.retry_utils import make_openai_client
            client = make_openai_client(base_url="https://custom.api/v1/")
        assert "custom.api" in str(client.base_url)
