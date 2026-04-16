"""Tests for sf_hallucinate._llm — shared LLM API utility."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from sf_hallucinate._llm import call_chat_completion


class TestCallChatCompletion:
    def test_missing_api_key_raises(self) -> None:
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key"):
                call_chat_completion(
                    [{"role": "user", "content": "hi"}],
                    model="gpt-4o-mini",
                )
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old

    @patch("urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen: MagicMock) -> None:
        response_data = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = call_chat_completion(
            [{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            api_key="test-key",
        )

        assert result == "Hello!"
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_custom_base_url(self, mock_urlopen: MagicMock) -> None:
        response_data = {
            "choices": [{"message": {"content": "OK"}}]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        call_chat_completion(
            [{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            api_key="test-key",
            base_url="https://custom.example.com/v1",
        )

        # Check the URL used
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert "custom.example.com" in request.full_url

    @patch("urllib.request.urlopen")
    def test_env_var_api_key(self, mock_urlopen: MagicMock) -> None:
        response_data = {
            "choices": [{"message": {"content": "OK"}}]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        old = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "env-key"
        try:
            result = call_chat_completion(
                [{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
            )
            assert result == "OK"
        finally:
            if old is None:
                del os.environ["OPENAI_API_KEY"]
            else:
                os.environ["OPENAI_API_KEY"] = old
