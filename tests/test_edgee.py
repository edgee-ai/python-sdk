"""Tests for Edgee SDK"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from edgee import Edgee, EdgeeConfig


class TestEdgeeConstructor:
    """Test Edgee constructor"""

    def setup_method(self):
        # Clear environment variables before each test
        os.environ.pop("EDGEE_API_KEY", None)
        os.environ.pop("EDGEE_BASE_URL", None)

    def test_with_string_api_key(self):
        """Should use provided API key (backward compatibility)"""
        client = Edgee("test-api-key")
        assert isinstance(client, Edgee)

    def test_with_empty_string_raises_error(self):
        """Should throw error when empty string is provided as API key"""
        with pytest.raises(ValueError, match="EDGEE_API_KEY is not set"):
            Edgee("")

    def test_with_config_dict(self):
        """Should use provided API key and base_url from dict"""
        client = Edgee({"api_key": "test-key", "base_url": "https://custom.example.com"})
        assert isinstance(client, Edgee)

    def test_with_config_object(self):
        """Should use provided API key and base_url from EdgeeConfig"""
        config = EdgeeConfig(api_key="test-key", base_url="https://custom.example.com")
        client = Edgee(config)
        assert isinstance(client, Edgee)

    def test_with_env_api_key(self):
        """Should use EDGEE_API_KEY environment variable"""
        os.environ["EDGEE_API_KEY"] = "env-api-key"
        client = Edgee()
        assert isinstance(client, Edgee)

    def test_with_env_base_url(self):
        """Should use EDGEE_BASE_URL environment variable"""
        os.environ["EDGEE_API_KEY"] = "env-api-key"
        os.environ["EDGEE_BASE_URL"] = "https://env-base-url.example.com"
        client = Edgee()
        assert isinstance(client, Edgee)

    def test_no_api_key_raises_error(self):
        """Should throw error when no API key provided"""
        with pytest.raises(ValueError, match="EDGEE_API_KEY is not set"):
            Edgee()

    def test_empty_config_with_env(self):
        """Should use environment variables when config is empty dict"""
        os.environ["EDGEE_API_KEY"] = "env-api-key"
        client = Edgee({})
        assert isinstance(client, Edgee)


class TestEdgeeSend:
    """Test Edgee.send method"""

    def setup_method(self):
        os.environ.pop("EDGEE_API_KEY", None)
        os.environ.pop("EDGEE_BASE_URL", None)

    def _mock_response(self, data: dict):
        """Create a mock response"""
        mock = MagicMock()
        mock.read.return_value = json.dumps(data).encode("utf-8")
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    @patch("edgee.urlopen")
    def test_send_with_string_input(self, mock_urlopen):
        """Should send request with string input"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Hello")

        assert len(result.choices) == 1
        assert result.choices[0].message["content"] == "Hello, world!"
        assert result.usage.total_tokens == 15

        # Verify the request
        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == "https://api.edgee.ai/v1/chat/completions"
        body = json.loads(call_args.data.decode("utf-8"))
        assert body["model"] == "gpt-4"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    @patch("edgee.urlopen")
    def test_send_with_input_object(self, mock_urlopen):
        """Should send request with InputObject (dict)"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        client.send(
            model="gpt-4",
            input={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

        call_args = mock_urlopen.call_args[0][0]
        body = json.loads(call_args.data.decode("utf-8"))
        assert body["messages"] == [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]

    @patch("edgee.urlopen")
    def test_send_with_tools(self, mock_urlopen):
        """Should include tools when provided"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        client = Edgee("test-api-key")
        result = client.send(
            model="gpt-4",
            input={
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "tools": tools,
                "tool_choice": "auto",
            },
        )

        call_args = mock_urlopen.call_args[0][0]
        body = json.loads(call_args.data.decode("utf-8"))
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"
        assert result.choices[0].message.get("tool_calls") is not None

    @patch("edgee.urlopen")
    def test_send_without_usage(self, mock_urlopen):
        """Should handle response without usage field"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Test")

        assert result.usage is None
        assert len(result.choices) == 1

    @patch("edgee.urlopen")
    def test_send_with_multiple_choices(self, mock_urlopen):
        """Should handle multiple choices in response"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "First response"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Second response"},
                    "finish_reason": "stop",
                },
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Test")

        assert len(result.choices) == 2
        assert result.choices[0].message["content"] == "First response"
        assert result.choices[1].message["content"] == "Second response"

    @patch("edgee.urlopen")
    def test_send_with_custom_base_url(self, mock_urlopen):
        """Should use custom base_url when provided"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        custom_base_url = "https://custom-api.example.com"
        client = Edgee({"api_key": "test-key", "base_url": custom_base_url})
        client.send(model="gpt-4", input="Test")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == f"{custom_base_url}/v1/chat/completions"

    @patch("edgee.urlopen")
    def test_send_with_env_base_url(self, mock_urlopen):
        """Should use EDGEE_BASE_URL environment variable"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        env_base_url = "https://env-base-url.example.com"
        os.environ["EDGEE_BASE_URL"] = env_base_url
        client = Edgee("test-key")
        client.send(model="gpt-4", input="Test")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == f"{env_base_url}/v1/chat/completions"

    @patch("edgee.urlopen")
    def test_config_base_url_overrides_env(self, mock_urlopen):
        """Should prioritize config base_url over environment variable"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        config_base_url = "https://config-base-url.example.com"
        os.environ["EDGEE_BASE_URL"] = "https://env-base-url.example.com"
        client = Edgee({"api_key": "test-key", "base_url": config_base_url})
        client.send(model="gpt-4", input="Test")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == f"{config_base_url}/v1/chat/completions"

    @patch("edgee.urlopen")
    def test_send_with_compression_response(self, mock_urlopen):
        """Should handle response with compression field"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "compression": {
                "input_tokens": 100,
                "saved_tokens": 42,
                "rate": 0.6102003642987249,
            },
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Test")

        assert result.compression is not None
        assert result.compression.input_tokens == 100
        assert result.compression.saved_tokens == 42
        assert result.compression.rate == 0.6102003642987249

    @patch("edgee.urlopen")
    def test_send_without_compression_response(self, mock_urlopen):
        """Should handle response without compression field"""
        mock_response_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_response(mock_response_data)

        client = Edgee("test-api-key")
        result = client.send(model="gpt-4", input="Test")

        assert result.compression is None
