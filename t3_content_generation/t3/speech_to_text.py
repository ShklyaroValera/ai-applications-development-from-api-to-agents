import json
from pathlib import Path
from typing import Any

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/docs/guides/speech-to-text

# Task: transcribe 'audio_sample.mp3' via /v1/audio/transcriptions (multipart/form-data)
# with `whisper-1` and `gpt-4o-transcribe` models and compare results.

_CURRENT_DIR = Path(__file__).parent


class OpenAITranscriptionClient:

    def __init__(self, endpoint: str = OPENAI_HOST + "/v1/audio/transcriptions"):
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("API key cannot be null or empty")

        self._api_key = "Bearer " + api_key
        self._endpoint = endpoint

    def call(self, audio_file_path: Path, print_response: bool = True, **kwargs) -> dict[str, Any]:
        # No 'Content-Type' header: requests builds 'multipart/form-data; boundary=...' itself
        headers = {"Authorization": self._api_key}

        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(
                url=self._endpoint,
                headers=headers,
                files={"file": (audio_file_path.name, audio_file, "audio/mpeg")},
                data=kwargs,
            )

        if response.status_code == 200:
            data = response.json()
            if print_response:
                print(json.dumps(data, indent=2))
            return data

        raise Exception(f"HTTP {response.status_code}: {response.text}")


if __name__ == "__main__":
    client = OpenAITranscriptionClient()
    audio_path = _CURRENT_DIR / "audio_sample.mp3"

    for model in ["whisper-1", "gpt-4o-transcribe"]:
        print("=" * 50 + f" {model} " + "=" * 50)
        client.call(audio_file_path=audio_path, model=model)
