import json
from datetime import datetime
from pathlib import Path

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


class Voice:
    alloy: str = 'alloy'
    ash: str = 'ash'
    ballad: str = 'ballad'
    coral: str = 'coral'
    echo: str = 'echo'
    fable: str = 'fable'
    nova: str = 'nova'
    onyx: str = 'onyx'
    sage: str = 'sage'
    shimmer: str = 'shimmer'


# https://developers.openai.com/api/docs/guides/text-to-speech
# Request:
# curl https://api.openai.com/v1/audio/speech \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "gpt-4o-mini-tts",
#     "input": "Why can't we say that black is white?",
#     "voice": "coral",
#     "instructions": "Speak in a cheerful and positive tone."
#   }' \
# Response:
#   bytes with audio

# Task: convert text to speech via /v1/audio/speech with `gpt-4o-mini-tts` and save the binary response as .mp3.

_CURRENT_DIR = Path(__file__).parent


class OpenAISpeechClient:

    def __init__(self, endpoint: str = OPENAI_HOST + "/v1/audio/speech"):
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("API key cannot be null or empty")

        self._api_key = "Bearer " + api_key
        self._endpoint = endpoint

    def call(self, print_request: bool = True, **kwargs) -> Path:
        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json"
        }

        if print_request:
            print(json.dumps(kwargs, indent=2))

        response = requests.post(url=self._endpoint, headers=headers, json=kwargs)

        if response.status_code == 200:
            voice = kwargs.get("voice", "voice")
            output_file = _CURRENT_DIR / f"speech_{voice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Audio saved to {output_file}")
            return output_file

        raise Exception(f"HTTP {response.status_code}: {response.text}")


if __name__ == "__main__":
    client = OpenAISpeechClient()
    client.call(
        model="gpt-4o-mini-tts",
        input="Why can't we say that black is white?",
        voice=Voice.coral,  # Experiment with other voices: Voice.alloy, Voice.nova, Voice.onyx, ...
        instructions="Speak in a cheerful and positive tone.",
    )
