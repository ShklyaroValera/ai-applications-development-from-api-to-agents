import base64
import json
from datetime import datetime
from pathlib import Path

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/docs/guides/audio#add-audio-to-your-existing-application

# Task: generate an answer in audio format based on the audio message ('question.mp3'):
#   /v1/chat/completions + gpt-4o-audio-preview, modalities=["text", "audio"], audio={"voice": "ballad", "format": "mp3"}.
#   Response audio comes as base64 in choices[0].message.audio.data -> decode and save as .mp3

_CURRENT_DIR = Path(__file__).parent


class OpenAIAudioChatClient:

    def __init__(self, endpoint: str = OPENAI_HOST + "/v1/chat/completions"):
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("API key cannot be null or empty")

        self._api_key = "Bearer " + api_key
        self._endpoint = endpoint

    def call(self, print_response: bool = True, **kwargs) -> Path | None:
        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(url=self._endpoint, headers=headers, json=kwargs)

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices present in the response")

        audio = choices[0].get("message", {}).get("audio") or {}
        audio_data = audio.get("data")

        if print_response:
            # Don't dump the huge base64 audio into the console
            printable = json.loads(json.dumps(data))
            printable_audio = printable["choices"][0].get("message", {}).get("audio")
            if printable_audio and printable_audio.get("data"):
                printable_audio["data"] = f"<base64 audio, {len(audio_data)} chars>"
            print(json.dumps(printable, indent=2))

        if not audio_data:
            print("No audio present in the response")
            return None

        output_file = _CURRENT_DIR / f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        with open(output_file, "wb") as f:
            f.write(base64.b64decode(audio_data))

        print(f"Transcript: {audio.get('transcript')}")
        print(f"Audio saved to {output_file}")
        return output_file


def _encode_audio(audio_file_path: Path) -> str:
    with open(audio_file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


if __name__ == "__main__":
    client = OpenAIAudioChatClient()
    client.call(
        # README names gpt-4o-audio-preview, which is no longer available on the account; gpt-audio is its successor
        model="gpt-audio",
        modalities=["text", "audio"],
        audio={"voice": "ballad", "format": "mp3"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": _encode_audio(_CURRENT_DIR / "question.mp3"),
                            "format": "mp3"
                        }
                    }
                ]
            }
        ]
    )
