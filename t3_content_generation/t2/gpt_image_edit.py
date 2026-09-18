import base64
from datetime import datetime
from pathlib import Path

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/reference/resources/images/methods/edit
# ---
# Request (multipart/form-data, NOT json):
# curl -X POST "https://api.openai.com/v1/images/edits" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -F "model=gpt-image-1" \
#     -F "image=@logo.png" \
#     -F "prompt=Add magical sparkles and glowing aura around the logo"
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": "Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO..."
#     }
#   ]
# }

# Task: edit an existing local image ('logo.png') with `gpt-image-2` model via /v1/images/edits
# (multipart/form-data, NOT json), decode returned base64 image and save it locally.

_CURRENT_DIR = Path(__file__).parent


def main(model_name: str, image_path: Path, prompt: str, **kwargs) -> Path:
    api_key = OPENAI_API_KEY
    if not api_key:
        raise ValueError("API key cannot be null or empty")

    url = OPENAI_HOST + "/v1/images/edits"
    # No 'Content-Type' header: requests sets 'multipart/form-data; boundary=...' itself when `files` is passed
    headers = {"Authorization": "Bearer " + api_key}

    with open(image_path, "rb") as image_file:
        files = {"image": (image_path.name, image_file, "image/png")}
        data = {"model": model_name, "prompt": prompt, **kwargs}

        print({"url": url, "data": data, "files": list(files.keys())})

        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    payload = response.json()
    image_base64 = payload["data"][0]["b64_json"]
    print(f"Usage: {payload.get('usage')}")

    image_bytes = base64.b64decode(image_base64)
    filename = _CURRENT_DIR / f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    with open(filename, "wb") as f:
        f.write(image_bytes)

    print(f"Edited image saved as {filename}")
    return filename


if __name__ == "__main__":
    main(
        model_name="gpt-image-2",
        image_path=_CURRENT_DIR / "logo.png",
        prompt=(
            "Add some magic to this logo: magical sparkles, glowing stars "
            "and a soft mystical aura around it. Keep the original text and shape clearly readable."
        ),
    )
