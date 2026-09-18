import base64
from datetime import datetime
from pathlib import Path

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/reference/resources/images/methods/generate
# ---
# Request:
# curl -X POST "https://api.openai.com/v1/images/generations" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -H "Content-type: application/json" \
#     -d '{
#         "model": "gpt-image-2",
#         "prompt": "smiling catdog."
#     }'
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO...,
#     }
#   ]
# }

# Task: create an image with `gpt-image-2` model ('Smiling catdog'), decode it from base64 and save it locally.

_CURRENT_DIR = Path(__file__).parent


def main(model_name: str, request: str, **kwargs) -> Path:
    client = OpenAIClientT3(endpoint=OPENAI_HOST + "/v1/images/generations")

    # print_response=False: the response contains a (huge) base64 image
    response = client.call(
        print_response=False,
        model=model_name,
        prompt=request,
        **kwargs
    )
    image_base64 = response["data"][0]["b64_json"]
    print(f"Usage: {response.get('usage')}")

    image_bytes = base64.b64decode(image_base64)
    filename = _CURRENT_DIR / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    with open(filename, "wb") as f:
        f.write(image_bytes)

    print(f"Image saved as {filename}")
    return filename


if __name__ == "__main__":
    main(
        model_name="gpt-image-2",
        request="Smiling catdog",
        # Optional params to experiment with, e.g.: size="1024x1024", quality="low"
    )
