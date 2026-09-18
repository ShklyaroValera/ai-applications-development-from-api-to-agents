import base64
from pathlib import Path

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/docs/guides/images-vision?format=url&lang=curl
# https://developers.openai.com/api/docs/guides/images-vision?format=base64-encoded

# Task: analyse these 2 images:
#   - https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg
#   - in this folder we have 'logo.png', load it as encoded data (see documentation)
# In the end load both images (url and base64 encoded 'logo.png'), ask "Generate poem based on images" and see what happens.

_CURRENT_DIR = Path(__file__).parent


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main(model_name: str, img_urls: list[str], request: str = "What's in this image/s?") -> str:
    client = OpenAIClientT3(OPENAI_HOST + "/v1/chat/completions")

    images_content = [
        {"type": "image_url", "image_url": {"url": img_url}}
        for img_url in img_urls
    ]

    data = client.call(
        print_request=False,  # base64 image is huge, don't dump it into the console
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request},
                    *images_content,
                ],
            }
        ],
    )

    content = data["choices"][0]["message"]["content"]
    print("=" * 50 + " AI " + "=" * 50)
    print(content)
    return content


if __name__ == "__main__":
    main(
        model_name="gpt-4o",
        img_urls=[
            "https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg",
            f"data:image/png;base64,{_encode_image(_CURRENT_DIR / 'logo.png')}",
        ],
        request="Generate poem based on images",
    )
