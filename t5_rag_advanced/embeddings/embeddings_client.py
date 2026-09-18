import json

import requests


class EmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, endpoint: str, model_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = endpoint
        self._api_key = "Bearer " + api_key
        self._model_name = model_name

    def get_embeddings(
            self, inputs: str | list[str],
            dimensions: int,
            print_response: bool = False
    ) -> dict[int, list[float]]:
        """
        Generate dict of indexed embeddings:
            inputs[0](text) -> [0][embedding]
            inputs[1](text) -> [1][embedding]
            ...

        Args:
            inputs: input text, can be singular string or list of strings
            dimensions: number of dimensions
            print_response: to print response in chat or not
        """
        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }
        request_data = {
            "input": inputs,
            "model": self._model_name,
            "dimensions": dimensions,
        }

        response = requests.post(url=self._endpoint, headers=headers, json=request_data, timeout=60)

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        response_json = response.json()
        if print_response:
            print("\n" + "=" * 50 + " RESPONSE " + "=" * 50)
            print(json.dumps(response_json, indent=2))
            print("=" * 110)

        return {item["index"]: item["embedding"] for item in response_json.get("data", [])}
