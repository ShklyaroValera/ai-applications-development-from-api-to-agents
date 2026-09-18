from t2_llms_output_tuning._clients.openai_responses_client import OpenAIResponsesClient
from t2_llms_output_tuning._main import run, select_experiment

# Responses API differences from Chat Completions:
#  - "messages" -> "input", "system" message -> "instructions" param
#  - "max_tokens" -> "max_output_tokens"
#  - "response_format" -> "text" param with format object
#  - "stop" is not available in Responses API
#  - built-in conversation state via "store" + "previous_response_id"
#  - "truncation" strategy for long contexts

# 1. temperature — controls randomness. Range: 0.0-2.0, default: 1.0
#  Query: "Give me a name for a coffee shop"
#  Try: temperature=0.0 vs temperature=2.0, compare outputs

# 2. top_p — nucleus sampling. Range: 0.0-1.0, default: 1.0
#  Query: "List 5 alternative uses for a paperclip"
#  Try: top_p=0.1 vs top_p=0.9

# 3. max_output_tokens — max tokens in response (was "max_tokens" in Chat Completions)
#  Query: "Explain quantum computing"
#  Try: max_output_tokens=50 vs max_output_tokens=2048

# 4. text — structured output format (replaces "response_format" from Chat Completions)
#  Uses text={"format": {...}} instead of response_format={...}
#  Query: "List 3 programming languages with their year of creation"
#  Try: text={"format": {"type": "json_schema", "name": "languages", "strict": True, "schema": {"type": "object", "properties": {"languages": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"}}, "required": ["name", "year"], "additionalProperties": False}}}, "required": ["languages"], "additionalProperties": False}}}

# 5. truncation — controls how long contexts are handled. Default: "disabled"
#  "auto" = drops older input messages to fit context window
#  Try: truncation="auto"

# 6. metadata — attach key-value pairs to a response for tracking/filtering. Not available in Chat Completions
#  Up to 16 key-value pairs, keys up to 64 chars, values up to 512 chars
#  Try: metadata={"project": "demo", "user": "student-1"}

# 7. reasoning — extended thinking config (replaces "reasoning_effort" from Chat Completions)
#  ⚠️ Note: does NOT work with non-default temperature
#  Query: "How many r's are in the word strawberry?"
#  Try: reasoning={"effort": "high"} vs reasoning={"effort": "low"}


# Bonus (README): instructions — replaces the system message
#  Try: instructions="Answer like a pirate, in one sentence"
# Bonus (README): store — keep the response on OpenAI side (can later be referenced via previous_response_id)
#  Try: store=True

_LANGUAGES_SCHEMA = {
    "type": "object",
    "properties": {
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
                "required": ["name", "year"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["languages"],
    "additionalProperties": False,
}

# Experiment presets: name -> (model, request params, suggested query).
# Select one with env var T2_EXPERIMENT=<name> (default: "temperature_0"), override model with T2_MODEL=<model>.
# Sampling params (temperature/top_p) are rejected by reasoning models, so those presets use gpt-4o.
EXPERIMENTS = {
    "none": ("gpt-5.2", {}, "Any question (baseline, no tuning params)"),
    # 1. temperature
    "temperature_0": ("gpt-4o", {"temperature": 0.0}, "Give me a name for a coffee shop"),
    "temperature_2": ("gpt-4o", {"temperature": 2.0}, "Give me a name for a coffee shop"),
    # 2. top_p
    "top_p_0.1": ("gpt-4o", {"top_p": 0.1}, "List 5 alternative uses for a paperclip"),
    "top_p_0.9": ("gpt-4o", {"top_p": 0.9}, "List 5 alternative uses for a paperclip"),
    # 3. max_output_tokens (minimum is 16)
    "max_output_tokens_50": ("gpt-4o", {"max_output_tokens": 50}, "Explain quantum computing"),
    "max_output_tokens_2048": ("gpt-4o", {"max_output_tokens": 2048}, "Explain quantum computing"),
    # 4. text (structured output)
    "text_format": (
        "gpt-5.2",
        {"text": {"format": {"type": "json_schema", "name": "languages", "strict": True, "schema": _LANGUAGES_SCHEMA}}},
        "List 3 programming languages with their year of creation",
    ),
    # 5. truncation
    "truncation": ("gpt-5.2", {"truncation": "auto"}, "Any question (behaviour differs only on very long contexts)"),
    # 6. metadata
    "metadata": ("gpt-5.2", {"metadata": {"project": "demo", "user": "student-1"}}, "Any question (see 'metadata' in response)"),
    # 7. reasoning (no custom temperature!)
    "reasoning_low": ("gpt-5.2", {"reasoning": {"effort": "low"}}, "How many r's are in the word strawberry?"),
    "reasoning_high": ("gpt-5.2", {"reasoning": {"effort": "high"}}, "How many r's are in the word strawberry?"),
    # Bonus: instructions / store
    "instructions": ("gpt-5.2", {"instructions": "Answer like a pirate, in one sentence"}, "What is the capital of France?"),
    "store": ("gpt-5.2", {"store": True}, "Any question (response is stored, see 'store' in response)"),
}

model_name, params = select_experiment(EXPERIMENTS, default="temperature_0")

run(
    client=OpenAIResponsesClient(model_name),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    **params,
)
