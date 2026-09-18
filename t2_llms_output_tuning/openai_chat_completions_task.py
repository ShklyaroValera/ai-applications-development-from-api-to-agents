from t2_llms_output_tuning._clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from t2_llms_output_tuning._main import run, select_experiment

# 1. n — number of completions to generate per request. Default: 1
#  ⚠️ Note: NOT available in Responses API
#  Query: "Give me a name for a coffee shop"
#  Try: n=3 — returns 3 different completions in choices[]

# 2. temperature — controls randomness. Range: 0.0-2.0, default: 1.0
#  Lower = more deterministic, higher = more creative
#  Query: "Why white is white?"
#  Try: temperature=0.0 vs temperature=2.0, compare outputs
#  ⚠️ Note: it is okay that after temperature=1.5 you get some odd characters in output 😅

# 3. top_p — nucleus sampling, keeps tokens within cumulative probability. Range: 0.0-1.0, default: 1.0
#  Lower = fewer token choices, more focused output
#  Query: "List 5 alternative uses for a paperclip"
#  Try: top_p=0.1 vs top_p=0.9

# 4. max_tokens — max number of tokens in the response. Default: model-dependent
#  ⚠️ Note: Will work for models like gpt-4o. For gpt-5+ - `max_completion_tokens`.
#  Query: "Explain quantum computing"
#  Try: max_tokens=50 vs max_tokens=2048

# 5. stop — list of strings (up to 4) that stop generation when encountered
#  ⚠️ Note: Will work for models like gpt-4o
#  Query: "Count from 1 to 20, comma separated"
#  Try: stop=["5"] — generation stops before reaching 5

# 6. response_format — enforce structured output format
#  "text" (default) or "json_schema" with a schema definition
#  Query: "List 3 programming languages with their year of creation"
#  Try: response_format={"type": "json_schema", "json_schema": {"name": "languages", "strict": True, "schema": {"type": "object", "properties": {"languages": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"}}, "required": ["name", "year"], "additionalProperties": False}}}, "required": ["languages"], "additionalProperties": False}}}

# 7. frequency_penalty — penalizes tokens based on how often they appeared so far. Range: -2.0 to 2.0, default: 0
#  ⚠️ Note: Will work for models like gpt-4o
#  Positive = reduces repetition, negative = encourages repetition
#  Query: "Write a paragraph about the ocean"
#  Try: frequency_penalty=0.0 vs frequency_penalty=1.5

# 8. presence_penalty — penalizes tokens based on whether they appeared at all. Range: -2.0 to 2.0, default: 0
#  ⚠️ Note: Will work for models like gpt-4o
#  Positive = encourages new topics, negative = stays on topic
#  Query: "Write a paragraph about the ocean"
#  Try: presence_penalty=0.0 vs presence_penalty=1.5

# 9. seed — attempts deterministic output. Same seed + same input = same output (best effort)
#  ⚠️ Note: Will work for models like gpt-4o
#  Query: "Give me a name for a coffee shop"
#  Try: seed=42 — run twice with the same seed and compare outputs

# 10. reasoning_effort — controls how much thinking the model does. Values: "low", "medium", "high" (default)
#  Lower effort = faster, cheaper responses; higher = more thorough reasoning
#  ⚠️ Note: does NOT work with non-default temperature (must omit temperature or set to 1.0)
#  Query: "How many r's are in the word strawberry?"
#  Try: reasoning_effort="low" vs reasoning_effort="high"


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
# Select one with env var T2_EXPERIMENT=<name> (default: "n"), override model with T2_MODEL=<model>.
# Sampling/penalty/stop/max_tokens params are rejected by reasoning models (gpt-5.x), so those presets use gpt-4o.
EXPERIMENTS = {
    "none": ("gpt-5.2", {}, "Any question (baseline, no tuning params)"),
    # 1. n
    "n": ("gpt-5.2", {"n": 3}, "Give me a name for a coffee shop"),
    # 2. temperature
    "temperature_0": ("gpt-4o", {"temperature": 0.0}, "Why white is white?"),
    "temperature_2": ("gpt-4o", {"temperature": 2.0}, "Why white is white?"),
    # 3. top_p
    "top_p_0.1": ("gpt-4o", {"top_p": 0.1}, "List 5 alternative uses for a paperclip"),
    "top_p_0.9": ("gpt-4o", {"top_p": 0.9}, "List 5 alternative uses for a paperclip"),
    # 4. max_tokens (gpt-4o) / max_completion_tokens (gpt-5+)
    "max_tokens_50": ("gpt-4o", {"max_tokens": 50}, "Explain quantum computing"),
    "max_tokens_2048": ("gpt-4o", {"max_tokens": 2048}, "Explain quantum computing"),
    "max_completion_tokens_50": ("gpt-5.2", {"max_completion_tokens": 50}, "Explain quantum computing"),
    # 5. stop
    "stop": ("gpt-4o", {"stop": ["5"]}, "Count from 1 to 20, comma separated"),
    # 6. response_format
    "response_format": (
        "gpt-5.2",
        {"response_format": {"type": "json_schema", "json_schema": {"name": "languages", "strict": True, "schema": _LANGUAGES_SCHEMA}}},
        "List 3 programming languages with their year of creation",
    ),
    # 7. frequency_penalty
    "frequency_penalty_0": ("gpt-4o", {"frequency_penalty": 0.0}, "Write a paragraph about the ocean"),
    "frequency_penalty_1.5": ("gpt-4o", {"frequency_penalty": 1.5}, "Write a paragraph about the ocean"),
    # 8. presence_penalty
    "presence_penalty_0": ("gpt-4o", {"presence_penalty": 0.0}, "Write a paragraph about the ocean"),
    "presence_penalty_1.5": ("gpt-4o", {"presence_penalty": 1.5}, "Write a paragraph about the ocean"),
    # 9. seed (run twice and compare)
    "seed": ("gpt-4o", {"seed": 42}, "Give me a name for a coffee shop"),
    # 10. reasoning_effort (no custom temperature!)
    "reasoning_low": ("gpt-5.2", {"reasoning_effort": "low"}, "How many r's are in the word strawberry?"),
    "reasoning_high": ("gpt-5.2", {"reasoning_effort": "high"}, "How many r's are in the word strawberry?"),
}

model_name, params = select_experiment(EXPERIMENTS, default="n")

run(
    client=OpenAIChatCompletionsClient(model_name=model_name),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    **params,
)
