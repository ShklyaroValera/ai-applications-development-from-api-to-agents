from t2_llms_output_tuning._clients.anthropic_client import AnthropicAIClient
from t2_llms_output_tuning._main import run, select_experiment

# 1. temperature — controls randomness. Range: 0.0-1.0, default: 1.0
#  Lower = more deterministic, higher = more creative
#  Query: "Give me a name for a coffee shop"
#  Try: temperature=0.0 vs temperature=1.0, compare outputs

# 2. top_p — nucleus sampling, keeps tokens within cumulative probability. Range: 0.0-1.0, default: 1.0 (disabled)
#  Lower = fewer token choices, more focused output
#  Query: "List 5 alternative uses for a paperclip"
#  Try: top_p=0.1 vs top_p=0.9

# 3. top_k — limits token selection to top K candidates. Default: not set (disabled)
#  Lower = fewer choices per token, more predictable
#  Query: "Write a one-sentence story about a robot"
#  Try: top_k=1 vs top_k=50

# 4. stop_sequences — list of strings that stop generation when encountered
#  Query: "Count from 1 to 20, comma separated"
#  Try: stop_sequences=["10"] — generation stops before reaching 10

# 5. output_config — enforce structured JSON output
#  Query: "List 3 programming languages with their year of creation"
#  Try: output_config={"format":{"type": "json_schema", "schema": {"type": "object", "additionalProperties": False, "properties": {"languages": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},"required": ["name", "year"]}}}}}}

# 6. thinking — enables extended thinking (chain-of-thought). Requires budget_tokens param
#  Model reasons step-by-step before answering. Needs max_tokens > budget_tokens
#  Query: "How many r's are in the word strawberry?"
#  Try: thinking={"type": "enabled", "budget_tokens": 5000}, max_tokens=8000


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
# Note: Anthropic does not allow temperature and top_p together for newer models, so presets set one at a time.
EXPERIMENTS = {
    "none": ("claude-sonnet-4-5", {}, "Any question (baseline, no tuning params)"),
    # 1. temperature
    "temperature_0": ("claude-sonnet-4-5", {"temperature": 0.0}, "Give me a name for a coffee shop"),
    "temperature_1": ("claude-sonnet-4-5", {"temperature": 1.0}, "Give me a name for a coffee shop"),
    # 2. top_p
    "top_p_0.1": ("claude-sonnet-4-5", {"top_p": 0.1}, "List 5 alternative uses for a paperclip"),
    "top_p_0.9": ("claude-sonnet-4-5", {"top_p": 0.9}, "List 5 alternative uses for a paperclip"),
    # 3. top_k
    "top_k_1": ("claude-sonnet-4-5", {"top_k": 1}, "Write a one-sentence story about a robot"),
    "top_k_50": ("claude-sonnet-4-5", {"top_k": 50}, "Write a one-sentence story about a robot"),
    # 4. stop_sequences
    "stop_sequences": ("claude-sonnet-4-5", {"stop_sequences": ["10"]}, "Count from 1 to 20, comma separated"),
    # 5. output_config (structured output)
    "output_config": (
        "claude-sonnet-4-5",
        {"output_config": {"format": {"type": "json_schema", "schema": _LANGUAGES_SCHEMA}}},
        "List 3 programming languages with their year of creation",
    ),
    # 6. thinking (max_tokens must be > budget_tokens)
    "thinking": (
        "claude-sonnet-4-5",
        {"thinking": {"type": "enabled", "budget_tokens": 5000}, "max_tokens": 8000},
        "How many r's are in the word strawberry?",
    ),
}

model_name, params = select_experiment(EXPERIMENTS, default="temperature_0")

run(
    client=AnthropicAIClient(model_name),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    **params,
)
