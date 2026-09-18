from t2_llms_output_tuning._clients.gemini_client import GeminiAIClient
from t2_llms_output_tuning._main import run, select_experiment

# All parameters below must be passed inside generationConfig={...}

# 1. temperature — controls randomness. Range: 0.0-2.0, default: 1.0
#  Lower = more deterministic, higher = more creative
#  Query: "Give me a name for a coffee shop"
#  Try: "temperature": 0.0 vs "temperature": 2.0, compare outputs

# 2. topP — nucleus sampling, keeps tokens within cumulative probability. Range: 0.0-1.0, default: 0.95
#  Lower = fewer token choices, more focused output
#  Query: "List 5 alternative uses for a paperclip"
#  Try: "topP": 0.1 vs "topP": 0.95

# 3. topK — limits token selection to top K candidates. Default: 40
#  Lower = fewer choices per token, more predictable
#  Query: "Write a one-sentence story about a robot"
#  Try: "topK": 1 vs "topK": 64

# 4. maxOutputTokens — max number of tokens in the response. Required, default: 1024 (set in gemini_client.py)
#  Query: "Explain quantum computing"
#  Try: "maxOutputTokens": 50 vs "maxOutputTokens": 2048

# 5. responseMimeType + responseSchema — enforce structured output format
#  responseMimeType: "text/plain" (default), "application/json", "text/x.enum"
#  responseSchema: JSON Schema defining the expected structure (requires responseMimeType="application/json")
#  Query: "List 3 programming languages with their year of creation"
#  Try: "responseMimeType": "application/json",
#       "responseSchema": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"}}}}

# 6. thinkingConfig — enables extended thinking (chain-of-thought). Requires thinkBudget param
#  Model reasons step-by-step before answering
#  Query: "How many r's are in the word strawberry?"
#  Try: "thinkingConfig": {"thinkMode": "THINKING_MODE_ENABLED", "thinkBudget": 5000}


#  ⚠️ Note: the real Gemini API field names are "thinkingBudget" / "includeThoughts" (Gemini 2.5)
#  and "thinkingLevel" (Gemini 3: "low" | "high"); "thinkMode"/"thinkBudget" are not accepted.

# Experiment presets: name -> (model, generationConfig, suggested query).
# Select one with env var T2_EXPERIMENT=<name> (default: "temperature_0"), override model with T2_MODEL=<model>.
EXPERIMENTS = {
    "none": ("gemini-3-flash-preview", {}, "Any question (baseline, only default maxOutputTokens=1024)"),
    # 1. temperature
    "temperature_0": ("gemini-3-flash-preview", {"temperature": 0.0}, "Give me a name for a coffee shop"),
    "temperature_2": ("gemini-3-flash-preview", {"temperature": 2.0}, "Give me a name for a coffee shop"),
    # 2. topP
    "top_p_0.1": ("gemini-3-flash-preview", {"topP": 0.1}, "List 5 alternative uses for a paperclip"),
    "top_p_0.95": ("gemini-3-flash-preview", {"topP": 0.95}, "List 5 alternative uses for a paperclip"),
    # 3. topK
    "top_k_1": ("gemini-3-flash-preview", {"topK": 1}, "Write a one-sentence story about a robot"),
    "top_k_64": ("gemini-3-flash-preview", {"topK": 64}, "Write a one-sentence story about a robot"),
    # 4. maxOutputTokens
    "max_output_tokens_50": ("gemini-3-flash-preview", {"maxOutputTokens": 50}, "Explain quantum computing"),
    "max_output_tokens_2048": ("gemini-3-flash-preview", {"maxOutputTokens": 2048}, "Explain quantum computing"),
    # 5. responseMimeType + responseSchema
    "structured_output": (
        "gemini-3-flash-preview",
        {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
                },
            },
        },
        "List 3 programming languages with their year of creation",
    ),
    # 6. thinkingConfig
    "thinking": (
        "gemini-3-flash-preview",
        {"maxOutputTokens": 8000, "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}},
        "How many r's are in the word strawberry?",
    ),
    "thinking_budget": (
        "gemini-2.5-flash",
        {"maxOutputTokens": 8000, "thinkingConfig": {"includeThoughts": True, "thinkingBudget": 5000}},
        "How many r's are in the word strawberry?",
    ),
}

model_name, generation_config = select_experiment(EXPERIMENTS, default="temperature_0")

run(
    client=GeminiAIClient(model_name),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    generationConfig=generation_config,
)
