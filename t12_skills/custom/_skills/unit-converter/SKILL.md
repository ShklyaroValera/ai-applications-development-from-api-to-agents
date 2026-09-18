---
name: unit-converter
description: >
  Converts values between units of measurement across categories: length (km, miles, feet,
  inches), weight (kg, lbs, oz, stone), temperature (Celsius, Fahrenheit, Kelvin and more),
  volume (liters, gallons, cups, ml), area (m², hectares, acres, ft²), speed (km/h, mph, knots),
  time (milliseconds to years), data storage (bits, bytes to petabytes), pressure, and energy.
  Use when the user asks to convert a measurement, asks "how many X in Y", or wants a value
  expressed in different units.
license: Apache-2.0
metadata:
  author: ai-powered-apps-development-expert
  version: "1.0"
allowed-tools: execute_code
---

# Unit Converter

- Invocation examples and all supported units: [examples.md](examples.md)
- How `execute_code` sessions work: [references/how-code-execution-works.md](references/how-code-execution-works.md)
- Conversion script: [scripts/convert.py](scripts/convert.py) (defines `convert_units(value, from_unit, to_unit)`
  returning `(result, category)` and `fmt(value)` for formatting)

Never do the conversion math yourself. Always run it through `execute_code`.

## Workflow

### Step 1: Load the script (first call only, session_id = "")

Call `execute_code` with:
- `script_path = "/unit-converter/scripts/convert.py"`: the tool reads the script from disk and prepends
  its content to your `code` (`<script content>` + `\n\n` + `code`), so you don't need to copy the script.
- `code`: the conversion call from Step 2.
- `session_id = ""`: an empty string creates a new session.

Save the `session_id` returned in `session_info` and reuse it for every later call in this conversation.

### Step 2: Write the conversion call (the `code` parameter)

Map the user's units to the unit keys from examples.md (for example `°F` -> `fahrenheit`,
`TB` -> `tb`, `km/h` -> `km/h`, `acres` -> `acres`), then pass:

```python
result, category = convert_units(<value>, "<from_unit>", "<to_unit>")
print(f"Category: {category}")
print(f"Input:    {fmt(<value>)} <from_unit>")
print(f"Result:   {fmt(result)} <to_unit>")
```

If the user asks for several target units (e.g. "Celsius and Kelvin"), repeat the block once per
target unit in the same `code`.

### Step 3: Return output

Return the printed output as-is. Don't reformat, round, or paraphrase the numbers.

### Step 4: Reuse the session for follow-up conversions

For the next conversion in the same conversation, skip `script_path`: the functions are already loaded.
Call `execute_code` with only `code` (Step 2) and the saved `session_id`.

### Step 5: Error handling

- Unknown unit or incompatible categories (the script raises `ValueError`): report the error message
  and list the supported units of the relevant category from examples.md.
- The value is not a valid number: ask the user to clarify.
- Session expired or `NameError: convert_units is not defined`: silently restart from Step 1 with
  `session_id = ""`.
