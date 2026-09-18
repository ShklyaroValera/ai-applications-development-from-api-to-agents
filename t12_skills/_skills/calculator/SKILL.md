---
name: calculator
description: >
  Evaluates mathematical expressions precisely with a safe Python script: arithmetic, powers,
  square roots, floor division, modulo, logarithms, rounding, trigonometry, and the constants
  pi and e. Use it whenever the user asks to calculate, compute, evaluate, or solve a numeric
  expression or math problem, instead of doing the math in your head.
---

# Calculator Skill

Always compute results with the script. Never calculate mentally, even for "easy" expressions.

## Quick Start

```bash
python /skills/calculator/scripts/calculate.py "<expression>"
```

Example:

```bash
python /skills/calculator/scripts/calculate.py "(144 * 3) + sqrt(256) - 2^8"
# Expression: (144 * 3) + sqrt(256) - 2^8
# Result: 192
```

## Supported Operations

| Category              | Syntax                                              | Example              |
|-----------------------|-----------------------------------------------------|----------------------|
| Arithmetic            | `+`, `-`, `*`, `/`, unary `-` / `+`                 | `12 / 4 - 1`         |
| Power                 | `^` or `**` (`^` is converted to `**`)              | `2^10`               |
| Square                | `x^2` / `x**2`                                      | `9^2`                |
| Square root           | `sqrt(x)`                                           | `sqrt(256)`          |
| Floor division/modulo | `//`, `%`                                           | `17 // 5`, `17 % 5`  |
| Rounding / absolute   | `round(x)`, `round(x, n)`, `floor(x)`, `ceil(x)`, `abs(x)` | `round(pi, 3)` |
| Logarithms            | `log(x)` (natural), `log(x, base)`, `log10(x)`      | `log10(1000)`        |
| Trigonometry          | `sin(x)`, `cos(x)`, `tan(x)` (x in **radians**)     | `sin(pi / 2)`        |
| Constants             | `pi`, `e`                                           | `2 * pi`             |
| Grouping              | parentheses `( )`                                   | `(2 + 3) * 4`        |

Anything else (other names, attributes, strings, comparisons) is rejected as unsafe.

## Workflow

1. Extract the math expression from the user's request and rewrite it in the supported syntax:
   - words to operators ("squared" -> `^2`, "square root of" -> `sqrt(...)`, "mod" -> `%`);
   - degrees to radians for trig functions: `sin(30 * pi / 180)`.
2. Run the script with the expression wrapped in double quotes.
3. Read the `Result:` line from the output.
4. Reply with the expression and the result. For multi-step problems, run one call per step or
   combine into a single expression, and show the intermediate results.
5. If the script prints `Error:` (invalid syntax, unknown name, division by zero), explain the
   problem to the user in plain words and, if possible, fix the expression and retry once.
