# 🧠 Guide to Adding a New Model to `cfg.json`

## ⚠️ Important Warning
Only GGUF models specifically fine-tuned for chat will work. Ensure your model has the "chat" prefix in its configuration or prompt setup to enable proper chat functionality. Non-chat-tuned models may fail or produce unexpected results.

---

## 📁 1. Configuration File Location

The `cfg.json` file is located in the `models/` directory, for example:

```
models/cfg.json
```

This is the central file used to:

* Declare available models.
* Configure paths, format types (`format`).
* Set descriptions and model loading parameters.

---

## ⚙️ 2. Standard Structure of a Model in `cfg.json`

Example:

```json
{
  "models": [
    {
      "name": "Llama-3-8B-chat",
      "path": "models/llama-3-8b-chat.Q4_K_M.gguf",
      "format": "llama-3",
      "description": "LLaMA 3 8B model fine-tuned for conversation.",
      "max_context": 4096,
      "temperature": 0.7,
      "top_p": 0.9
    },
    {
      "name": "MistralLite-7B",
      "path": "models/mistrallite-7b.Q4_K_M.gguf",
      "format": "mistrallite",
      "description": "Compact version of Mistral 7B.",
      "max_context": 4096,
      "temperature": 0.8,
      "top_p": 0.9
    }
  ],
  "valid_formats": [
    "llama-2", "llama-3", "alpaca", "qwen", "vicuna", "oasst_llama",
    "baichuan-2", "baichuan", "openbuddy", "redpajama-incite", "snoozy",
    "phind", "intel", "open-orca", "mistrallite", "zephyr", "pygmalion",
    "chatml", "mistral-instruct", "chatglm3", "openchat", "saiga",
    "gemma", "functionary", "functionary-v2", "functionary-v1",
    "chatml-function-calling"
  ]
}
```

---

## 🧩 3. Adding a New Model

To add a model:

1. Open the `models/cfg.json` file in a text editor (VSCode, nano, etc.).
2. Add a new block to the `"models"` array, following this template:

```json
{
  "name": "Display name of the model",
  "path": "path_to_model_file.gguf",
  "format": "format_from_valid_formats",
  "description": "Brief description of the model",
  "max_context": 4096,
  "temperature": 0.7,
  "top_p": 0.9
}
```

3. Ensure the `format` you enter is in the `"valid_formats"` list.

If the format doesn't exist, **add it to the `valid_formats` list** at the end of the file:

```json
"valid_formats": [
  ...,
  "mymodel-format"
]
```

---

## 🔍 4. Common Errors and Troubleshooting

| Error                                          | Cause                                                              | Fix                                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| `string indices must be integers, not 'str'`   | `cfg.json` has invalid JSON structure (missing `,` or `]`)         | Validate structure with [https://jsonlint.com](https://jsonlint.com)   |
| `KeyError: 'format'`                           | Model missing the `"format"` field                                 | Add a valid `format` field                                             |
| `Invalid format`                               | `format` not in `valid_formats`                                    | Add the format to the list or select a different one                   |

---

## ✅ 5. Complete Example of Adding a New Model

```json
{
  "name": "Gemma-2B-Chat",
  "path": "models/gemma-2b-chat.Q4_K_M.gguf",
  "format": "gemma",
  "description": "Lightweight chat model optimized for speed.",
  "max_context": 4096,
  "temperature": 0.6,
  "top_p": 0.85
}
```

After adding, save the file and restart the program:

```bash
python main.py
```

---

## 💡 Tips:

* Keep `cfg.json` in **UTF-8 without BOM** format to avoid file reading errors.
* If you have many models, use clear descriptions for easy differentiation.
* You can use `// comment` syntax in the file **if your JSON parser supports extended JSON**; otherwise, remove comments.