import os, json, re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Req(BaseModel):
    text: str
    schema: Dict[str, str]

LEADING_WORDS = ["the ", "a ", "an "]

def snap_to_source(value: str, source_text: str) -> str:
    """If value isn't found verbatim in source_text, try trimming leading
    articles/words until it matches, or find the best verbatim substring."""
    if not isinstance(value, str) or not value:
        return value

    # Already an exact verbatim match (case-insensitive)
    if value.lower() in source_text.lower():
        # Return the exact casing as it appears in source_text
        idx = source_text.lower().find(value.lower())
        return source_text[idx:idx + len(value)]

    # Try stripping leading articles one word at a time
    candidate = value
    words = candidate.split(" ")
    for cut in range(1, min(3, len(words))):
        trimmed = " ".join(words[cut:])
        if trimmed and trimmed.lower() in source_text.lower():
            idx = source_text.lower().find(trimmed.lower())
            return source_text[idx:idx + len(trimmed)]

    # Try stripping trailing punctuation/words similarly
    trimmed = candidate.rstrip(".,;: ")
    if trimmed.lower() in source_text.lower():
        idx = source_text.lower().find(trimmed.lower())
        return source_text[idx:idx + len(trimmed)]

    # No verbatim match found — return original model output as-is
    return value

def coerce(value, type_str, source_text):
    if value is None:
        return None
    try:
        if type_str == "integer":
            return int(float(value))
        if type_str == "float":
            return float(value)
        if type_str == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "yes", "1")
        if type_str == "array[string]":
            arr = value if isinstance(value, list) else [value]
            return [snap_to_source(str(v), source_text) for v in arr]
        if type_str == "array[integer]":
            arr = value if isinstance(value, list) else [value]
            return [int(v) for v in arr]
        if type_str == "string":
            return snap_to_source(str(value), source_text)
        # date -> keep as string, already ISO from model
        return str(value)
    except Exception:
        return None

@app.post("/dynamic-extract")
def dynamic_extract(req: Req):
    schema_desc = "\n".join(f'- "{k}": {v}' for k, v in req.schema.items())

    system_prompt = f"""You are a precise data extraction tool. Extract fields from the given text according to this schema:

{schema_desc}

Rules:
- Return EXACTLY these keys, no extras, no missing keys.
- Use null for any field that cannot be found in the text.
- For string fields, extract the value EXACTLY as it appears in the text — verbatim, same casing, same wording. Do NOT add articles (a/an/the), do NOT paraphrase, do NOT add extra words, do NOT trim words that are part of the value.
- Copy the minimal exact phrase that answers the field — no surrounding context words unless they are part of the actual value.
- For type "date", return ISO format YYYY-MM-DD.
- For type "float"/"integer", return as JSON numbers, not strings.
- For type "boolean", return true or false.
- For type "array[string]"/"array[integer]", return a JSON array.
- Output ONLY the raw JSON object. No markdown, no code fences, no explanation.
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.text}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = completion.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    result = {}
    for key, type_str in req.schema.items():
        result[key] = coerce(data.get(key), type_str, req.text)

    return result

@app.get("/")
def root():
    return {"status": "ok"}
