import os, json
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

def coerce(value, type_str):
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
            return [str(v) for v in value] if isinstance(value, list) else [str(value)]
        if type_str == "array[integer]":
            return [int(v) for v in value] if isinstance(value, list) else [int(value)]
        # string, date -> keep as string
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
        result[key] = coerce(data.get(key), type_str)

    return result

@app.get("/")
def root():
    return {"status": "ok"}
