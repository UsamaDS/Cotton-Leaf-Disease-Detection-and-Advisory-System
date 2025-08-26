"""
advisory.py
Prompt templates and LLM wrapper to generate short, local advisory.

We provide:
- rule_based_advisory(detections, severity)  -- fallback quick advisory
- llm_advisory(detections, severity, location='Pakistan') -- uses OpenAI

Make sure to set environment variable OPENAI_API_KEY if you want LLM calls.
"""

import os
import openai
from groq import Groq
from typing import List, Dict

# Simple rule templates (fallback)
RULE_TEMPLATES = {
    "healthy": {
        "en": "Leaf is healthy. No action required.",
        "ur": "پتہ صحت مند ہے۔ کسی کارروائی کی ضرورت نہیں۔"
    },
    "curl_stage1": {
        "en": "Mild curl detected. Monitor plant closely and remove early infected leaves.",
        "ur": "ہلکی کرل بیماری کی نشاندہی ہوئی ہے۔ پودے پر نظر رکھیں اور ابتدائی متاثرہ پتوں کو ہٹا دیں۔"
    },
    "curl_stage2": {
        "en": "Advanced curl detected. Consider vector (whitefly) control and consult extension services.",
        "ur": "شدید کرل بیماری کی نشاندہی ہوئی ہے۔ سفید مکھی کے کنٹرول پر غور کریں اور زرعی ماہرین سے مشورہ کریں۔"
    },
    "sooty_mold": {
        "en": "Sooty mold detected. Reduce honeydew-producing insects (aphids, whiteflies). Wash leaves if infestation is low.",
        "ur": "سوٹی پھپھوندی کی نشاندہی ہوئی ہے۔ میٹھا مادہ پیدا کرنے والے کیڑوں (افڈز، سفید مکھی) کو کم کریں۔ اگر انفیکشن کم ہے تو پتوں کو دھو لیں۔"
    },
    "leaf_enation": {
        "en": "Leaf enation detected. Monitor crop and consult agronomist for resistant varieties and vector management.",
        "ur": "لیف انیشن کی نشاندہی ہوئی ہے۔ فصل پر نظر رکھیں اور مزاحم اقسام اور کیڑوں کے کنٹرول کے لئے ماہرین سے مشورہ کریں۔"
    }
    # "curl_stage2": { ** },  # you can replicate or provide specifics per class
}
def rule_based_advisory(detections: List[Dict], severity_result: Dict) -> str:
    """
    Compose short advisory from rule templates.
    """
    if not detections:
        return {"en": "No disease detected.", "ur": "کوئی بیماری نہیں ملی۔"}

    # primary disease = highest-per-box ratio or highest conf
    per_box = severity_result.get("per_box", [])
    if per_box:
        primary = max(per_box, key=lambda x: x.get("ratio", x.get("lesion_area",0)))
        disease_name = primary["name"]
        ratio = primary.get("ratio", primary.get("lesion_area",0)/ (severity_result.get("leaf_area",1) if severity_result.get("leaf_area") else 1))
    else:
        disease_name = detections[0]["name"]
        ratio = severity_result.get("overall_ratio", 0.0)

    # Rule-based fallback
    advices_en = []
    advices_ur = []
    sev_label = severity_result.get("overall_label", "mild")
    # fallback text
    template = RULE_TEMPLATES.get(disease_name, None)
    if template:
        # text = template.get(sev_label, template.get("mild"))
        # return text.format(ratio=ratio)
        advices_en.append(template.get(disease_name, {}).get("en", f"No advisory available for {disease_name}"))
        advices_ur.append(template.get(disease_name, {}).get("ur", f"{disease_name} کے لئے کوئی مشورہ دستیاب نہیں۔"))

        return {"en": "\n".join(advices_en), "ur": "\n".join(advices_ur)}

    # generic fallback
    return f"Detected {disease_name} with severity {sev_label} (~{ratio:.2%}). Immediate: remove badly infected leaves. Prevent: manage vectors, maintain sanitation. Consult local extension for chemical recommendations."

# -------------------------
# LLM wrapper (OpenAI)
# -------------------------
import os
import json
import requests
from typing import List, Dict, Any

# ensure you have openai installed if you plan to use OpenAI
try:
    import openai
except Exception:
    openai = None  # we'll check before using

def _fallback_from_rule(detections: List[Dict], severity_result: Dict) -> Dict[str, str]:
    """
    Produce a bilingual fallback advisory using the rule_based_advisory helper.
    If rule_based_advisory returns a dict (with 'en' and 'ur'), return that.
    If it returns a single string, return it as English and attach a short Urdu note.
    """
    try:
        fb = rule_based_advisory(detections, severity_result)  # assume this function exists
    except Exception:
        fb = "No rule-based advisory available."

    if isinstance(fb, dict):
        # Already bilingual (expected format {'en': ..., 'ur': ...})
        return {"en": fb.get("en", ""), "ur": fb.get("ur", "")}
    else:
        # Single-language fallback -> return English and an Urdu placeholder
        return {"en": fb, "ur": fb + "\n\n(اردو میں مشورہ دستیاب نہیں.)"}


def llm_advisory(
    detections: List[Dict],
    severity_result: Dict,
    location: str = "Pakistan",
    model_name: str = "gpt-4o-mini",
    llmChoice: str = "grok"
) -> Dict[str, str]:
    """
    Generate advisory in BOTH English ('en') and Urdu ('ur') using either OpenAI or Grok.
    - detections: list of dicts containing {name, conf, box...}
    - severity_result: output of compute_severity_from_boxes
    - llmChoice: 'openAI' or 'grok'
    Returns: dict {'en': english_advisory, 'ur': urdu_advisory}
    """

    # Compose shared prompt info
    disease_list = ", ".join(sorted(set([d.get("name", str(d.get("cls", ""))) for d in detections])))
    overall_ratio = severity_result.get("overall_ratio", 0.0)
    overall_label = severity_result.get("overall_label", "mild")
    per_class = severity_result.get("per_box", [])

    # Short English prompt (concise)
    prompt_en = (
        f"You are an agricultural advisor for smallholder cotton farmers in {location}.\n"
        f"Short and clear bullet-point advisory (3–6 lines) for the farmer based on:\n"
        f"- Detected diseases: {disease_list}\n"
        f"- Overall severity: {overall_label} (~{overall_ratio:.2%})\n"
        f"- Per-disease details: {per_class}\n\n"
        "Return:\n"
        "1) Immediate treatment steps (concise)\n"
        "2) Preventive measures for future\n"
        "3) Monitoring instructions (how often to check)\n"
        "4) Safety precautions (mention consult local extension; do NOT give dosages)\n"
        "Use simple language and keep it under 90 words."
    )

    # Short Urdu prompt (concise, Urdu-language)
    # Keep the structure similar but written in Urdu
    prompt_ur = (
        f"آپ چھوٹے کسانوں کے لیے کپاس کے پودوں کے لیے زرعی مشیر ہیں، مقام: {location}۔\n"
        f"مختصر اور واضح نکات (3-6 سطریں) دیں جن کی بنیاد یہ ہیں:\n"
        f"- معلوم شدہ بیماریاں: {disease_list}\n"
        f"- مجموعی شدت: {overall_label} (تقریباً {overall_ratio:.2%})\n"
        f"- فی بیماری تفصیل: {per_class}\n\n"
        "جواب میں دیں:\n"
        "1) فوری علاجی اقدامات (مختصر)\n"
        "2) مستقبل کے لئے احتیاطی اقدامات\n"
        "3) مانیٹرنگ ہدایات (کتنی بار چیک کریں)\n"
        "4) حفاظتی ہدایات (مقامی توسیعی دفتر سے مشورہ کریں؛ خوراک یا ڈوز نہ دیں)\n"
        "سادہ زبان استعمال کریں اور 90 الفاظ سے کم رکھیں۔"
    )

    # Helper fallback
    fallback = _fallback_from_rule(detections, severity_result)

    # -----------------------
    # OpenAI branch
    # -----------------------
    if llmChoice.lower() == "openai":
        if openai is None:
            return fallback  # openai not installed; fallback

        api_key = "sk-proj-kGpdzm3UkeoNu2le-"
        if not api_key:
            return fallback

        try:
            openai.api_key = api_key

            # English
            resp_en = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a concise, practical agricultural extension officer."},
                    {"role": "user", "content": prompt_en},
                ],
                max_tokens=250,
                temperature=0.2,
            )
            text_en = resp_en["choices"][0]["message"]["content"].strip()

            # Urdu
            resp_ur = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "آپ ایک مختصر اور عملی زرعی مشیر ہیں۔"},
                    {"role": "user", "content": prompt_ur},
                ],
                max_tokens=250,
                temperature=0.2,
            )
            text_ur = resp_ur["choices"][0]["message"]["content"].strip()

            return {"en": text_en, "ur": text_ur}

        except Exception as e:
            # If OpenAI call fails, return fallback bilingual advisory
            return fallback

    # -----------------------
    # Grok branch
    # -----------------------
    elif llmChoice.lower() == "grok":
        # Try to use a Python Grok client (if available), otherwise attempt HTTP endpoint via GROQ_API_URL
        groq_api_key = 'gsk_s8hoHKPrf8HZ2'

        if not groq_api_key:
            return fallback

        # First, attempt to use a Groq client if installed (some orgs expose `Groq`)
        try:
            # This will only work if a "groq" or "grok" package with Groq class is installed
            try:
                from groq import Groq  # type: ignore
            except Exception:
                # some environments may provide a Groq class elsewhere; try alternative import
                from groq_client import Groq  # type: ignore

            client = Groq(api_key=groq_api_key) if groq_api_key else Groq()
            # Create messages in the same shape used by other chat APIs
            messages_en = [
                {"role": "system", "content": f"You are an agricultural advisor for smallholder cotton farmers in {location}."},
                {"role": "user", "content": prompt_en},
            ]
            messages_ur = [
                {"role": "system", "content": f"آپ کپاس کے پودوں کے کسانوں کے لیے زرعی مشیر ہیں۔ مقام: {location}۔"},
                {"role": "user", "content": prompt_ur},
            ]

            # Choose a reasonable Grok model name (adapt if your provider uses another name)
            grok_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

            resp_en = client.chat.completions.create(model=grok_model, messages=messages_en)
            text_en = getattr(resp_en.choices[0].message, "content", None) or resp_en.choices[0]["message"]["content"]

            resp_ur = client.chat.completions.create(model=grok_model, messages=messages_ur)
            text_ur = getattr(resp_ur.choices[0].message, "content", None) or resp_ur.choices[0]["message"]["content"]

            # Ensure they are strings
            text_en = text_en.strip() if isinstance(text_en, str) else str(text_en)
            text_ur = text_ur.strip() if isinstance(text_ur, str) else str(text_ur)
            return {"en": text_en, "ur": text_ur}

        except Exception:
            # If the client import/usage fails, fall back to HTTP call (if GROQ_API_URL supplied)
            if groq_api_url:
                try:
                    headers = {"Authorization": f"Bearer {groq_api_key}"} if groq_api_key else {}
                    payload_en = {"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                                  "messages": [{"role": "user", "content": prompt_en}]}
                    payload_ur = {"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                                  "messages": [{"role": "user", "content": prompt_ur}]}

                    # English
                    r_en = requests.post(groq_api_url, headers=headers, json=payload_en, timeout=30)
                    r_en.raise_for_status()
                    j_en = r_en.json()
                    # Accessing choice content depends on the endpoint schema; try common patterns
                    text_en = (
                        j_en.get("choices", [{}])[0].get("message", {}).get("content")
                        or j_en.get("choices", [{}])[0].get("text")
                        or json.dumps(j_en)
                    )

                    # Urdu
                    r_ur = requests.post(groq_api_url, headers=headers, json=payload_ur, timeout=30)
                    r_ur.raise_for_status()
                    j_ur = r_ur.json()
                    text_ur = (
                        j_ur.get("choices", [{}])[0].get("message", {}).get("content")
                        or j_ur.get("choices", [{}])[0].get("text")
                        or json.dumps(j_ur)
                    )

                    return {"en": text_en.strip() if isinstance(text_en, str) else str(text_en),
                            "ur": text_ur.strip() if isinstance(text_ur, str) else str(text_ur)}

                except Exception:
                    return fallback
            else:
                return fallback

    # -----------------------
    # Unknown llmChoice -> fallback
    # -----------------------
    else:
        return fallback

# Update my below llm_advisory function to include both english and urdu advisory generation using the Grok LLM API. Ensure that the function can switch between OpenAI and Grok based on a parameter llmChoice. If llmChoice is 'openAI', use the OpenAI API; if 'grok', use the Grok API..
# def llm_advisory(detections: List[Dict], severity_result: Dict, location: str = "Pakistan", model_name: str = "gpt-4o-mini", llmChoice: str = "grok") -> str:
#     """
#     Call OpenAI chat completion to produce short advisory.
#     - detections: list of dicts containing {name, conf, box...}
#     - severity_result: output of compute_severity_from_boxes
#     Returns: short string advisory
#     """
#     if llmChoice == 'openAI':
#         # Set your OpenAI api key
#         api_key = "sk-proj-kGpdzm3UkeoNu2le-"
#         if not api_key:
#             return "OPENAI_API_KEY not set. Please set environment variable to enable LLM-based advisory."

#         openai.api_key = api_key

#         # Compose small prompt
#         disease_list = ", ".join(sorted(set([d["name"] for d in detections])))
#         overall_ratio = severity_result.get("overall_ratio", 0.0)
#         overall_label = severity_result.get("overall_label", "mild")
#         per_class = severity_result.get("per_box", [])

#         prompt = f"""You are an agricultural advisor for smallholder cotton farmers in {location}. 
#         Short and clear bullet-point advisory (3–6 lines) for the farmer based on:
#         - Detected diseases: {disease_list}
#         - Overall severity: {overall_label} (~{overall_ratio:.2%})
#         - Per-disease details: {per_class}

#         Return:
#         1) Immediate treatment steps (concise)
#         2) Preventive measures for future
#         3) Monitoring instructions (how often to check)
#         4) Safety precautions (mention to consult local extension for pesticides; no dosages).
#         Use simple language, and keep it under 90 words.
#         """

#         # Chat completion
#         try:
#             resp = openai.ChatCompletion.create(
#                 model=model_name,
#                 messages=[
#                     {"role":"system", "content": "You are a helpful, concise agricultural extension officer."},
#                     {"role":"user", "content": prompt}
#                 ],
#                 max_tokens=250,
#                 temperature=0.2
#             )
#             text = resp['choices'][0]['message']['content'].strip()
#             return text
#         except Exception as e:
#             return f"LLM call failed: {e}\nFalling back to rule-based advisory.\n\n" + rule_based_advisory(detections, severity_result)
#     elif llmChoice == 'grok':
#         # Set your Grok API Key
#         # Set API key manually
#         os.environ["GROQ_API_KEY"] = 'gsk_s8hoH' # add your API key here: ######

#         groq_api_key = os.getenv('GROQ_API_KEY')        
#         if not groq_api_key:
#             return "Grok_API_KEY not set. Please set environment variable to enable LLM-based advisory."
#         try:
#             # Create a Groq instance
#             groq = Groq()

#             # Compose small prompt
#             disease_list = ", ".join(sorted(set([d["name"] for d in detections])))
#             overall_ratio = severity_result.get("overall_ratio", 0.0)
#             overall_label = severity_result.get("overall_label", "mild")
#             per_class = severity_result.get("per_box", [])

#             messages = [{"role": "system", "content": f"You are an agricultural advisor for smallholder cotton farmers in {location}."},
#                         {"role": "user", "content": f""" 
#                             Short and clear bullet-point advisory (3–6 lines) for the farmer based on:
#                             - Detected diseases: {disease_list}
#                             - Overall severity: {overall_label} (~{overall_ratio:.2%})
#                             - Per-disease details: {per_class}

#                             Return:
#                             1) Immediate treatment steps (concise)
#                             2) Preventive measures for future
#                             3) Monitoring instructions (how often to check)
#                             4) Safety precautions (mention to consult local extension for pesticides; no dosages).
#                             Use simple language, and keep it under 90 words.
#                         """}
#                     ]

#             response = groq.chat.completions.create(model='llama-3.3-70b-versatile', messages=messages)
#             # Read the agentic solution
#             agentic_solution = response.choices[0].message.content
#             return agentic_solution
#         except Exception as e:
#             return f"LLM call failed: {e}\nFalling back to rule-based advisory.\n\n" + rule_based_advisory(detections, severity_result)
