import os
import io
import json
import ast
import re
import uuid
import threading
from pathlib import Path
from typing import Optional

import spaces
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gradio import Server
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Gemma4ForConditionalGeneration,
    AutoProcessor,
    AutoModelForImageTextToText,
    TextIteratorStreamer,
)
from qwen_vl_utils import process_vision_info

# --- App Configuration & Initialization ---
app = Server()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

QWEN_VL_2B_MODEL_NAME   = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_VL_4B_MODEL_NAME   = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_4B_UNREDACTED_NAME = "prithivMLmods/Qwen3.5-4B-Unredacted-MAX"
QWEN_4B_MODEL_NAME      = "Qwen/Qwen3.5-4B"
QWEN_2B_MODEL_NAME      = "Qwen/Qwen3.5-2B"
LFM_450_MODEL_NAME      = "LiquidAI/LFM2.5-VL-450M"
GEMMA4_E2B_NAME         = "google/gemma-4-E2B-it"
LFM_16_MODEL_NAME       = "LiquidAI/LFM2.5-VL-1.6B"
QWEN_UNREDACTED_NAME    = "prithivMLmods/Qwen3.5-2B-Unredacted-MAX"
QWEN25_VL_3B_NAME       = "Qwen/Qwen2.5-VL-3B-Instruct"

# ── Qwen3-VL-2B-Instruct ────────────────────────────────
print(f"Loading Qwen3-VL-2B model: {QWEN_VL_2B_MODEL_NAME} on {DEVICE}...")
try:
    qwen_vl_2b_model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_VL_2B_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    qwen_vl_2b_processor = AutoProcessor.from_pretrained(QWEN_VL_2B_MODEL_NAME, trust_remote_code=True)
    print("Qwen3-VL-2B model loaded successfully.")
except Exception as e:
    print(f"Warning: Qwen3-VL-2B model loading failed. Error: {e}")
    qwen_vl_2b_model = None
    qwen_vl_2b_processor = None

# ── Qwen3-VL-4B-Instruct ────────────────────────────────
print(f"Loading Qwen3-VL-4B model: {QWEN_VL_4B_MODEL_NAME} on {DEVICE}...")
try:
    qwen_vl_4b_model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_VL_4B_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    qwen_vl_4b_processor = AutoProcessor.from_pretrained(QWEN_VL_4B_MODEL_NAME, trust_remote_code=True)
    print("Qwen3-VL-4B model loaded successfully.")
except Exception as e:
    print(f"Warning: Qwen3-VL-4B model loading failed. Error: {e}")
    qwen_vl_4b_model = None
    qwen_vl_4b_processor = None

# ── Qwen3.5-4B-Unredacted-MAX ───────────────────────────
print(f"Loading Qwen3.5-4B-Unredacted-MAX: {QWEN_4B_UNREDACTED_NAME} on {DEVICE}...")
try:
    qwen_4b_unredacted_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_4B_UNREDACTED_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()
    qwen_4b_unredacted_processor = AutoProcessor.from_pretrained(QWEN_4B_UNREDACTED_NAME)
    print("Qwen3.5-4B-Unredacted-MAX model loaded successfully.")
except Exception as e:
    print(f"Warning: Qwen3.5-4B-Unredacted-MAX model loading failed. Error: {e}")
    qwen_4b_unredacted_model = None
    qwen_4b_unredacted_processor = None

# ── Qwen3.5-4B ──────────────────────────────────────────
print(f"Loading Qwen3.5-4B model: {QWEN_4B_MODEL_NAME} on {DEVICE}...")
try:
    qwen_4b_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_4B_MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()
    qwen_4b_processor = AutoProcessor.from_pretrained(QWEN_4B_MODEL_NAME)
    print("Qwen3.5-4B model loaded successfully.")
except Exception as e:
    print(f"Warning: Qwen3.5-4B model loading failed. Error: {e}")
    qwen_4b_model = None
    qwen_4b_processor = None

# ── Qwen3.5-2B ──────────────────────────────────────────
print(f"Loading Qwen3.5-2B model: {QWEN_2B_MODEL_NAME} on {DEVICE}...")
try:
    qwen_2b_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_2B_MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()
    qwen_2b_processor = AutoProcessor.from_pretrained(QWEN_2B_MODEL_NAME)
    print("Qwen3.5-2B model loaded successfully.")
except Exception as e:
    print(f"Warning: Qwen3.5-2B model loading failed. Error: {e}")
    qwen_2b_model = None
    qwen_2b_processor = None

# ── LFM2.5-VL-450M ──────────────────────────────────────
print(f"Loading LFM-450M model: {LFM_450_MODEL_NAME} on {DEVICE}...")
try:
    lfm_450_model = AutoModelForImageTextToText.from_pretrained(
        LFM_450_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
    ).eval()
    lfm_450_processor = AutoProcessor.from_pretrained(LFM_450_MODEL_NAME)
    print("LFM-450M model loaded successfully.")
except Exception as e:
    print(f"Warning: LFM-450M model loading failed. Error: {e}")
    lfm_450_model = None
    lfm_450_processor = None

# ── Gemma4-E2B-it ───────────────────────────────────────
print(f"Loading Gemma4-E2B-it: {GEMMA4_E2B_NAME} on {DEVICE}...")
try:
    gemma4_e2b_model = Gemma4ForConditionalGeneration.from_pretrained(
        GEMMA4_E2B_NAME, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    if not torch.cuda.is_available():
        gemma4_e2b_model = gemma4_e2b_model.to(DEVICE)
    gemma4_e2b_processor = AutoProcessor.from_pretrained(GEMMA4_E2B_NAME)
    print("Gemma4-E2B-it model loaded successfully.")
except Exception as e:
    print(f"Warning: Gemma4-E2B-it model loading failed. Error: {e}")
    gemma4_e2b_model = None
    gemma4_e2b_processor = None

# ── LFM2.5-VL-1.6B ──────────────────────────────────────
print(f"Loading LFM-1.6B model: {LFM_16_MODEL_NAME} on {DEVICE}...")
try:
    lfm_16_model = AutoModelForImageTextToText.from_pretrained(
        LFM_16_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
    ).eval()
    lfm_16_processor = AutoProcessor.from_pretrained(LFM_16_MODEL_NAME)
    print("LFM-1.6B model loaded successfully.")
except Exception as e:
    print(f"Warning: LFM-1.6B model loading failed. Error: {e}")
    lfm_16_model = None
    lfm_16_processor = None

# ── Qwen3.5-2B-Unredacted-MAX ───────────────────────────
print(f"Loading Qwen3.5-2B-Unredacted-MAX: {QWEN_UNREDACTED_NAME} on {DEVICE}...")
try:
    qwen_unredacted_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_UNREDACTED_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()
    qwen_unredacted_processor = AutoProcessor.from_pretrained(QWEN_UNREDACTED_NAME)
    print("Qwen3.5-2B-Unredacted-MAX model loaded successfully.")
except Exception as e:
    print(f"Warning: Qwen3.5-2B-Unredacted-MAX model loading failed. Error: {e}")
    qwen_unredacted_model = None
    qwen_unredacted_processor = None

# ── Qwen2.5-VL-3B-Instruct ──────────────────────────────
print(f"Loading Qwen2.5-VL-3B-Instruct: {QWEN25_VL_3B_NAME} on {DEVICE}...")
try:
    qwen25_vl_3b_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN25_VL_3B_NAME, torch_dtype="auto", device_map="auto",
    ).eval()
    qwen25_vl_3b_processor = AutoProcessor.from_pretrained(QWEN25_VL_3B_NAME)
    print("Qwen2.5-VL-3B-Instruct model loaded successfully.")
except Exception as e:
    print(f"Warning: Qwen2.5-VL-3B-Instruct model loading failed. Error: {e}")
    qwen25_vl_3b_model = None
    qwen25_vl_3b_processor = None


# ---------------------------------------------------------------------------
# Utility: safe JSON parser (strips markdown fences, handles ast fallback)
# ---------------------------------------------------------------------------
def safe_parse_json(text: str):
    text = text.strip()
    # strip <think>…</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    # Try to find the first JSON array or object in the text
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Server-side annotation  (mirrors reference annotate_image exactly)
# ---------------------------------------------------------------------------
PALETTE_COLORS = [
    (78, 205, 196),   # teal
    (124, 106, 247),  # purple
    (255, 107, 107),  # red
    (255, 217, 61),   # yellow
    (107, 203, 119),  # green
    (255, 146, 43),   # orange
    (204, 93, 232),   # magenta
    (51, 154, 240),   # blue
]


def _get_font(size: int = 14):
    """Try to load a truetype font, fall back to default."""
    for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def annotate_detections(image: Image.Image, objects: list) -> Image.Image:
    """
    Draw bounding boxes + labels on image.
    objects: list of {label, x_min, y_min, x_max, y_max}  (all coords 0-1 fractions)
    """
    image = image.convert("RGB").copy()
    W, H = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    font_lbl = _get_font(max(12, W // 40))

    for i, obj in enumerate(objects):
        col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
        col_rgba_fill  = col + (46,)   # ~18% opacity fill
        col_rgba_solid = col + (255,)

        x1 = int(obj["x_min"] * W)
        y1 = int(obj["y_min"] * H)
        x2 = int(obj["x_max"] * W)
        y2 = int(obj["y_max"] * H)
        # clamp
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Filled rectangle
        draw.rectangle([x1, y1, x2, y2], fill=col_rgba_fill)
        # Border (draw 2px by drawing twice)
        lw = max(2, W // 200)
        for t in range(lw):
            draw.rectangle([x1+t, y1+t, x2-t, y2-t], outline=col_rgba_solid)

        # Corner accents
        ca = min(18, (x2-x1)//4, (y2-y1)//4)
        cw = max(2, lw + 1)
        for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x2,y2,-1,-1),(x1,y2,1,-1)]:
            draw.line([cx, cy, cx+dx*ca, cy], fill=col_rgba_solid, width=cw)
            draw.line([cx, cy, cx, cy+dy*ca], fill=col_rgba_solid, width=cw)

        # Label pill
        label = obj.get("label", "object")
        try:
            bb = font_lbl.getbbox(label)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        except Exception:
            tw, th = len(label)*7, 12
        pad = 5
        pw, ph = tw + pad*2, th + pad*2
        lx = max(0, min(x1, W - pw))
        ly = max(0, y1 - ph) if y1 - ph >= 0 else y1 + 2
        draw.rounded_rectangle([lx, ly, lx+pw, ly+ph], radius=4, fill=col_rgba_solid)
        draw.text((lx+pad, ly+pad), label, fill=(255,255,255,255), font=font_lbl)

    return image


def annotate_points(image: Image.Image, points: list) -> Image.Image:
    """
    Draw point markers + labels on image.
    points: list of {label, x, y}  (coords 0-1 fractions)
    """
    image = image.convert("RGB").copy()
    W, H = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    font_lbl = _get_font(max(12, W // 40))
    r = max(7, W // 55)

    for i, pt in enumerate(points):
        col = PALETTE_COLORS[i % len(PALETTE_COLORS)]
        col_rgba = col + (255,)
        glow_rgba = col + (40,)
        mid_rgba  = col + (64,)

        cx = int(pt["x"] * W)
        cy = int(pt["y"] * H)
        cx = max(r, min(W-r, cx))
        cy = max(r, min(H-r, cy))

        # Outer glow
        draw.ellipse([cx-r*2, cy-r*2, cx+r*2, cy+r*2], fill=glow_rgba)
        # Mid ring
        draw.ellipse([cx-int(r*1.4), cy-int(r*1.4), cx+int(r*1.4), cy+int(r*1.4)], fill=mid_rgba)
        # Core dot
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col_rgba, outline=(255,255,255,255), width=max(2,r//3))
        # Centre white dot
        cr = max(2, r//3)
        draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], fill=(255,255,255,255))

        # Label
        label = pt.get("label", "")
        if label:
            try:
                bb = font_lbl.getbbox(label)
                tw, th = bb[2]-bb[0], bb[3]-bb[1]
            except Exception:
                tw, th = len(label)*7, 12
            pad = 5
            pw, ph = tw + pad*2, th + pad*2
            lx = min(cx + r + 6, W - pw)
            ly = max(0, cy - ph//2)
            draw.rounded_rectangle([lx, ly, lx+pw, ly+ph], radius=4, fill=col_rgba)
            draw.text((lx+pad, ly+pad), label, fill=(255,255,255,255), font=font_lbl)

    return image


def parse_and_annotate(image: Image.Image, full_text: str, category: str):
    """
    Parse model output and return annotated PIL image + structured result dict.
    Mirrors the reference code logic exactly.
    """
    parsed = safe_parse_json(full_text)
    if parsed is None:
        return image, {"error": "No JSON found in model output", "raw": full_text[:500]}

    if category == "Point":
        result = {"points": []}
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if isinstance(item, dict) and "point_2d" in item:
                coords = item["point_2d"]
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
                    # Reference divides by 1000.0 — Qwen uses 0-1000 scale
                    result["points"].append({
                        "label": item.get("label", ""),
                        "x": x / 1000.0,
                        "y": y / 1000.0,
                    })
        annotated = annotate_points(image.copy(), result["points"])
        return annotated, result

    elif category == "Detect":
        result = {"objects": []}
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if isinstance(item, dict) and "bbox_2d" in item:
                coords = item["bbox_2d"]
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    xmin, ymin, xmax, ymax = [float(v) for v in coords]
                    result["objects"].append({
                        "label": item.get("label", "object"),
                        "x_min": xmin / 1000.0,
                        "y_min": ymin / 1000.0,
                        "x_max": xmax / 1000.0,
                        "y_max": ymax / 1000.0,
                    })
        annotated = annotate_detections(image.copy(), result["objects"])
        return annotated, result

    return image, {}


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Inference Generator (Streaming)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def generate_inference_stream(
    image: Image.Image, category: str, prompt: str, model_id: str = "qwen_vl_2b"
):
    if category == "Query":
        full_prompt = prompt
    elif category == "Caption":
        full_prompt = f"Provide a {prompt} length caption for the image."
    elif category == "Point":
        full_prompt = f"Provide 2d point coordinates for {prompt}. Report in JSON format."
    elif category == "Detect":
        full_prompt = f"Provide bounding box coordinates for {prompt}. Report in JSON format."
    else:
        full_prompt = prompt

    # ── Qwen3-VL-2B ─────────────────────────────────────
    if model_id == "qwen_vl_2b":
        if qwen_vl_2b_model is None or qwen_vl_2b_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] Qwen3-VL-2B model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        text_input = qwen_vl_2b_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_vl_2b_processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(qwen_vl_2b_model.device)
        streamer = TextIteratorStreamer(qwen_vl_2b_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=qwen_vl_2b_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.0, do_sample=True))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── Qwen3-VL-4B ─────────────────────────────────────
    elif model_id == "qwen_vl_4b":
        if qwen_vl_4b_model is None or qwen_vl_4b_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] Qwen3-VL-4B model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        text_input = qwen_vl_4b_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_vl_4b_processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(qwen_vl_4b_model.device)
        streamer = TextIteratorStreamer(qwen_vl_4b_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=qwen_vl_4b_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.0, do_sample=True))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── Qwen3.5-4B-Unredacted-MAX ───────────────────────
    elif model_id == "qwen_4b_unredacted":
        if qwen_4b_unredacted_model is None or qwen_4b_unredacted_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] Qwen3.5-4B-Unredacted-MAX model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        text_input = qwen_4b_unredacted_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_4b_unredacted_processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(qwen_4b_unredacted_model.device)
        streamer = TextIteratorStreamer(qwen_4b_unredacted_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=qwen_4b_unredacted_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── Qwen3.5-4B ──────────────────────────────────────
    elif model_id == "qwen_4b":
        if qwen_4b_model is None or qwen_4b_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] Qwen3.5-4B model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        text_input = qwen_4b_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_4b_processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(qwen_4b_model.device)
        streamer = TextIteratorStreamer(qwen_4b_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=qwen_4b_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── Qwen3.5-2B ──────────────────────────────────────
    elif model_id == "qwen_2b":
        if qwen_2b_model is None or qwen_2b_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] Qwen3.5-2B model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        text_input = qwen_2b_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_2b_processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(qwen_2b_model.device)
        streamer = TextIteratorStreamer(qwen_2b_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=qwen_2b_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── LFM-450M ────────────────────────────────────────
    elif model_id == "lfm_450":
        if lfm_450_model is None or lfm_450_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] LFM-450M model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        conversation = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        inputs = lfm_450_processor.apply_chat_template(
            conversation, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, tokenize=True,
        ).to(lfm_450_model.device)
        streamer = TextIteratorStreamer(lfm_450_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=lfm_450_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── Gemma4-E2B-it ───────────────────────────────────
    elif model_id == "gemma4_e2b":
        if gemma4_e2b_model is None or gemma4_e2b_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] Gemma4-E2B-it model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        text_input = gemma4_e2b_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = gemma4_e2b_processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(gemma4_e2b_model.device)
        streamer = TextIteratorStreamer(gemma4_e2b_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=gemma4_e2b_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.0, do_sample=True))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── LFM-1.6B ────────────────────────────────────────
    elif model_id == "lfm_16":
        if lfm_16_model is None or lfm_16_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] LFM-1.6B model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        conversation = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        inputs = lfm_16_processor.apply_chat_template(
            conversation, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, tokenize=True,
        ).to(lfm_16_model.device)
        streamer = TextIteratorStreamer(lfm_16_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=lfm_16_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── Qwen3.5-2B-Unredacted-MAX ───────────────────────
    elif model_id == "qwen_unredacted":
        if qwen_unredacted_model is None or qwen_unredacted_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] Qwen3.5-2B-Unredacted-MAX model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        text_input = qwen_unredacted_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_unredacted_processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(qwen_unredacted_model.device)
        streamer = TextIteratorStreamer(qwen_unredacted_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=qwen_unredacted_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    # ── Qwen2.5-VL-3B-Instruct ──────────────────────────
    elif model_id == "qwen25_vl_3b":
        if qwen25_vl_3b_model is None or qwen25_vl_3b_processor is None:
            yield f"data: {json.dumps({'chunk': '[Error] Qwen2.5-VL-3B-Instruct model not loaded.'})}\n\n"
            yield "data: [DONE]\n\n"; return
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": full_prompt},
        ]}]
        text_input = qwen25_vl_3b_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen25_vl_3b_processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        ).to(qwen25_vl_3b_model.device)
        streamer = TextIteratorStreamer(qwen25_vl_3b_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=qwen25_vl_3b_model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.0, do_sample=True))
        thread.start()
        for tok in streamer:
            if tok: yield f"data: {json.dumps({'chunk': tok})}\n\n"
        thread.join()

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# New endpoint: /api/annotate  — receives image + model output text + category
# Returns annotated PNG + structured JSON
# ---------------------------------------------------------------------------
@app.post("/api/annotate")
async def annotate_endpoint(
    image:    UploadFile = File(...),
    text:     str        = Form(...),
    category: str        = Form(...),
):
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        annotated_img, result_dict = parse_and_annotate(img, text, category)
        png_bytes = pil_to_png_bytes(annotated_img)
        return JSONResponse({
            "image_b64": __import__("base64").b64encode(png_bytes).decode(),
            "result": result_dict,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Main inference endpoint
# ---------------------------------------------------------------------------
@app.post("/api/run")
async def run_inference(
    image:    UploadFile = File(...),
    category: str        = Form(...),
    prompt:   str        = Form(...),
    model_id: str        = Form("qwen_vl_2b"),
):
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((512, 512))
        return StreamingResponse(
            generate_inference_stream(img, category, prompt, model_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal-Edge-Comparator</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:          #0d0d0f;
            --grid:        #1a1a1f;
            --node-bg:     #13131a;
            --node-header: #1c1c26;
            --node-border: #2a2a3a;
            --accent:      #7c6af7;
            --accent2:     #4ecdc4;
            --accent3:     #ff6b6b;
            --text:        #e8e8f0;
            --muted:       #6b6b8a;
            --port:        #4ecdc4;
            --wire:        #2a2a4a;
            --wire-active: #7c6af7;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            min-height: 100%; background: var(--bg);
            color: var(--text); font-family: 'JetBrains Mono', monospace;
        }
        body {
            background-image:
                radial-gradient(circle at 20% 50%, rgba(124,106,247,0.04) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(78,205,196,0.04) 0%, transparent 50%),
                linear-gradient(var(--grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--grid) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 24px 24px, 24px 24px;
            overflow-x: auto; overflow-y: auto;
        }
        /* ── Top Bar ── */
        .top-bar {
            position: sticky; top: 0; left: 0; right: 0; height: 42px;
            background: rgba(13,13,15,0.95); border-bottom: 1px solid var(--node-border);
            display: flex; align-items: center; padding: 0 20px;
            gap: 12px; z-index: 1000; backdrop-filter: blur(12px);
        }
        .top-bar .logo  { font-size: 13px; font-weight: 700; color: var(--accent); letter-spacing: 0.05em; }
        .top-bar .sep   { color: var(--node-border); }
        .top-bar .sub   { font-size: 11px; color: var(--muted); }
        .top-bar .badge {
            margin-left: auto; background: rgba(124,106,247,0.15);
            border: 1px solid rgba(124,106,247,0.3); padding: 3px 10px;
            border-radius: 20px; font-size: 10px; color: var(--accent);
        }
        /* ── Canvas ── */
        #canvas {
            position: relative; width: 1360px;
            min-height: calc(100vh - 42px); height: 900px; margin: 0 auto;
        }
        svg.wires {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none; z-index: 2; overflow: visible;
        }
        path.wire { fill: none; stroke: var(--wire); stroke-width: 2.5; stroke-linecap: round; }
        path.wire.active {
            stroke: var(--wire-active); stroke-width: 3;
            stroke-dasharray: 8 4; animation: flow 0.6s linear infinite;
        }
        @keyframes flow { to { stroke-dashoffset: -24; } }
        /* ── Nodes ── */
        .node {
            position: absolute; width: 295px;
            background: var(--node-bg); border: 1px solid var(--node-border);
            border-radius: 9px; box-shadow: 0 8px 28px rgba(0,0,0,0.5);
            z-index: 10; display: flex; flex-direction: column; transition: box-shadow 0.2s;
        }
        .node:hover { box-shadow: 0 8px 28px rgba(0,0,0,0.5), 0 0 0 1px rgba(124,106,247,0.3); }
        .node.fixed-height { height: 340px; }
        .node-header {
            background: var(--node-header); padding: 7px 12px;
            border-bottom: 1px solid var(--node-border); border-radius: 9px 9px 0 0;
            font-size: 11px; font-weight: 700; cursor: grab;
            display: flex; justify-content: space-between; align-items: center;
            flex-shrink: 0; user-select: none;
        }
        .node-header:active { cursor: grabbing; }
        .node-header .id {
            font-size: 10px; color: var(--muted);
            background: rgba(255,255,255,0.04); padding: 2px 7px; border-radius: 4px;
        }
        .node-body { padding: 10px; display: flex; flex-direction: column; gap: 8px; flex: 1; overflow: hidden; }
        /* ── Ports ── */
        .port {
            position: absolute; width: 11px; height: 11px;
            background: var(--node-bg); border: 2px solid var(--port);
            border-radius: 50%; z-index: 30;
        }
        .port.out { right: -6px; }
        .port.in  { left:  -6px; }
        /* ── Labels ── */
        label {
            font-size: 10px; color: var(--muted); font-weight: 600;
            display: block; margin-bottom: 3px; letter-spacing: 0.07em; text-transform: uppercase;
        }
        input[type="file"] { display: none; }
        /* ── Upload Zone ── */
        .file-upload {
            border: 1.5px dashed var(--node-border); border-radius: 7px; padding: 12px 10px;
            text-align: center; cursor: pointer; font-size: 11px; color: var(--muted);
            transition: border-color 0.2s, background 0.2s; background: rgba(255,255,255,0.01);
            display: flex; flex-direction: column; align-items: center; gap: 5px;
        }
        .file-upload:hover { border-color: var(--accent); background: rgba(124,106,247,0.04); }
        .file-upload svg { opacity: 0.5; transition: opacity 0.2s; }
        .file-upload:hover svg { opacity: 0.9; }
        /* ── Preview wrapper ── */
        .preview-wrap {
            display: none; position: relative; border-radius: 7px;
            overflow: hidden; border: 1px solid var(--node-border); background: #000;
        }
        .preview-wrap.visible { display: block; }
        .img-preview { width: 100%; height: 170px; object-fit: contain; display: block; }
        /* ── Clear button ── */
        .clear-btn {
            position: absolute; top: 6px; right: 6px; width: 24px; height: 24px;
            border-radius: 50%; background: rgba(13,13,15,0.80);
            border: 1px solid var(--node-border); color: var(--accent3); cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            transition: background 0.18s, border-color 0.18s, transform 0.12s;
            z-index: 20; backdrop-filter: blur(6px);
        }
        .clear-btn:hover { background: rgba(255,107,107,0.18); border-color: var(--accent3); transform: scale(1.08); }
        .clear-btn:active { transform: scale(0.95); }
        .clear-btn svg { pointer-events: none; }
        /* ── Filename chip ── */
        .img-chip {
            display: none; align-items: center; gap: 6px;
            background: rgba(124,106,247,0.08); border: 1px solid rgba(124,106,247,0.22);
            border-radius: 5px; padding: 4px 8px; font-size: 9px; color: var(--muted); overflow: hidden;
        }
        .img-chip.visible { display: flex; }
        .img-chip .chip-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent2); flex-shrink: 0; box-shadow: 0 0 4px var(--accent2); }
        .img-chip .chip-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; color: var(--text); font-size: 9px; }
        .img-chip .chip-size { color: var(--muted); flex-shrink: 0; font-size: 9px; }
        select, textarea {
            width: 100%; background: rgba(0,0,0,0.3); border: 1px solid var(--node-border);
            color: var(--text); padding: 7px 9px; border-radius: 5px; outline: none;
            font-size: 11px; font-family: 'JetBrains Mono', monospace;
            resize: none; transition: border-color 0.2s;
        }
        select:focus, textarea:focus { border-color: var(--accent); }
        select option { background: #1c1c26; }
        button.run-btn {
            background: linear-gradient(135deg, var(--accent), #9b59b6);
            color: #fff; border: none; padding: 8px; border-radius: 6px;
            font-weight: 700; font-size: 11px; font-family: 'JetBrains Mono', monospace;
            cursor: pointer; transition: opacity 0.2s, transform 0.1s;
            display: flex; justify-content: center; align-items: center; gap: 8px;
            letter-spacing: 0.04em; flex-shrink: 0;
        }
        button.run-btn:hover   { opacity: 0.9; }
        button.run-btn:active  { transform: scale(0.98); }
        button.run-btn:disabled { background: var(--node-border); cursor: not-allowed; color: #555; }
        /* ── Output node ── */
        .output-node-body { padding: 10px; display: flex; flex-direction: column; gap: 6px; flex: 1; overflow: hidden; }
        .output-header-row { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
        /* ── Icon buttons ── */
        .icon-btn {
            display: flex; align-items: center; gap: 5px;
            background: rgba(124,106,247,0.10); border: 1px solid rgba(124,106,247,0.25);
            border-radius: 5px; padding: 3px 8px; font-size: 9px; font-weight: 700;
            font-family: 'JetBrains Mono', monospace; color: var(--accent); cursor: pointer;
            letter-spacing: 0.05em; transition: background 0.18s, border-color 0.18s, transform 0.1s;
            flex-shrink: 0; text-decoration: none;
        }
        .icon-btn:hover { background: rgba(124,106,247,0.22); border-color: var(--accent); }
        .icon-btn:active { transform: scale(0.95); }
        .icon-btn.teal { background: rgba(78,205,196,0.10); border-color: rgba(78,205,196,0.25); color: var(--accent2); }
        .icon-btn.teal:hover { background: rgba(78,205,196,0.22); border-color: var(--accent2); }
        .icon-btn.copied { background: rgba(78,205,196,0.15); border-color: var(--accent2); color: var(--accent2); }
        .icon-btn svg { pointer-events: none; flex-shrink: 0; }
        .output-box {
            background: rgba(0,0,0,0.4); border: 1px solid var(--node-border);
            border-radius: 5px; padding: 10px; flex: 1; overflow-y: auto;
            font-size: 11px; line-height: 1.6; color: #c8c8e0; white-space: pre-wrap;
            user-select: text; font-family: 'JetBrains Mono', monospace; min-height: 0;
        }
        /* ── Grounding node ── */
        .ground-node-body { padding: 10px; display: flex; flex-direction: column; gap: 6px; flex: 1; overflow: hidden; }
        .ground-header-row { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
        .ground-img-wrap {
            position: relative; flex: 1; border: 1px solid var(--node-border);
            border-radius: 5px; overflow: hidden; background: #111; min-height: 0;
            display: flex; align-items: center; justify-content: center;
        }
        /* annotated image displayed via <img> tag — no canvas needed */
        .ground-img-wrap img.overlay-img {
            max-width: 100%; max-height: 100%;
            object-fit: contain; display: block;
        }
        .ground-placeholder {
            position: absolute; inset: 0; display: flex; align-items: center;
            justify-content: center; font-size: 11px; color: var(--muted);
            text-align: center; padding: 10px; pointer-events: none; z-index: 5;
        }
        .loader {
            width: 11px; height: 11px; border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #fff; border-radius: 50%;
            animation: spin 0.7s linear infinite; display: none;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); display: inline-block; margin-right: 6px; }
        .status-dot.active { background: var(--accent2); box-shadow: 0 0 5px var(--accent2); }
        /* ── Model badges ── */
        .model-badge {
            display: inline-block; padding: 2px 7px; border-radius: 4px;
            font-size: 9px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
        }
        .model-badge.qvl2b    { background: rgba(255,150,50,0.15);  color: #ff9632; border: 1px solid rgba(255,150,50,0.35); }
        .model-badge.qvl4b    { background: rgba(255,100,80,0.15);  color: #ff6450; border: 1px solid rgba(255,100,80,0.35); }
        .model-badge.q4bunred { background: rgba(255,80,80,0.18);   color: #ff5050; border: 1px solid rgba(255,80,80,0.40); }
        .model-badge.q4b      { background: rgba(255,200,80,0.15);  color: #ffc850; border: 1px solid rgba(255,200,80,0.35); }
        .model-badge.q2b      { background: rgba(124,106,247,0.2);  color: var(--accent); border: 1px solid rgba(124,106,247,0.3); }
        .model-badge.lfm450   { background: rgba(78,205,196,0.15);  color: var(--accent2); border: 1px solid rgba(78,205,196,0.3); }
        .model-badge.g4e2b    { background: rgba(66,197,107,0.15);  color: #42c56b; border: 1px solid rgba(66,197,107,0.35); }
        .model-badge.lfm16    { background: rgba(107,203,119,0.15); color: #6bcb77; border: 1px solid rgba(107,203,119,0.35); }
        .model-badge.qunred   { background: rgba(255,80,160,0.15);  color: #ff50a0; border: 1px solid rgba(255,80,160,0.35); }
        .model-badge.q25vl3b  { background: rgba(80,180,255,0.15);  color: #50b4ff; border: 1px solid rgba(80,180,255,0.35); }
        .model-info-box { border-radius: 6px; padding: 9px; font-size: 10px; color: var(--muted); line-height: 1.55; flex-shrink: 0; }
        .canvas-footer { height: 36px; }
    </style>
</head>
<body>

<div class="top-bar">
    <span class="logo">MULTIMODAL EDGE</span>
    <span class="sep">|</span>
    <span class="sub">Node-Based Inference Canvas</span>
    <span class="badge">10x Vision Models</span>
</div>

<div id="canvas">
    <svg class="wires">
        <path id="wire-img-task"   class="wire" />
        <path id="wire-model-task" class="wire" />
        <path id="wire-task-out"   class="wire" />
        <path id="wire-task-gnd"   class="wire" />
    </svg>

    <!-- ─── ID 01 : Image Input ─── -->
    <div class="node fixed-height" id="node-img" style="left:40px; top:52px;">
        <div class="node-header">
            <span><span class="status-dot" id="dot-img"></span>Input Image</span>
            <span class="id">ID: 01</span>
        </div>
        <div class="node-body">
            <div>
                <label>Upload Image</label>
                <div class="file-upload" id="dropZone">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none"
                         stroke="#7c6af7" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                    </svg>
                    <span>Click or drop image here</span>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
                <div class="preview-wrap" id="previewWrap">
                    <img id="imgPreview" class="img-preview" />
                    <button class="clear-btn" id="clearBtn" title="Remove image">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                             stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="img-chip" id="imgChip" style="margin-top:6px;">
                    <span class="chip-dot"></span>
                    <span class="chip-name" id="chipName">—</span>
                    <span class="chip-size" id="chipSize"></span>
                </div>
            </div>
        </div>
        <div class="port out" id="port-img-out" style="top:50%;transform:translateY(-50%);"></div>
    </div>

    <!-- ─── ID 02 : Model Selector ─── -->
    <div class="node fixed-height" id="node-model" style="left:40px; top:412px;">
        <div class="node-header">
            <span><span class="status-dot" id="dot-model"></span>Model Selector</span>
            <span class="id">ID: 02</span>
        </div>
        <div class="node-body">
            <div>
                <label>Active Model</label>
                <select id="modelSelect">
                    <option value="qwen_vl_2b">Qwen3-VL-2B-Instruct</option>
                    <option value="qwen_vl_4b">Qwen3-VL-4B-Instruct</option>
                    <option value="qwen_4b_unredacted">Qwen3.5-4B-Unredacted-MAX</option>
                    <option value="qwen_4b">Qwen3.5-4B</option>
                    <option value="qwen_2b">Qwen3.5-2B</option>
                    <option value="lfm_450">LFM2.5-VL-450M (LiquidAI)</option>
                    <option value="gemma4_e2b">Gemma4-E2B-it (Google)</option>
                    <option value="lfm_16">LFM2.5-VL-1.6B (LiquidAI)</option>
                    <option value="qwen_unredacted">Qwen3.5-2B-Unredacted-MAX</option>
                    <option value="qwen25_vl_3b">Qwen2.5-VL-3B-Instruct</option>
                </select>
            </div>
            <div id="modelInfoBox" class="model-info-box"
                 style="background:rgba(255,150,50,0.07);border:1px solid rgba(255,150,50,0.3);">
                <span class="model-badge qvl2b">QWEN3-VL · 2B</span><br><br>
                Qwen3-VL-2B-Instruct — dedicated vision-language model by Alibaba Cloud.
                Strong spatial grounding, OCR &amp; instruction-following.
            </div>
            <div style="flex:1;"></div>
        </div>
        <div class="port out" id="port-model-out" style="top:50%;transform:translateY(-50%);"></div>
    </div>

    <!-- ─── ID 03 : Task Config ─── -->
    <div class="node fixed-height" id="node-task" style="left:425px; top:52px;">
        <div class="port in" id="port-task-in" style="top:50%;transform:translateY(-50%);"></div>
        <div class="node-header">
            <span><span class="status-dot" id="dot-task"></span>Task Config</span>
            <span class="id">ID: 03</span>
        </div>
        <div class="node-body">
            <div>
                <label>Task Category</label>
                <select id="categorySelect">
                    <option value="Query">Query</option>
                    <option value="Caption">Caption</option>
                    <option value="Point">Point</option>
                    <option value="Detect">Detect</option>
                </select>
            </div>
            <div>
                <label>Prompt Directive</label>
                <textarea id="promptInput" rows="4"
                    placeholder="e.g., Count the total number of boats and describe the environment."></textarea>
            </div>
            <button class="run-btn" id="runBtn">
                <span>Execute</span>
                <span class="loader" id="btnLoader"></span>
            </button>
        </div>
        <div class="port out" id="port-task-out" style="top:50%;transform:translateY(-50%);"></div>
    </div>

    <!-- ─── ID 04 : Output Stream ─── -->
    <div class="node fixed-height" id="node-out" style="left:810px; top:52px;">
        <div class="port in" id="port-out-in" style="top:50%;transform:translateY(-50%);"></div>
        <div class="node-header">
            <span><span class="status-dot" id="dot-out"></span>Output Stream</span>
            <span class="id">ID: 04</span>
        </div>
        <div class="output-node-body">
            <div class="output-header-row">
                <label style="margin-bottom:0;">Streamed Result</label>
                <button class="icon-btn" id="copyBtn" title="Copy result to clipboard">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    COPY
                </button>
            </div>
            <div class="output-box" id="outputBox">Results will stream here...</div>
        </div>
    </div>

    <!-- ─── ID 05 : Grounding Visualiser ─── -->
    <div class="node fixed-height" id="node-gnd" style="left:810px; top:412px;">
        <div class="port in" id="port-gnd-in" style="top:50%;transform:translateY(-50%);"></div>
        <div class="node-header">
            <span><span class="status-dot" id="dot-gnd"></span>View Grounding</span>
            <span class="id">ID: 05</span>
        </div>
        <div class="ground-node-body">
            <div class="ground-header-row">
                <label style="margin-bottom:0;">Point / Detect Overlay</label>
                <a class="icon-btn teal" id="downloadBtn" title="Download overlay image" style="display:none;">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    SAVE
                </a>
            </div>
            <div class="ground-img-wrap" id="groundWrap">
                <img class="overlay-img" id="overlayImg" src="" style="display:none;" />
                <div class="ground-placeholder" id="groundPlaceholder">
                    Active for Point / Detect tasks.<br>Run inference to visualise.
                </div>
            </div>
        </div>
    </div>

    <div class="canvas-footer"></div>
</div>

<script>
// ══════════════════════════════════════════════
//  WIRE DRAWING
// ══════════════════════════════════════════════
const canvasEl = document.getElementById('canvas');
function portCenter(id) {
    const el = document.getElementById(id);
    if (!el) return {x:0,y:0};
    const er = el.getBoundingClientRect(), cr = canvasEl.getBoundingClientRect();
    return { x: er.left + er.width/2 - cr.left, y: er.top + er.height/2 - cr.top };
}
function bezier(p1, p2) {
    const dx = Math.abs(p2.x - p1.x) * 0.55;
    return `M ${p1.x} ${p1.y} C ${p1.x+dx} ${p1.y}, ${p2.x-dx} ${p2.y}, ${p2.x} ${p2.y}`;
}
function updateWires() {
    const wires = [
        ['wire-img-task',   'port-img-out',  'port-task-in'],
        ['wire-model-task', 'port-model-out','port-task-in'],
        ['wire-task-out',   'port-task-out', 'port-out-in'],
        ['wire-task-gnd',   'port-task-out', 'port-gnd-in'],
    ];
    for (const [id, from, to] of wires) {
        const el = document.getElementById(id);
        if (el) el.setAttribute('d', bezier(portCenter(from), portCenter(to)));
    }
}

// ══════════════════════════════════════════════
//  DRAGGING
// ══════════════════════════════════════════════
document.querySelectorAll('.node').forEach(node => {
    const header = node.querySelector('.node-header');
    let drag = false, sx, sy, il, it;
    header.addEventListener('mousedown', e => {
        drag=true; sx=e.clientX; sy=e.clientY;
        il=parseInt(node.style.left)||0; it=parseInt(node.style.top)||0;
        node.style.zIndex=100; e.preventDefault();
    });
    document.addEventListener('mousemove', e => {
        if (!drag) return;
        node.style.left=`${il+e.clientX-sx}px`; node.style.top=`${it+e.clientY-sy}px`;
        updateWires();
    });
    document.addEventListener('mouseup', () => { if(drag){drag=false;node.style.zIndex=10;} });
});
window.addEventListener('resize', updateWires);
window.addEventListener('scroll', updateWires);
document.addEventListener('scroll', updateWires, true);
requestAnimationFrame(updateWires);

// ══════════════════════════════════════════════
//  FILE UPLOAD + CLEAR
// ══════════════════════════════════════════════
let currentFile = null;
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const previewWrap = document.getElementById('previewWrap');
const imgPreview  = document.getElementById('imgPreview');
const clearBtn    = document.getElementById('clearBtn');
const imgChip     = document.getElementById('imgChip');
const chipName    = document.getElementById('chipName');
const chipSize    = document.getElementById('chipSize');
const dotImg      = document.getElementById('dot-img');

function formatBytes(b) {
    if (b<1024) return b+' B';
    if (b<1048576) return (b/1024).toFixed(1)+' KB';
    return (b/1048576).toFixed(1)+' MB';
}
function handleFile(file) {
    if (!file||!file.type.startsWith('image/')) return;
    currentFile=file;
    imgPreview.src=URL.createObjectURL(file);
    previewWrap.classList.add('visible');
    dropZone.style.display='none';
    chipName.textContent=file.name;
    chipSize.textContent=formatBytes(file.size);
    imgChip.classList.add('visible');
    dotImg.classList.add('active');
    requestAnimationFrame(updateWires);
}
function clearImage() {
    currentFile=null; imgPreview.src='';
    previewWrap.classList.remove('visible');
    dropZone.style.display='';
    imgChip.classList.remove('visible');
    chipName.textContent='—'; chipSize.textContent='';
    fileInput.value=''; dotImg.classList.remove('active');
    requestAnimationFrame(updateWires);
}
dropZone.onclick     = () => fileInput.click();
fileInput.onchange   = e  => handleFile(e.target.files[0]);
clearBtn.onclick     = e  => { e.stopPropagation(); clearImage(); };
dropZone.ondragover  = e  => { e.preventDefault(); dropZone.style.borderColor='var(--accent)'; };
dropZone.ondragleave = ()  => { dropZone.style.borderColor=''; };
dropZone.ondrop      = e  => {
    e.preventDefault(); dropZone.style.borderColor='';
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
};

// ══════════════════════════════════════════════
//  MODEL SELECTOR
// ══════════════════════════════════════════════
const modelSelect  = document.getElementById('modelSelect');
const modelInfoBox = document.getElementById('modelInfoBox');
const dotModel     = document.getElementById('dot-model');
dotModel.classList.add('active');

const MODEL_INFO = {
    qwen_vl_2b: {
        html: `<span class="model-badge qvl2b">QWEN3-VL · 2B</span><br><br>
               Qwen3-VL-2B-Instruct — dedicated vision-language model by Alibaba Cloud.
               Strong spatial grounding, OCR &amp; instruction-following.`,
        bg: 'rgba(255,150,50,0.07)', border: 'rgba(255,150,50,0.30)',
    },
    qwen_vl_4b: {
        html: `<span class="model-badge qvl4b">QWEN3-VL · 4B</span><br><br>
               Qwen3-VL-4B-Instruct — enhanced vision-language model by Alibaba Cloud.
               Superior spatial grounding, richer OCR &amp; stronger multi-step reasoning.`,
        bg: 'rgba(255,100,80,0.07)', border: 'rgba(255,100,80,0.25)',
    },
    qwen_4b_unredacted: {
        html: `<span class="model-badge q4bunred">QWEN 3.5 · 4B UNREDACTED MAX</span><br><br>
               Qwen3.5-4B-Unredacted-MAX by prithivMLmods. Uncensored fine-tune of Qwen3.5-4B
               with extended instruction-following &amp; unrestricted reasoning.`,
        bg: 'rgba(255,80,80,0.07)', border: 'rgba(255,80,80,0.30)',
    },
    qwen_4b: {
        html: `<span class="model-badge q4b">QWEN 3.5 · 4B</span><br><br>
               Qwen3.5 4B multimodal model by Alibaba Cloud.
               Enhanced capacity — richer reasoning &amp; better instruction following.`,
        bg: 'rgba(255,200,80,0.07)', border: 'rgba(255,200,80,0.30)',
    },
    qwen_2b: {
        html: `<span class="model-badge q2b">QWEN 3.5 · 2B</span><br><br>
               Qwen3.5 2B multimodal model by Alibaba Cloud.
               Lightweight &amp; fast — ideal for quick Query, Caption, Point &amp; Detect tasks.`,
        bg: 'rgba(124,106,247,0.07)', border: 'rgba(124,106,247,0.25)',
    },
    lfm_450: {
        html: `<span class="model-badge lfm450">LFM · 450M</span><br><br>
               LFM2.5-VL 450M by LiquidAI. Ultra-lightweight edge model
               with solid grounding capabilities.`,
        bg: 'rgba(78,205,196,0.07)', border: 'rgba(78,205,196,0.25)',
    },
    gemma4_e2b: {
        html: `<span class="model-badge g4e2b">GEMMA 4 · E2B</span><br><br>
               Gemma4-E2B-it by Google DeepMind. Efficient 2B multimodal model
               with strong vision-language understanding &amp; instruction-following.`,
        bg: 'rgba(66,197,107,0.07)', border: 'rgba(66,197,107,0.25)',
    },
    lfm_16: {
        html: `<span class="model-badge lfm16">LFM · 1.6B</span><br><br>
               LFM2.5-VL 1.6B by LiquidAI. Larger liquid-state model offering
               enhanced reasoning &amp; richer visual understanding.`,
        bg: 'rgba(107,203,119,0.07)', border: 'rgba(107,203,119,0.25)',
    },
    qwen_unredacted: {
        html: `<span class="model-badge qunred">QWEN 3.5 · 2B UNREDACTED MAX</span><br><br>
               Qwen3.5-2B-Unredacted-MAX by prithivMLmods. Fine-tuned variant of Qwen3.5-2B
               with uncensored &amp; extended instruction-following capabilities.`,
        bg: 'rgba(255,80,160,0.07)', border: 'rgba(255,80,160,0.25)',
    },
    qwen25_vl_3b: {
        html: `<span class="model-badge q25vl3b">QWEN 2.5-VL · 3B</span><br><br>
               Qwen2.5-VL-3B-Instruct by Alibaba Cloud. Powerful 3B vision-language model
               with strong grounding, OCR &amp; multi-task visual reasoning.`,
        bg: 'rgba(80,180,255,0.07)', border: 'rgba(80,180,255,0.25)',
    },
};
modelSelect.onchange = () => {
    const info = MODEL_INFO[modelSelect.value];
    if (!info) return;
    modelInfoBox.innerHTML = info.html;
    modelInfoBox.style.background = info.bg;
    modelInfoBox.style.border = `1px solid ${info.border}`;
};

// ══════════════════════════════════════════════
//  CATEGORY PLACEHOLDER
// ══════════════════════════════════════════════
const categorySelect = document.getElementById('categorySelect');
const promptInput    = document.getElementById('promptInput');
const PLACEHOLDERS = {
    Query:   'e.g., Count the total number of boats and describe the environment.',
    Caption: 'e.g., short | normal | detailed',
    Point:   'e.g., The gun held by the person.',
    Detect:  'e.g., The headlight of the car.',
};
categorySelect.onchange = e => { promptInput.placeholder = PLACEHOLDERS[e.target.value] || ''; };

// ══════════════════════════════════════════════
//  COPY BUTTON
// ══════════════════════════════════════════════
const copyBtn   = document.getElementById('copyBtn');
const outputBox = document.getElementById('outputBox');
let   copyTimer = null;

function resetCopyBtn() {
    copyBtn.classList.remove('copied');
    copyBtn.innerHTML = `
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
        </svg> COPY`;
}
copyBtn.onclick = () => {
    const txt = outputBox.innerText || '';
    if (!txt || txt === 'Results will stream here...') return;
    navigator.clipboard.writeText(txt).then(() => {
        copyBtn.classList.add('copied');
        copyBtn.innerHTML = `
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"/>
            </svg> COPIED`;
        clearTimeout(copyTimer);
        copyTimer = setTimeout(resetCopyBtn, 2000);
    }).catch(() => {
        const ta = document.createElement('textarea');
        ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select(); document.execCommand('copy');
        document.body.removeChild(ta);
    });
};

// ══════════════════════════════════════════════
//  GROUNDING DISPLAY  (server-side annotated image)
// ══════════════════════════════════════════════
const overlayImg        = document.getElementById('overlayImg');
const groundPlaceholder = document.getElementById('groundPlaceholder');
const downloadBtn       = document.getElementById('downloadBtn');
const dotGnd            = document.getElementById('dot-gnd');

function showOverlay(b64png) {
    const src = 'data:image/png;base64,' + b64png;
    overlayImg.src = src;
    overlayImg.style.display = 'block';
    groundPlaceholder.style.display = 'none';
    dotGnd.classList.add('active');

    // Update download button
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    downloadBtn.href     = src;
    downloadBtn.download = `grounding_${ts}.png`;
    downloadBtn.style.display = 'flex';
}

function resetOverlay(msg) {
    overlayImg.src = '';
    overlayImg.style.display = 'none';
    groundPlaceholder.textContent = msg || 'Active for Point / Detect tasks.\nRun inference to visualise.';
    groundPlaceholder.style.display = 'flex';
    downloadBtn.style.display = 'none';
    dotGnd.classList.remove('active');
}

// ══════════════════════════════════════════════
//  RUN INFERENCE
// ══════════════════════════════════════════════
const runBtn    = document.getElementById('runBtn');
const btnLoader = document.getElementById('btnLoader');
const allWires  = ['wire-img-task','wire-model-task','wire-task-out','wire-task-gnd'];
const dotTask   = document.getElementById('dot-task');
const dotOut    = document.getElementById('dot-out');

runBtn.onclick = async () => {
    if (!currentFile) { alert('Please upload an image into the Input Node.'); return; }
    const promptStr = promptInput.value.trim();
    if (!promptStr)  { alert('Please enter a prompt directive.'); return; }

    // ── Reset UI ──────────────────────────────
    runBtn.disabled = true;
    btnLoader.style.display = 'inline-block';
    outputBox.innerText = '';
    outputBox.style.color = '';
    dotTask.classList.add('active');
    dotOut.classList.remove('active');
    allWires.forEach(id => document.getElementById(id)?.classList.add('active'));
    resetCopyBtn();
    resetOverlay('Running inference…');

    const category = categorySelect.value;
    const modelId  = modelSelect.value;

    // ── Step 1: stream text from /api/run ─────
    const formData = new FormData();
    formData.append('image',    currentFile);
    formData.append('category', category);
    formData.append('prompt',   promptStr);
    formData.append('model_id', modelId);

    let fullText = '';

    try {
        const response = await fetch('/api/run', { method: 'POST', body: formData });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Execution failed.');
        }

        const reader  = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6);
                if (payload === '[DONE]') break;
                try {
                    const data = JSON.parse(payload);
                    if (data.chunk) {
                        fullText += data.chunk;
                        outputBox.innerText = fullText;
                        outputBox.scrollTop = outputBox.scrollHeight;
                    }
                } catch (_) {}
            }
        }

        dotOut.classList.add('active');

        // ── Step 2: if Point or Detect → call /api/annotate ──
        if ((category === 'Point' || category === 'Detect') && fullText.trim()) {
            groundPlaceholder.textContent = 'Annotating image…';
            groundPlaceholder.style.display = 'flex';

            try {
                const annotForm = new FormData();
                annotForm.append('image',    currentFile);
                annotForm.append('text',     fullText);
                annotForm.append('category', category);

                const annotResp = await fetch('/api/annotate', {
                    method: 'POST', body: annotForm,
                });
                if (!annotResp.ok) throw new Error('Annotation request failed');

                const annotData = await annotResp.json();
                if (annotData.error) {
                    resetOverlay('Annotation error: ' + annotData.error);
                } else if (annotData.image_b64) {
                    showOverlay(annotData.image_b64);
                } else {
                    resetOverlay('No coordinates found in model output.');
                }
            } catch (annotErr) {
                resetOverlay('Annotation failed: ' + annotErr.message);
                console.error('Annotation error:', annotErr);
            }
        } else if (category !== 'Point' && category !== 'Detect') {
            resetOverlay('Active for Point / Detect tasks.\nRun inference to visualise.');
        }

    } catch (err) {
        outputBox.innerText = `[Error] ${err.message}`;
        outputBox.style.color = '#ff6b6b';
        resetOverlay('Inference error — see Output Stream node.');
    } finally {
        runBtn.disabled = false;
        btnLoader.style.display = 'none';
        dotTask.classList.remove('active');
        allWires.forEach(id => document.getElementById(id)?.classList.remove('active'));
    }
};
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.launch(show_error=True, ssr_mode=False)