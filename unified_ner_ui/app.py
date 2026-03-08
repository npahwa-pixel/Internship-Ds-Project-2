import os
import re
import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import gradio as gr
from jsonschema import validate as jsonschema_validate, ValidationError
from transformers import AutoTokenizer, AutoModelForTokenClassification

import xml.etree.ElementTree as ET

try:
    from peft import PeftModel, PeftConfig  # optional (LoRA adapters)
    _PEFT_OK = True
except Exception:
    _PEFT_OK = False


# -----------------------------
# Auto-detect project root (no env var required)
# -----------------------------
def resolve_project_root() -> str:
    here = Path(__file__).resolve()
    candidates = [here.parent.parent, Path.cwd().resolve()] + list(here.parents)

    required = [
        "ner_json_conll_deberta",
        "ner_json_deberta_fullft_multi",
        "ner_json_deberta_fullft_multi_4ds",
    ]

    for c in candidates:
        if all((c / r).exists() for r in required):
            return str(c)

    for c in candidates:
        if any((c / r).exists() for r in required):
            return str(c)

    return str(Path.cwd().resolve())


# -----------------------------
# JSON schema (Option-A)
# -----------------------------
NER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["PERSON", "ORG", "LOC", "MISC"]},
                    "text": {"type": "string"},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["type", "text", "start", "end"],
                "additionalProperties": True,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}


# -----------------------------
# Tokenization with offsets
# -----------------------------
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for m in TOKEN_PATTERN.finditer(text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


# -----------------------------
# BIO span builder
# -----------------------------
def bio_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    spans = []
    cur = None  # (s, e, typ)

    def flush():
        nonlocal cur
        if cur is not None:
            spans.append(cur)
            cur = None

    for i, tag in enumerate(tags):
        if tag is None or tag == "O":
            flush()
            continue
        if "-" not in tag:
            flush()
            continue
        p, typ = tag.split("-", 1)
        p = p.upper()
        if p == "B" or cur is None or cur[2] != typ:
            flush()
            cur = (i, i + 1, typ)
        else:
            cur = (cur[0], i + 1, typ)
    flush()
    return spans


def map_type_suffix(typ: str) -> str:
    t = typ.upper()
    if t in {"PER", "PERSON"}:
        return "PERSON"
    if t in {"ORG", "ORGANIZATION", "CORPORATION", "COMPANY", "AGENCY", "GROUP"}:
        return "ORG"
    if t in {"LOC", "LOCATION", "GPE", "FAC", "FACILITY", "GEOPOLITICAL_AREA"}:
        return "LOC"
    return "MISC"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def empty_device_cache():
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# -----------------------------
# Model discovery helpers
# -----------------------------
def _has_full_weights(dir_path: str) -> bool:
    return (
        os.path.isfile(os.path.join(dir_path, "config.json"))
        and (
            os.path.isfile(os.path.join(dir_path, "model.safetensors"))
            or os.path.isfile(os.path.join(dir_path, "pytorch_model.bin"))
        )
    )

def _is_lora_dir(dir_path: str) -> bool:
    return (
        os.path.isfile(os.path.join(dir_path, "adapter_config.json"))
        or os.path.isfile(os.path.join(dir_path, "adapter_model.safetensors"))
    )

def _list_checkpoints(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    ckpts = []
    for name in os.listdir(base_dir):
        if name.startswith("checkpoint-"):
            p = os.path.join(base_dir, name)
            if os.path.isdir(p):
                ckpts.append(p)

    def key(p: str) -> int:
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=key)
    return ckpts

def expand_models(models_map: Dict[str, str]) -> Dict[str, str]:
    expanded: Dict[str, str] = {}
    for label, path in models_map.items():
        if not os.path.isdir(path):
            expanded[label] = path
            continue

        ckpts = _list_checkpoints(path)
        added_any = False

        if _has_full_weights(path) or _is_lora_dir(path):
            expanded[f"{label} (final)"] = path
            added_any = True

        for ckpt in ckpts:
            if _has_full_weights(ckpt) or _is_lora_dir(ckpt):
                expanded[f"{label} ({os.path.basename(ckpt)})"] = ckpt
                added_any = True

        if not added_any:
            expanded[label] = path

    return expanded


# -----------------------------
# Default models.json + loader
# -----------------------------
def default_models(project_root: str) -> Dict[str, str]:
    return {
        "CoNLL DeBERTa head": os.path.join(
            project_root,
            "ner_json_conll_deberta",
            "ner_json_conll_deberta",
            "models",
            "deberta_conll_head",
        ),
        "FullFT multi (3 datasets) checkpoint-14319": os.path.join(
            project_root,
            "ner_json_deberta_fullft_multi",
            "ner_json_deberta_fullft_multi",
            "models",
            "deberta_fullft_multi",
            "checkpoint-14319",
        ),
        "FullFT multi (4 datasets) checkpoint-12538": os.path.join(
            project_root,
            "ner_json_deberta_fullft_multi_4ds",
            "ner_json_deberta_fullft_multi_4ds",
            "models",
            "deberta_fullft_multi_4ds",
            "checkpoint-12538",
        ),
    }

def load_models_config(project_root: str) -> Dict[str, str]:
    cfg_path = os.path.join(os.path.dirname(__file__), "models.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = {k: v.replace("{PROJECT_ROOT}", project_root) for k, v in data.items()}
            return expand_models(out)
        except Exception:
            pass
    return expand_models(default_models(project_root))


# -----------------------------
# Loading: HF model or LoRA adapter
# -----------------------------
@dataclass
class LoadedModel:
    model_dir: str
    tokenizer: Any
    model: Any
    device: str
    id2label: Dict[int, str]
    is_lora: bool

STATE: Dict[str, Any] = {
    "loaded": None,
    "loaded_key": None,
    "max_length": int(os.environ.get("MAX_LENGTH", "192")),
}

def load_token_classifier(model_dir: str) -> LoadedModel:
    device = pick_device()

    if STATE.get("loaded") is not None:
        try:
            del STATE["loaded"]
        except Exception:
            pass
        STATE["loaded"] = None
        empty_device_cache()

    if _is_lora_dir(model_dir):
        if not _PEFT_OK:
            raise RuntimeError("LoRA adapter detected but 'peft' is not installed.")
        peft_cfg = PeftConfig.from_pretrained(model_dir)
        base_id = peft_cfg.base_model_name_or_path

        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)

        base = AutoModelForTokenClassification.from_pretrained(base_id, attn_implementation="eager")
        model = PeftModel.from_pretrained(base, model_dir)
        model.to(device)
        model.eval()
        id2label = getattr(model.config, "id2label", None) or getattr(base.config, "id2label", {})
        return LoadedModel(model_dir=model_dir, tokenizer=tok, model=model, device=device, id2label=id2label, is_lora=True)

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, attn_implementation="eager")
    model.to(device)
    model.eval()
    id2label = getattr(model.config, "id2label", {})
    return LoadedModel(model_dir=model_dir, tokenizer=tok, model=model, device=device, id2label=id2label, is_lora=False)

def ensure_loaded(model_key: str, model_dir: str) -> LoadedModel:
    if STATE.get("loaded") is None or STATE.get("loaded_key") != model_key:
        STATE["loaded"] = load_token_classifier(model_dir)
        STATE["loaded_key"] = model_key
    return STATE["loaded"]


# -----------------------------
# Output builders: JSON / XML / Plain
# -----------------------------
def build_entities(text: str, offsets: List[Tuple[int, int]], tags: List[str], confs: List[float]) -> List[Dict[str, Any]]:
    spans = bio_spans(tags)
    ents: List[Dict[str, Any]] = []
    for s_tok, e_tok, typ in spans:
        start = offsets[s_tok][0]
        end = offsets[e_tok - 1][1]
        ent_text = text[start:end]
        confidence = float(min(confs[s_tok:e_tok])) if e_tok > s_tok else 0.0
        ents.append(
            {
                "type": map_type_suffix(typ),
                "text": ent_text,
                "start": int(start),
                "end": int(end),
                "confidence": confidence,
            }
        )
    return ents

def build_json_output(entities: List[Dict[str, Any]]) -> str:
    obj = {"entities": entities}
    jsonschema_validate(instance=obj, schema=NER_JSON_SCHEMA)
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _indent_xml(elem: ET.Element, level: int = 0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def build_xml_output(entities: List[Dict[str, Any]]) -> str:
    root = ET.Element("entities")
    for e in entities:
        node = ET.SubElement(
            root,
            "entity",
            attrib={
                "type": str(e.get("type", "")),
                "start": str(e.get("start", "")),
                "end": str(e.get("end", "")),
                "confidence": f'{float(e.get("confidence", 0.0)):.4f}',
            },
        )
        node.text = e.get("text", "")
    _indent_xml(root)
    return ET.tostring(root, encoding="unicode")

def build_plain_output(entities: List[Dict[str, Any]]) -> str:
    if not entities:
        return "No entities found."
    lines = []
    for e in entities:
        t = e.get("type", "")
        txt = e.get("text", "")
        s = e.get("start", "")
        en = e.get("end", "")
        c = e.get("confidence", None)
        if c is None:
            lines.append(f"- {t}: {txt} [{s},{en}]")
        else:
            lines.append(f"- {t}: {txt} [{s},{en}] (conf={float(c):.4f})")
    return "\n".join(lines)

def format_output(fmt: str, entities: List[Dict[str, Any]]) -> str:
    fmt = (fmt or "JSON").upper()
    if fmt == "XML":
        return build_xml_output(entities)
    if fmt == "PLAIN":
        return build_plain_output(entities)
    return build_json_output(entities)


# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def predict_tags(tokens: List[str], threshold: float, loaded: LoadedModel) -> Tuple[List[str], List[float]]:
    tok = loaded.tokenizer
    model = loaded.model
    device = loaded.device
    max_len = STATE["max_length"]

    enc = tok(tokens, is_split_into_words=True, truncation=True, max_length=max_len, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    logits = out.logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).tolist()
    pred_conf = torch.max(probs, dim=-1).values.tolist()

    id2label = loaded.id2label
    seen = set()
    word_tags: List[str] = []
    word_conf: List[float] = []

    for wi, pid, conf in zip(word_ids, pred_ids, pred_conf):
        if wi is None or wi in seen:
            continue
        seen.add(wi)
        lab = id2label.get(int(pid), None)
        if lab is None:
            lab = getattr(model.config, "id2label", {}).get(int(pid), "O")
        c = float(conf)
        if c < float(threshold):
            lab = "O"
        if not isinstance(lab, str) or ("-" not in lab and lab != "O"):
            lab = "O"
        word_tags.append(lab)
        word_conf.append(c)

    if len(word_tags) < len(tokens):
        pad = len(tokens) - len(word_tags)
        word_tags.extend(["O"] * pad)
        word_conf.extend([0.0] * pad)

    return word_tags[:len(tokens)], word_conf[:len(tokens)]


def run_infer(model_choice: str, custom_model_dir: str, out_format: str, text: str, threshold: float):
    project_root = resolve_project_root()
    models_map = load_models_config(project_root)

    if custom_model_dir and custom_model_dir.strip():
        chosen_dir = os.path.expanduser(custom_model_dir.strip())
        chosen_key = f"custom::{chosen_dir}"
    else:
        if model_choice not in models_map:
            return "", [], [], False, f"Unknown model choice: {model_choice}"
        chosen_dir = models_map[model_choice]
        chosen_key = model_choice

    if not os.path.isdir(chosen_dir):
        return "", [], [], False, f"Model directory not found: {chosen_dir}"

    try:
        loaded = ensure_loaded(chosen_key, chosen_dir)
    except Exception as e:
        return "", [], [], False, f"Failed to load model: {type(e).__name__}: {e}"

    tokens, offsets = tokenize_with_offsets(text or "")
    if not tokens:
        empty = {"entities": []}
        return json.dumps(empty, indent=2), [], [], True, ""

    try:
        tags, confs = predict_tags(tokens, threshold, loaded)
        entities = build_entities(text or "", offsets, tags, confs)

        output_text = format_output(out_format, entities)

        # JSON schema validity checkbox should only reflect JSON mode.
        schema_ok = True
        if (out_format or "").upper() == "JSON":
            jsonschema_validate(instance={"entities": entities}, schema=NER_JSON_SCHEMA)

        ent_rows = [[e["type"], e["text"], e["start"], e["end"], e.get("confidence", None)] for e in entities]
        tok_rows = [[i, tokens[i], offsets[i][0], offsets[i][1], tags[i], confs[i]] for i in range(len(tokens))]

        return output_text, ent_rows, tok_rows, schema_ok, ""
    except ValidationError as ve:
        return "", [], [], False, f"JSON schema validation failed: {ve.message}"
    except Exception as e:
        return "", [], [], False, f"Inference failed: {type(e).__name__}: {e}"


def ui_header(project_root: str) -> str:
    models_map = load_models_config(project_root)
    lines = [
        "# Unified NER Output UI",
        "",
        f"**Auto PROJECT_ROOT:** `{project_root}`",
        f"**Device:** `{pick_device()}`",
        "",
        "### Detected model paths (expanded checkpoints)",
    ]
    for k, v in models_map.items():
        ok = "✅" if os.path.isdir(v) else "❌"
        lines.append(f"- {ok} **{k}** → `{v}`")
    if not _PEFT_OK:
        lines.append("")
        lines.append("> Note: `peft` is not installed; LoRA adapter directories won't load.")
    return "\n".join(lines)


def main():
    project_root = resolve_project_root()
    models_map = load_models_config(project_root)

    with gr.Blocks(title="Unified NER Output UI") as demo:
        gr.Markdown(ui_header(project_root))

        model_choice = gr.Dropdown(
            choices=list(models_map.keys()),
            value=list(models_map.keys())[0] if models_map else None,
            label="Select model (includes all checkpoints)",
        )

        with gr.Accordion("Advanced: use a custom model directory", open=False):
            custom_model_dir = gr.Textbox(
                label="Custom model dir (overrides dropdown)",
                placeholder="/absolute/path/to/checkpoint-XXXXX or adapter dir",
            )

        out_format = gr.Dropdown(
            choices=["JSON", "XML", "Plain"],
            value="JSON",
            label="Output format",
        )

        inp = gr.Textbox(label="Input text", placeholder="Paste any text…", lines=7)

        with gr.Row():
            threshold = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Confidence threshold (low-confidence tokens become O)")
            btn = gr.Button("Extract", variant="primary")

        out_box = gr.Textbox(label="Output", lines=16)

        schema_ok = gr.Checkbox(label="JSON schema valid (only for JSON mode)", value=True, interactive=False)
        err = gr.Textbox(label="Error (if any)", value="", interactive=False)

        gr.Markdown("## Extracted entities")
        ent_table = gr.Dataframe(
            headers=["type", "text", "start", "end", "confidence"],
            datatype=["str", "str", "number", "number", "number"],
            col_count=(5, "fixed"),
            interactive=False,
        )

        gr.Markdown("## Token-level predictions (debug)")
        tok_table = gr.Dataframe(
            headers=["idx", "token", "start", "end", "tag", "confidence"],
            datatype=["number", "str", "number", "number", "str", "number"],
            col_count=(6, "fixed"),
            interactive=False,
        )

        btn.click(
            run_infer,
            inputs=[model_choice, custom_model_dir, out_format, inp, threshold],
            outputs=[out_box, ent_table, tok_table, schema_ok, err],
        )

    demo.launch()

if __name__ == "__main__":
    main()
