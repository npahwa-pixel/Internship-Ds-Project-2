from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, List, Tuple

import gradio as gr


def _ensure_py_path():
    import sys

    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    src_str = str(root / "src")
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _patch_gradio_client_bool_schema() -> None:
    """
    Fix: gradio_client.utils.get_type(schema) crashes when schema is bool.
    """
    try:
        from gradio_client import utils as client_utils  # type: ignore
    except Exception:
        return

    if getattr(client_utils, "_BOOL_SCHEMA_PATCHED", False):
        return

    orig_get_type = getattr(client_utils, "get_type", None)
    if not callable(orig_get_type):
        return

    def get_type_patched(schema):  # type: ignore
        if isinstance(schema, bool):
            return "boolean"
        return orig_get_type(schema)

    client_utils.get_type = get_type_patched  # type: ignore
    client_utils._BOOL_SCHEMA_PATCHED = True  # type: ignore


def _ensure_no_proxy_for_localhost() -> None:
    needed = ["localhost", "127.0.0.1", "::1"]
    for key in ("no_proxy", "NO_PROXY"):
        cur = os.environ.get(key, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        changed = False
        for host in needed:
            if host not in parts:
                parts.append(host)
                changed = True
        if changed:
            os.environ[key] = ",".join(parts)


_ensure_py_path()
_patch_gradio_client_bool_schema()

# Project imports (these match your patch structure)
from src.model_registry import discover_models, get_spec_by_key, ModelSpec
from src.model_loader import load_any_model
from src.infer_service import infer_text, render_highlight_html
from src.output_format import format_output
from src.storage_jsonl import append_jsonl, read_jsonl, utc_now_iso
from src.active_learning import get_uncertain_sample

DEFAULT_JSONL = "active_learning/corrections.jsonl"

# Important: Dataframe must NOT start as undefined/empty in some Gradio versions
EMPTY_TABLE_ROW = [["", "", 0, 0]]  # type, text, start, end


def specs_to_choices(specs: List[ModelSpec]) -> List[Tuple[str, str]]:
    return [(s.display_name, s.key) for s in specs]


def _format_meta(spec: ModelSpec | None) -> str:
    if spec is None:
        return "No model selected."
    return (
        f"**Key:** {spec.key}\n\n"
        f"**Kind:** {spec.kind}\n\n"
        f"**Path:** {spec.path}\n\n"
        f"**Base model (if LoRA):** {spec.base_model or '-'}\n"
    )


def _spec_from_custom_path(custom_path: str) -> ModelSpec:
    p = Path(custom_path).expanduser().resolve()
    if not p.exists():
        raise gr.Error(f"Custom path not found: {p}")

    # If it's an adapter dir, treat as LoRA
    kind = "lora" if (p / "adapter_config.json").exists() else "head"
    return ModelSpec(
        key=f"custom::{p.name}",
        display_name=f"Custom · {p.name}",
        kind=kind,
        path=p,
        base_model=None,
    )


def build_app(models_dir: str, jsonl_path: str) -> gr.Blocks:
    def refresh_models(models_dir_in: str):
        new_specs = discover_models(models_dir_in)
        choices = specs_to_choices(new_specs)
        if not choices:
            return gr.Dropdown(choices=[], value=None), "No models found in that folder."
        return gr.Dropdown(choices=choices, value=choices[0][1]), _format_meta(new_specs[0])

    def _get_spec(models_dir_in: str, key: str, custom_path: str) -> ModelSpec:
        if custom_path and custom_path.strip():
            return _spec_from_custom_path(custom_path)

        specs = discover_models(models_dir_in)
        try:
            return get_spec_by_key(specs, key)
        except Exception:
            raise gr.Error("Selected model not found. Click Refresh models.")

    def run_infer(models_dir_in: str, model_key: str, custom_model: str, device: str, out_fmt: str, text: str):
        spec = _get_spec(models_dir_in, model_key, custom_model)

        # ✅ FIX: model_loader expects (kind, path, device_str=...)
        lm = load_any_model(spec.kind, spec.path, device_str=device)

        res = infer_text(text, lm)  # ✅ FIX: infer_text(text, lm)
        entities = res.entities

        html = render_highlight_html(text, entities)
        out = format_output(out_fmt, entities)  # ✅ FIX: format_output(fmt, entities)
        meta = _format_meta(spec)
        return html, out, meta

    def fetch_uncertain(models_dir_in: str, model_key: str, custom_model: str, device: str, dataset: str, split: str, lang: str):
        spec = _get_spec(models_dir_in, model_key, custom_model)
        lm = load_any_model(spec.kind, spec.path, device_str=device)

        # get_uncertain_sample returns {text, idx, score, tokens}
        sample = get_uncertain_sample(lm, dataset=dataset, split=split, lang=lang)
        text = sample["text"]
        idx = sample.get("idx")
        score = sample.get("score")

        res = infer_text(text, lm)
        entities = res.entities

        html = render_highlight_html(text, entities)
        table = [[e.get("type", ""), e.get("text", ""), int(e.get("start", 0)), int(e.get("end", 0))] for e in entities]
        if not table:
            table = [row[:] for row in EMPTY_TABLE_ROW]

        meta = f"idx={idx}, score={score}, dataset={dataset}, split={split}, lang={lang}"
        return text, html, table, meta

    def save_correction(jsonl_path_in: str, model_key: str, custom_model: str, dataset: str, split: str, lang: str, text: str, table: List[List[Any]]):
        entities = []
        for row in (table or []):
            if not row or len(row) < 4:
                continue
            etype, etext, start, end = row[0], row[1], row[2], row[3]
            if etype is None or etext is None:
                continue
            etype_s = str(etype).strip()
            etext_s = str(etext).strip()
            if not etype_s or not etext_s:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except Exception:
                continue
            entities.append({"type": etype_s, "text": etext_s, "start": start_i, "end": end_i})

        rec = {
            "ts": utc_now_iso(),
            "model": custom_model.strip() if custom_model and custom_model.strip() else model_key,
            "dataset": dataset,
            "split": split,
            "lang": lang,
            "text": text,
            "entities": entities,
        }

        p = Path(jsonl_path_in).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(str(p), rec)
        return f"Saved {len(entities)} entities → {p}"

    def stats(jsonl_path_in: str):
        p = Path(jsonl_path_in).expanduser()
        if not p.exists():
            return "No JSONL file yet."
        rows = read_jsonl(str(p))
        return f"rows={len(rows)}\npath={p}"

    demo = gr.Blocks(title="NER Demo (Head + LoRA)")

    with demo:
        gr.Markdown("# NER UI (Head + LoRA)\nSelect a model, input text, choose output format.")
        with gr.Row():
            models_dir_in = gr.Textbox(value=models_dir, label="Models directory", interactive=True, placeholder="models")
            refresh_btn = gr.Button("Refresh models")

        specs = discover_models(models_dir)
        choices = specs_to_choices(specs)
        default_key = choices[0][1] if choices else None

        with gr.Row():
            model_dd = gr.Dropdown(choices=choices, value=default_key, label="Model (dropdown)")
            custom_model = gr.Textbox(label="Custom model path (optional)", placeholder="/path/to/model_dir_or_adapter")
        meta = gr.Markdown(_format_meta(specs[0]) if specs else "No models found.")

        with gr.Row():
            device = gr.Dropdown(choices=["cpu", "mps", "cuda"], value="cpu", label="Device")
            out_fmt = gr.Dropdown(choices=["json", "xml", "markdown", "plain"], value="json", label="Output format")

        with gr.Tabs():
            with gr.Tab("Inference"):
                text_in = gr.Textbox(lines=6, label="Input text")
                run_btn = gr.Button("Run")
                html_out = gr.HTML(label="Highlighted entities")
                out_box = gr.Textbox(lines=12, label="Output")

                run_btn.click(
                    fn=run_infer,
                    inputs=[models_dir_in, model_dd, custom_model, device, out_fmt, text_in],
                    outputs=[html_out, out_box, meta],
                )

            with gr.Tab("Active Learning / Corrections"):
                gr.Markdown("Fetch an uncertain sample, edit entities, and save to JSONL.")
                with gr.Row():
                    dataset = gr.Dropdown(
                        choices=["conll", "conll2003", "wikiann", "ontonotes5", "wnut17"],
                        value="conll2003",
                        label="Dataset",
                    )
                    split = gr.Dropdown(choices=["train", "validation", "test"], value="validation", label="Split")
                    lang = gr.Textbox(value="en", label="Lang (wikiann only)")

                jsonl_path_in = gr.Textbox(value=jsonl_path, label="Corrections JSONL path")
                fetch_btn = gr.Button("Fetch uncertain sample")

                corr_text = gr.Textbox(lines=6, label="Text (sample)", interactive=True)
                corr_html = gr.HTML(label="Prediction highlight")

                corr_table = gr.Dataframe(
                    value=[row[:] for row in EMPTY_TABLE_ROW],
                    headers=["type", "text", "start", "end"],
                    datatype=["str", "str", "number", "number"],
                    row_count=(1, "dynamic"),
                    col_count=(4, "fixed"),
                    label="Edit entities (add/remove rows, fix type/start/end)",
                    interactive=True,
                )

                corr_meta = gr.Textbox(label="Info", interactive=False)

                with gr.Row():
                    save_btn = gr.Button("Save correction → JSONL")
                    export_btn = gr.Button("Download JSONL")
                    stats_btn = gr.Button("Show stats")

                save_status = gr.Textbox(label="Save status", interactive=False)
                stats_box = gr.Textbox(label="JSONL stats", interactive=False)
                dl = gr.File(label="Download")

                fetch_btn.click(
                    fn=fetch_uncertain,
                    inputs=[models_dir_in, model_dd, custom_model, device, dataset, split, lang],
                    outputs=[corr_text, corr_html, corr_table, corr_meta],
                )
                save_btn.click(
                    fn=save_correction,
                    inputs=[jsonl_path_in, model_dd, custom_model, dataset, split, lang, corr_text, corr_table],
                    outputs=[save_status],
                )
                stats_btn.click(fn=stats, inputs=[jsonl_path_in], outputs=[stats_box])

                def _download(path: str):
                    p = Path(path)
                    if not p.exists():
                        raise gr.Error(f"No file at: {path}")
                    return str(p)

                export_btn.click(fn=_download, inputs=[jsonl_path_in], outputs=[dl])

        refresh_btn.click(fn=refresh_models, inputs=[models_dir_in], outputs=[model_dd, meta])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--jsonl_path", type=str, default=DEFAULT_JSONL)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    demo = build_app(args.models_dir, args.jsonl_path)

    if not args.share:
        _ensure_no_proxy_for_localhost()

    q = demo.queue(default_concurrency_limit=2)
    launch_kwargs = dict(server_name=args.host, server_port=args.port, share=args.share)

    try:
        try:
            q.launch(**launch_kwargs, show_api=False)
        except TypeError:
            q.launch(**launch_kwargs)
    except ValueError as e:
        # last-resort fallback if localhost check fails due to proxy env
        if (not args.share) and ("localhost is not accessible" in str(e)):
            _ensure_no_proxy_for_localhost()
            launch_kwargs["share"] = True
            try:
                q.launch(**launch_kwargs, show_api=False)
            except TypeError:
                q.launch(**launch_kwargs)
        else:
            raise


if __name__ == "__main__":
    main()
