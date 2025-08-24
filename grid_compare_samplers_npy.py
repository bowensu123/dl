# save as: grid_compare_samplers.py
import os, argparse, time, gc
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch_npu
from PIL import Image, ImageDraw, ImageFont

from diffusers import (
    FluxPipeline, StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler, UniPCMultistepScheduler, HeunDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler,
    DDIMScheduler, PNDMScheduler, FlowMatchEulerDiscreteScheduler, DDPMWuerstchenScheduler
)

def _resolve_model_path(repo: str, local: Optional[str] = None) -> str:
    if local and os.path.isdir(local):
        return local
    root = os.getenv("HF_LOCAL_MODELS")
    if root:
        cand = os.path.join(root, repo.replace("/", os.sep))
        if os.path.isdir(cand):
            return cand
    return repo

SCHED_MAP = {
    "dpmpp": DPMSolverMultistepScheduler,
    "unipc": UniPCMultistepScheduler,
    "heun": HeunDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
    "flow_euler": FlowMatchEulerDiscreteScheduler,
    "ddpm_wuerstchen": DDPMWuerstchenScheduler,
}

MODELS = {
    "FLUX.1-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "local": "/mnt/research/huanglx_group/FLUX.1-schnell",
        "kind": "flux",
        "default_cfg": 0.0,
        "height": 768, "width": 1360,
        "compatible_samplers": ["flow_euler"],
        "extra": {"max_sequence_length": 256},
    },
    "Segmind-Vega": {
        "repo": "segmind/Segmind-Vega", "kind": "sdxl",
        "default_cfg": 7.5, "height": 1024, "width": 1024,
        "compatible_samplers": ["dpmpp", "unipc", "heun", "euler", "euler_a", "lms", "ddim", "pndm"],
    },
    "SSD-1B": {
        "repo": "segmind/SSD-1B", "kind": "sdxl",
        "default_cfg": 9.0, "height": 1024, "width": 1024,
        "compatible_samplers": ["dpmpp", "unipc", "heun", "euler_a", "ddim", "pndm"],
    },
}

def load_pipe(entry: Dict[str, Any], torch_dtype):
    repo, kind = entry["repo"], entry["kind"]
    local = entry.get("local")
    model_path = _resolve_model_path(repo, local)

    token = None
    if model_path == repo:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    if kind == "flux":
        pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, token=token)
    elif kind == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, token=token)
    elif kind == "wuerstchen":
        from diffusers import WuerstchenPipeline
        pipe = WuerstchenPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, token=token)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    if torch.npu.is_available():
        try:
            pipe.to("npu")
        except Exception as e:
            print(f"Warning: Failed to move model to NPU: {e}")
            print("Falling back to CPU")
    pipe.set_progress_bar_config(disable=True)
    return pipe

def set_scheduler(pipe, sched_name: str):
    cls = SCHED_MAP[sched_name]
    pipe.scheduler = cls.from_config(pipe.scheduler.config)

def run_once(pipe, entry, prompt: str, negative: str, steps: int, cfg: float, seed: int):
    H, W = entry["height"], entry["width"]
    extra = entry.get("extra", {})
    device = "npu" if torch.npu.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(
        prompt=prompt, negative_prompt=negative,
        height=H, width=W, num_inference_steps=steps,
        guidance_scale=cfg, generator=gen
    )
    kwargs.update(extra)
    with torch.inference_mode():
        out = pipe(**kwargs)
    if hasattr(out, "images"):
        return out.images[0]
    if isinstance(out, list) and len(out) > 0:
        return out[0]
    raise RuntimeError("Pipeline output format not recognized.")

def _get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _wrap_text(text: str, draw: ImageDraw.ImageDraw, max_width: int, font) -> str:
    words = text.split(" ")
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if draw.textlength(test, font=font) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)

def make_grid(
    rows_imgs: List[List[Image.Image]],
    col_titles: List[str],
    row_titles: List[str],
    out_path: str,
    pad=8,
    title=None
):
    assert len(rows_imgs) > 0 and all(len(r) > 0 for r in rows_imgs), "rows_imgs must be a non-empty 2D list"
    rows = len(rows_imgs)
    cols = len(rows_imgs[0])
    assert all(len(r) == cols for r in rows_imgs), "all rows must have the same number of columns"

    W = max(im.width for row in rows_imgs for im in row)
    H = max(im.height for row in rows_imgs for im in row)
    tiles = [[(im if (im.width == W and im.height == H) else im.resize((W, H), Image.LANCZOS))
              for im in row] for row in rows_imgs]

    out_w = cols * W + (cols - 1) * pad
    out_h = rows * H + (rows - 1) * pad
    out_img = Image.new("RGB", (out_w, out_h))

    def _h_strip(left_img, right_img, pad):
        if pad <= 0: return None
        L = np.array(left_img)[:, -1:, :].astype(np.float32)
        R = np.array(right_img)[:, :1, :].astype(np.float32)
        Hh = L.shape[0]
        w = np.linspace(0.0, 1.0, pad, dtype=np.float32)[None, :, None]
        strip = (1 - w) * np.repeat(L, pad, axis=1) + w * np.repeat(R, pad, axis=1)
        return Image.fromarray(np.clip(strip + 0.5, 0, 255).astype(np.uint8))

    def _v_strip(top_img, bot_img, pad):
        if pad <= 0: return None
        T = np.array(top_img)[-1:, :, :].astype(np.float32)
        B = np.array(bot_img)[:1, :, :].astype(np.float32)
        Wh = T.shape[1]
        h = np.linspace(0.0, 1.0, pad, dtype=np.float32)[:, None, None]
        strip = (1 - h) * np.repeat(T, pad, axis=0) + h * np.repeat(B, pad, axis=0)
        return Image.fromarray(np.clip(strip + 0.5, 0, 255).astype(np.uint8))

    def _corner(tl, tr, bl, br, pad):
        if pad <= 0: return None
        TL = np.array(tl)[-1:, -1:, :].astype(np.float32)
        TR = np.array(tr)[-1:, :1, :].astype(np.float32)
        BL = np.array(bl)[:1, -1:, :].astype(np.float32)
        BR = np.array(br)[:1, :1, :].astype(np.float32)
        u = np.linspace(0.0, 1.0, pad, dtype=np.float32)[None, :, None]
        v = np.linspace(0.0, 1.0, pad, dtype=np.float32)[:, None, None]
        TLm = np.repeat(np.repeat(TL, pad, axis=0), pad, axis=1)
        TRm = np.repeat(np.repeat(TR, pad, axis=0), pad, axis=1)
        BLm = np.repeat(np.repeat(BL, pad, axis=0), pad, axis=1)
        BRm = np.repeat(np.repeat(BR, pad, axis=0), pad, axis=1)
        top_mix = (1 - u) * TLm + u * TRm
        bot_mix = (1 - u) * BLm + u * BRm
        corner = (1 - v) * top_mix + v * bot_mix
        return Image.fromarray(np.clip(corner + 0.5, 0, 255).astype(np.uint8))

    for i in range(rows):
        for j in range(cols):
            x = j * (W + pad)
            y = i * (H + pad)
            out_img.paste(tiles[i][j], (x, y))

            if j < cols - 1 and pad > 0:
                strip = _h_strip(tiles[i][j], tiles[i][j+1], pad)
                out_img.paste(strip, (x + W, y))

            if i < rows - 1 and pad > 0:
                strip = _v_strip(tiles[i][j], tiles[i+1][j], pad)
                out_img.paste(strip, (x, y + H))

            if i < rows - 1 and j < cols - 1 and pad > 0:
                cimg = _corner(tiles[i][j], tiles[i][j+1], tiles[i+1][j], tiles[i+1][j+1], pad)
                out_img.paste(cimg, (x + W, y + H))

    out_img.save(out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="grids_out")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--prompts", nargs="*", default=[
        "A futuristic Huawei headquarters skyscraper, shimmering glass facade reflecting neon city lights, cyberpunk atmosphere, ultra-detailed, cinematic wide shot.",
        "A sleek futuristic sports car with glowing neon edges racing through a rainy cyberpunk street, reflections on wet asphalt, cinematic speed blur, ultra-detailed concept art",
        "A traditional Chinese meal of braised pork knuckle rice served in a festive setting, with red lanterns, golden ingots, and the Chinese character for 'fortune' glowing in the background."
    ])
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--steps_list", nargs="*", type=int, default=[8, 16, 32])
    ap.add_argument("--samplers", nargs="*", default=["dpmpp", "unipc", "flow_euler"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    pipes: Dict[str, Any] = {}
    for m in args.models:
        pipes[m] = load_pipe(MODELS[m], torch_dtype)

    for sampler in args.samplers:
        for m in args.models:
            entry = MODELS[m]
            if sampler not in entry["compatible_samplers"]:
                print(f"[SKIP] {m} not compatible with sampler {sampler}")
                continue
            try:
                set_scheduler(pipes[m], sampler)
            except Exception as e:
                print(f"[WARN] set_scheduler failed for {m} / {sampler}: {e}")
                continue

            row_imgs: List[List[Image.Image]] = []
            for prompt in args.prompts:
                row: List[Image.Image] = []
                for steps in args.steps_list:
                    cfg = entry["default_cfg"]
                    t0 = time.perf_counter()
                    img = run_once(pipes[m], entry, prompt, args.negative, steps, cfg, args.seed)
                    if torch.npu.is_available():
                        torch.npu.synchronize()
                    dt = time.perf_counter() - t0
                    row.append(img)
                    print(f"[OK] {m} | {sampler} | {steps} | {dt:.2f}s")
                row_imgs.append(row)

            col_titles = [f"{s} steps" for s in args.steps_list]
            grid_path = os.path.join(
                args.outdir,
                f"GRID_model-{m}_sampler-{sampler}_steps-{'-'.join(map(str, args.steps_list))}.png"
            )
            make_grid(
                row_imgs,
                col_titles=col_titles,
                row_titles=args.prompts,
                out_path=grid_path,
                title=f"Model: {m}   |   Sampler: {sampler}"
            )
            print(f"[GRID] {grid_path}")

    for m in list(pipes.keys()):
        del pipes[m]
    gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()

if __name__ == "__main__":
    main()