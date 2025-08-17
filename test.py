import os, copy, math
import torch
import torch_npu  
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# ===============================
# 1. Config
# ===============================
if torch.npu.is_available():
    device_type = "npu"
    device = torch.device("npu")
    try:
        torch.npu.set_device(0)
    except Exception:
        pass
else:
    device_type = "cpu"
    device = torch.device("cpu")

model_id = "stabilityai/stable-diffusion-2-1-base"
save_dir = "./snr_experiment_ldm"
os.makedirs(save_dir, exist_ok=True)

batch_size = 4
max_train_steps = 2000
learning_rate = 1e-4
timesteps = 1000
weight_decay = 1e-2
grad_clip_norm = 1.0
ACCUM_STEPS = 4
WARMUP_STEPS = 500
PRINT_EVERY = 100
VAL_EVERY = 200

prompts = ["a cat sitting on a chair", "a futuristic cityscape at night"]

# ===============================
# 2. Models (freeze VAE & text encoder)
# ===============================
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

to_args = dict(device=device)

text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(**to_args)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(**to_args)
for p in text_encoder.parameters(): p.requires_grad = False
for p in vae.parameters(): p.requires_grad = False
text_encoder.eval(); vae.eval()

unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(**to_args)

# 可选：channels_last（在 NPU/卷积上通常更友好）
for m in (vae, unet):
    m.to(memory_format=torch.channels_last)

# noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=timesteps,
    beta_schedule="squaredcos_cap_v2"
)

# ===============================
# 3. Datasets
# ===============================
class CIFAR10WithCaptions(Dataset):
    def __init__(self, root="./data", train=True):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.data = datasets.CIFAR10(root=root, train=train, download=True)
        self.classes = self.data.classes
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transform(image)
        caption = self.classes[label]
        return {"pixel_values": image, "captions": caption}

train_set = CIFAR10WithCaptions(train=True)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True,
    num_workers=2, pin_memory=False, persistent_workers=True
)

# fixed validation batch
val_set = CIFAR10WithCaptions(train=False)
val_loader = DataLoader(
    val_set, batch_size=8, shuffle=False, drop_last=True,
    num_workers=2, pin_memory=False, persistent_workers=True
)
val_batch = next(iter(val_loader))
val_B = val_batch["pixel_values"].size(0)
torch.manual_seed(42)
val_t = torch.randint(low=0, high=timesteps, size=(val_B,), device=device).long()
val_eps = None  # will initialize once with correct shape

# ===============================
# 4. Loss weighting
# ===============================
alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

def get_loss_weight(t, mode="uniform"):
    alpha_t = alphas_cumprod[t]
    sigma_t = (1 - alpha_t).clamp(min=1e-12).sqrt()
    snr = alpha_t / (sigma_t ** 2 + 1e-12)
    if mode == "uniform": return torch.ones_like(snr)
    if mode == "alpha_bar": return alpha_t
    if mode == "snr": return snr
    if mode == "log_snr": return torch.log(snr + 1e-12)
    raise ValueError("Unknown mode")

# ===============================
# 5. Optimizer, Scheduler, EMA
# ===============================
def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return float(step + 1) / float(max(1, WARMUP_STEPS))
    progress = (step - WARMUP_STEPS) / max(1, (max_train_steps - WARMUP_STEPS))
    progress = min(1.0, max(0.0, progress))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))
    def copy_to(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.shadow.items():
                if k in msd:
                    msd[k].copy_(v)

# ===============================
# 6. Validation (fixed t/eps/latents)
# ===============================
@torch.no_grad()
def eval_fixed_loss(unet_eval, weighting="uniform"):
    global val_eps
    unet_eval.eval()
    images = val_batch["pixel_values"].to(device, non_blocking=True)
    caps = val_batch["captions"]
    posterior = vae.encode(images).latent_dist
    latents = posterior.mean * 0.18215  # deterministic
    if (val_eps is None) or (val_eps.shape != latents.shape):
        torch.manual_seed(123)
        val_eps = torch.randn_like(latents)
    inputs = tokenizer(caps, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    text_emb = text_encoder(**inputs).last_hidden_state
    noisy = noise_scheduler.add_noise(latents, val_eps, val_t)
    pred = unet_eval(noisy, val_t, encoder_hidden_states=text_emb).sample
    per_sample = (pred - val_eps).pow(2).mean(dim=(1,2,3))
    weights = get_loss_weight(val_t, weighting)
    return (per_sample * weights).mean().item()

# ===============================
# 7. Training
# ===============================
ratios = ["uniform", "alpha_bar", "snr"]
loss_history = {w: [] for w in ratios}
val_history = {w: [] for w in ratios}

use_amp = (device_type == "npu")
scaler = GradScaler(enabled=use_amp)

for weighting in ratios:
    print(f"==== Training with {weighting} weighting ====")

    unet.to(device=device, dtype=torch.float32)
    optimizer = build_optimizer(unet)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(unet, decay=0.999)

    unet.train()
    step = 0
    accum_count = 0

    for batch in train_loader:
        if step >= max_train_steps:
            break

        images = batch["pixel_values"].to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(images).latent_dist
            latents = posterior.mean * 0.18215
            inputs = tokenizer(batch["captions"], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            text_emb = text_encoder(**inputs).last_hidden_state
            t = torch.randint(0, timesteps, (latents.shape[0],), device=device).long()
            eps = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, eps, t)

        with autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            pred = unet(noisy_latents, t, encoder_hidden_states=text_emb).sample
            per_sample = (pred - eps).pow(2).mean(dim=(1,2,3))
            weights = get_loss_weight(t, weighting)
            loss = (per_sample * weights).mean() / ACCUM_STEPS

        scaler.scale(loss).backward()
        accum_count += 1

        if accum_count % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(unet)

            loss_history[weighting].append(loss.item() * ACCUM_STEPS)
            if len(loss_history[weighting]) % (PRINT_EVERY // max(1, ACCUM_STEPS)) == 0:
                window = min(200, len(loss_history[weighting]))
                rolling = sum(loss_history[weighting][-window:]) / window
                print(f"[{weighting}] step {step:5d}  loss {loss.item() * ACCUM_STEPS:.4f}  ema{window}={rolling:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

            if len(loss_history[weighting]) % (VAL_EVERY // max(1, ACCUM_STEPS)) == 0:
                unet_eval = copy.deepcopy(unet).to(device)
                ema.copy_to(unet_eval)
                v = eval_fixed_loss(unet_eval, weighting=weighting)
                val_history[weighting].append((step, v))
                print(f"[{weighting}] step {step:5d}  VAL_fixed {v:.4f}")
                del unet_eval

            step += 1
            accum_count = 0

    ema.copy_to(unet)
    torch.save(unet.state_dict(), f"{save_dir}/unet_{weighting}.pt")

    # ===============================
    # 8. Inference (safe FP16 copy)
    # ===============================
    unet.eval()
    if device_type == "npu":
        unet_for_infer = copy.deepcopy(unet).to(device=device, dtype=torch.float16)
        pipe_dtype = torch.float16
    else:
        unet_for_infer = copy.deepcopy(unet).to(device=device, dtype=torch.float32)
        pipe_dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, unet=unet_for_infer, torch_dtype=pipe_dtype
    ).to(device)

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    images_out = []
    with torch.no_grad(), autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type=="npu")):
        for prompt in prompts:
            img = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            images_out.append(img)

    for i, img in enumerate(images_out):
        img.save(f"{save_dir}/sample_{weighting}_{i}.png")

# ===============================
# 9. Plot Training Losses
# ===============================
plt.figure(figsize=(10,6))
for weighting, losses in loss_history.items():
    plt.plot(losses, label=weighting)
plt.xlabel("Optimizer Steps (effective)"); plt.ylabel("Loss")
plt.title("Training Loss (per optimizer step) with Different SNR Weightings")
plt.legend(); plt.savefig(f"{save_dir}/loss_curves.png"); plt.show()

# ===============================
# 10. Plot Validation Losses
# ===============================
plt.figure(figsize=(10,6))
for weighting, vals in val_history.items():
    if len(vals) == 0: continue
    xs = [s for s, _ in vals]; ys = [v for _, v in vals]
    plt.plot(xs, ys, marker='o', label=f"{weighting} (fixed val)")
plt.xlabel("Optimizer Steps"); plt.ylabel("Fixed-batch Weighted Loss")
plt.title("Validation (Fixed Latents/t/eps) Across Weightings")
plt.legend(); plt.savefig(f"{save_dir}/val_curves.png"); plt.show()
