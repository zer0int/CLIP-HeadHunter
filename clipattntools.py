"""
MISC Tools for CLIP Attention Heads Max Visualization

zer0int
https://github.com/zer0int
"""
import os, json, hashlib, shutil
from typing import List
from torchvision.transforms import Resize
from PIL import Image
import numpy as np
import copy, contextlib, re, subprocess
import torch
from torch import nn as nn
from torch.nn import functional as F
from colorama import Fore, Style
import attnclip as clip
from attnclip.model import QuickGELU
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

#----------------------
# ---  CLIP MODEL   ---
#----------------------
def get_clip_dimensions(model, preprocess):
    model = model.eval()
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            input_dims = transform.size
            break
    num_layers = None
    num_features = None
    num_heads = None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features
        if hasattr(last_block, 'attn'):
            num_heads = last_block.attn.num_heads

    return input_dims, num_layers, num_features, num_heads

def clip_encode_image(premodel, img_tensor, img_name=None):
    visual = premodel.visual
    x = visual.conv1(img_tensor)                      # [B, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)         # [B, width, grid**2]
    x = x.permute(0, 2, 1)                            # [B, grid**2, width]

    class_embed = visual.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
    x = torch.cat([class_embed, x], dim=1)            # [B, grid**2 + 1, width]
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)                            # [L, B, width]
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                            # [B, L, width]
    x_cls = x[:, 0, :]                                # [B, width]
    x_ln = visual.ln_post(x_cls)
    if hasattr(visual, 'proj') and visual.proj is not None:
        x_proj = x_ln @ visual.proj
    else:
        x_proj = x_ln
    return x_proj

@torch.no_grad()
def _encode_prompt_emb(premodel, prompt: str, device):
    tokens = clip.tokenize([prompt]).to(device)
    t = premodel.encode_text(tokens).float()
    return t / (t.norm(dim=-1, keepdim=True) + 1e-8)

@torch.no_grad()
def _encode_clip_feat(premodel, post, img_tensor: torch.Tensor) -> torch.Tensor:
    # returns L2-normalized image embedding (1,D) on the same device as img_tensor
    feat = premodel.encode_image(post(img_tensor)).float()
    return feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)

@torch.no_grad()
def _encode_prompt_list(premodel, prompts_str: str | None, device):
    """
    Encode a list of prompts separated by ',' or '|'.
    Returns (emb_list, name_list). Both lists may be empty.
    """
    if not isinstance(prompts_str, str) or not prompts_str.strip():
        return [], []
    # split on comma or pipe; trim; drop empties; dedup preserving order
    raw = [p.strip() for sep in [",", "|"] for p in prompts_str.split(sep)]
    uniq = []
    seen = set()
    for p in raw:
        if p and p not in seen:
            uniq.append(p); seen.add(p)
    if not uniq:
        return [], []

    toks = clip.tokenize(uniq).to(device)
    with torch.no_grad():
        t = premodel.encode_text(toks).float()
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
    # return list of 1xD tensors for easy dotting
    return [t[i:i+1] for i in range(t.shape[0])], uniq

@torch.no_grad()
def _resize_positional_embedding(pos_embed: torch.Tensor, new_grid_hw: int) -> torch.Tensor:
    original_dim = pos_embed.dim()
    if original_dim == 2:
        pos_embed_3d = pos_embed.unsqueeze(0)
    elif original_dim == 3:
        pos_embed_3d = pos_embed
    else:
        raise ValueError(f"Unexpected pos_embed.dim()={original_dim}; expected 2 or 3.")

    cls, grid = pos_embed_3d[:, :1, :], pos_embed_3d[:, 1:, :]
    C = grid.shape[-1]
    old_hw = int((grid.shape[1]) ** 0.5)
    assert old_hw * old_hw == grid.shape[1], f"Grid length {grid.shape[1]} is not a square."

    grid = grid.reshape(1, old_hw, old_hw, C).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_grid_hw, new_grid_hw), mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid_hw * new_grid_hw, C)
    out_3d = torch.cat([cls, grid], dim=1)
    return out_3d.squeeze(0) if original_dim == 2 else out_3d


@contextlib.contextmanager # Swap pos-emb on premodel
def _temporary_pos_embed(premodel, new_pos_embed: torch.Tensor):
    visual = premodel.visual
    old_param = visual.positional_embedding
    device = old_param.device
    dtype = old_param.dtype
    old_tensor = old_param.data.detach().clone()
    visual.positional_embedding = torch.nn.Parameter(new_pos_embed.to(device=device, dtype=dtype), requires_grad=False)
    try:
        yield
    finally:
        visual.positional_embedding = torch.nn.Parameter(old_tensor.to(device=device, dtype=dtype), requires_grad=False)

# -------------
#  CLIP HOOKS
# -------------

@torch.no_grad()
def attncopy(model, target="visual", from_=None, to_=None, **kwargs):
    """
    Copy the entire attention submodule state. Negative indices allowed (-1 = last).
    from blocks in `from_` to blocks in `to_`. Sources are NOT modified.

    Args:
        model: CLIP model with `.visual.transformer.resblocks` and `.transformer.resblocks`.
        target: 'visual' or 'text'.
        from_: list[int]  (or pass via **{'from': [...]})
        to_:   list[int]  (or pass via **{'to':   [...]} )

    """
    # allow **{'from':..., 'to':...}
    if 'from' in kwargs: from_ = kwargs['from']
    if 'to'   in kwargs: to_   = kwargs['to']

    if target not in ("visual", "text"):
        raise ValueError(f"target must be 'visual' or 'text', got {target!r}")
    if not isinstance(from_, (list, tuple)) or not isinstance(to_, (list, tuple)):
        raise ValueError("Provide lists for 'from' and 'to' (e.g., from_=[22,23], to_=[2,3]).")
    if len(from_) != len(to_):
        print(f"[attncopy] WARNING: len(from)={len(from_)} != len(to)={len(to_)}; no changes applied.")
        return model

    resblocks = (
        model.visual.transformer.resblocks
        if target == "visual"
        else model.transformer.resblocks
    )
    n_blocks = len(resblocks)

    def _norm_idx(i: int) -> int:
        i = int(i)
        if i < 0:
            i = n_blocks + i
        if not (0 <= i < n_blocks):
            raise IndexError(f"Block index out of range: {i}; valid [0, {n_blocks-1}] or negative indices.")
        return i

    pairs = [(_norm_idx(a), _norm_idx(b)) for a, b in zip(from_, to_)]

    src_indices = sorted(set(a for a, _ in pairs))
    src_states = {}
    for a in src_indices:
        attn_a = getattr(resblocks[a], "attn", None)
        if attn_a is None:
            raise AttributeError(f"Missing 'attn' on resblock {a} (target={target}).")
        src_states[a] = {k: v.clone() for k, v in attn_a.state_dict().items()}

    for a, b in pairs:
        attn_b = getattr(resblocks[b], "attn", None)
        if attn_b is None:
            raise AttributeError(f"Missing 'attn' on resblock {b} (target={target}).")
        sda = src_states[a]
        sdb = attn_b.state_dict()
        if set(sda.keys()) != set(sdb.keys()):
            raise RuntimeError(f"Attention state_dict keys differ between src {a} and dst {b}.")
        for k in sda:
            if sda[k].shape != sdb[k].shape:
                raise RuntimeError(f"Shape mismatch for key '{k}' between src {a} and dst {b}: {sda[k].shape} vs {sdb[k].shape}")
        attn_b.load_state_dict(sda, strict=True)

    return model

class FeatureScalerHook: # For ablating register neurons that feed head 10 (for example).
    def __init__(self, model, layer_idx, feature_idx, scale_factor, transformer_type='visual'):
        self.model = model
        self.layer_idx = layer_idx
        self.feature_idx = feature_idx
        self.scale_factor = scale_factor
        self.transformer_type = transformer_type
        self.handle = None
        self.register_hook()

    def register_hook(self):
        def hook(module, input, output):
            output[:, :, self.feature_idx] *= self.scale_factor
            return output

        if self.transformer_type == 'visual':
            layer = self.model.visual.transformer.resblocks[self.layer_idx].mlp.c_fc
        else:
            layer = self.model.transformer.resblocks[self.layer_idx].mlp.c_fc
        self.handle = layer.register_forward_hook(hook)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

class MLPActHook:
    def __init__(self, block):
        self.activations = None
        self._handle = None
        gelu = None
        for m in block.mlp.modules():
            if isinstance(m, QuickGELU):
                gelu = m
                break
        if gelu is None:
            raise RuntimeError("QuickGELU not found in block.mlp")
        self._handle = gelu.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        self.activations = out

    def clear(self):
        self.activations = None

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

class AllHeadsCaptureHook:
    def __init__(self, block):
        self.activations = None
        self.block = block 
        self.hook_handle = block.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        seq_len, batch, embed_dim = output.shape
        num_heads = module.attn.num_heads
        head_dim = embed_dim // num_heads
        x = output.permute(1, 0, 2).contiguous()  # [batch, seq, embed_dim]
        x = x.view(batch, seq_len, num_heads, head_dim)  # [batch, seq, heads, head_dim]
        self.activations = x  # [batch, seq, num_heads, head_dim]

    def clear(self):
        self.activations = None

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

class HeadCaptureHook:
    def __init__(self, block, head_idx):
        self.activations = None
        self.head_idx = head_idx
        self.block = block
        self.hook_handle = block.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        seq_len, batch, embed_dim = output.shape
        num_heads = module.attn.num_heads
        head_dim = embed_dim // num_heads
        x = output.permute(1, 0, 2).contiguous()  # [batch, seq, embed_dim]
        x = x.view(batch, seq_len, num_heads, head_dim)  # [batch, seq, heads, head_dim]
        self.activations = x[:, :, self.head_idx, :]  # [batch, seq, head_dim]

    def clear(self):
        self.activations = None

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

# ---------------------------------
# Text Embeddings Gradient Ascent
# Uses a heavily modified version of Original CLIP Gradient Ascent Script: by Twitter / X: @advadnoun
# ---------------------------------
def _combine_ga_batches(emb: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    emb: [B,D] or [1,D] from GA; returns [1,D] L2-normalized
    """
    if emb.dim() == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)   # [1,D] -> [D] accidental cases
    if emb.dim() == 1:
        out = emb.unsqueeze(0)
    elif emb.dim() == 2:
        if mode == "sum":
            out = emb.sum(dim=0, keepdim=True)
        else:
            out = emb.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unexpected GA embed shape {tuple(emb.shape)}")
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

def load_image_for_ga(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255   
    im = F.interpolate(im, (sideX, sideY))
    return im

def augment(into, augs):
    return augs(into)

def clip_encode_text(premodel, text, many_tokens, prompt_ga):
    x = torch.matmul(text, premodel.token_embedding.weight)
    x = x + premodel.positional_embedding
    x = x.permute(1, 0, 2)
    x = premodel.transformer(x)
    x = x.permute(1, 0, 2)
    x = premodel.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt_ga) + 2] @ premodel.text_projection
    return x

class Pars(torch.nn.Module):
    def __init__(self, ga_batch_size, many_tokens, prompt_ga, positional_shape=77):
        super(Pars, self).__init__()
        self.ga_batch_size = ga_batch_size
        self.many_tokens = many_tokens
        self.prompt_ga = prompt_ga
        self.gumbel_temp = 1000

        st = torch.zeros(ga_batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())

        self.start = torch.zeros(ga_batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt_ga_embeddings = torch.zeros(ga_batch_size, len(prompt_ga), 49408).cuda()
        for jk, pt in enumerate(prompt_ga):
            self.prompt_ga_embeddings[:, jk, pt] = 1 

        pad_length = positional_shape - (self.many_tokens + len(self.prompt_ga) + 1)
        self.pad = torch.zeros(self.ga_batch_size, pad_length, 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        soft = F.gumbel_softmax(self.normu, tau=self.gumbel_temp, dim=-1, hard=True)

        return torch.cat([self.start, self.prompt_ga_embeddings, soft, self.pad], 1)

def ascend_txt(image, premodel, lats, many_tokens, prompt_ga, nom, augment, do_weird_thing):
    iii = nom(augment(image[:,:3,:,:].expand(lats.normu.shape[0], -1, -1, -1)))
    iii = premodel.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(premodel, lll, many_tokens, prompt_ga)   
    if do_weird_thing:
        loss = 100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1) # min cos sim
    else:
        loss = -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1) # max cos sim
    
    return loss, tx, lll

def train(image, premodel, lats, many_tokens, prompt_ga, optim_ga, nom, augment, do_weird_thing):
    with autocast():
        loss1, tx, lll = ascend_txt(image, premodel, lats, many_tokens, prompt_ga, nom, augment, do_weird_thing)
    loss = loss1.mean()
    optim_ga.zero_grad()
    scaler.scale(loss).backward(retain_graph=True)
    scaler.step(optim_ga)
    scaler.update()
    return loss1, tx, lll

def checkin(loss, tx, lll, tok, bests, imagename, cache_dir):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('<|startoftext|>', '').replace('<|endoftext|>', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        tokens = j.split()
        unique_tokens.update(tokens)
    os.makedirs(f"{cache_dir}/txtopinion", exist_ok=True)
    with open(f"{cache_dir}/txtopinion/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))


def generate_target_text_embeddings(img_path, premodel, lats, optim_ga, training_iterations, checkin_step, many_tokens, prompt_ga, nom, augment, tok, bests, args, cache_dir, do_weird_thing):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    input_dims = premodel.visual.input_resolution
    img_ga = load_image_for_ga(img_path, input_dims, input_dims)
    
    print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)
    scaler = GradScaler()
    best_loss = float('inf')
    best_text_embeddings = None

    for j in range(training_iterations):
        loss, tx, lll = train(img_ga, premodel, lats, many_tokens, prompt_ga, optim_ga, nom, augment, do_weird_thing)
        current_loss = loss.mean().item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_text_embeddings = copy.deepcopy(tx.detach())
            print(Fore.RED + Style.BRIGHT + f"New best loss: {best_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name, cache_dir)
            print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)

        if j % 50 == 0:
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name, cache_dir)

    os.makedirs(f"{cache_dir}/txtembeds", exist_ok=True)
    torch.save(best_text_embeddings, f"{cache_dir}/txtembeds/{img_name}_emb.pt")
    print(Fore.MAGENTA + Style.BRIGHT + f"\nBest text embedding saved to '{cache_dir}/txtembeds'.\nTokens (CLIP 'opinion') saved to 'txtopinion' folder.\n" + Fore.RESET)
    del optim_ga, lats, scaler, prompt_ga
    torch.cuda.empty_cache()
    return img_ga, best_text_embeddings, img_path, img_name

# --------------
#   Video Tool
# --------------

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _center_on_canvas_and_save(src_path: str, dst_path: str, max_side: int):
    with Image.open(src_path).convert("RGB") as im:
        w, h = im.size
        canvas = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        x = (max_side - w) // 2
        y = (max_side - h) // 2
        canvas.paste(im, (x, y))
        canvas.save(dst_path)

def _build_video_from_frames(temp_folder: str, max_side: int, steps_folder: str, framerate: int = 24):
    path_norm = temp_folder.replace("\\", "/")
    m = re.search(r"steps_H(\d+)_L(\d+)$", path_norm)
    head_tag, target_layer = None, None

    if m:
        head_tag = f"H{m.group(1)}"
        target_layer = m.group(2)
    else:
        parent = path_norm.rsplit("/", 1)[0]
        base = parent.rsplit("/", 1)[-1]
        m2 = re.match(r"^.+?_(H(?:\d+|multi_[\d\-]+))_L(\d+)_agg", base)
        if m2:
            head_tag = m2.group(1)
            target_layer = m2.group(2)

    if head_tag is None or target_layer is None:
        return False, f"[save_video] skip: could not parse head/layer from folder: {temp_folder}"

    try:
        all_entries = os.listdir(temp_folder)
    except FileNotFoundError:
        return False, f"[save_video] skip: folder not found: {temp_folder}"

    pngs = [f for f in all_entries if f.lower().endswith(".png")]
    if not pngs:
        return False, f"[save_video] no frames found in {temp_folder}"

    pngs.sort(key=_natural_key)  # ascending-natural

    temp_video_folder = os.path.join(temp_folder, "temp_video")

    with contextlib.suppress(Exception):
        shutil.rmtree(temp_video_folder)
    os.makedirs(temp_video_folder, exist_ok=True)

    for idx, fname in enumerate(pngs, start=1):
        src = os.path.join(temp_folder, fname)
        dst = os.path.join(temp_video_folder, f"{idx:04d}.png")
        try:
            _center_on_canvas_and_save(src, dst, max_side)
        except Exception as e:
            return False, f"[save_video] frame prep failed at '{src}': {e}"

    safe_tag = re.sub(r"[^A-Za-z0-9_\-]", "-", head_tag)
    output_file = os.path.join(steps_folder, f"video_{safe_tag}_L{target_layer}.mp4")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(framerate),
        "-i", os.path.join(temp_video_folder, "%04d.png"),
        "-c:v", "libx264",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        output_file
    ]

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False, (f"[save_video] ffmpeg not found. Frames saved to '{temp_video_folder}', "
                       f"but video could not be created. Reason: ffmpeg executable not in PATH.")
    try:
        proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            reason = (proc.stderr.decode("utf-8", errors="ignore") or proc.stdout.decode("utf-8", errors="ignore")).strip()
            return False, (f"[save_video] ffmpeg failed. Frames saved to '{temp_video_folder}', "
                           f"but video could not be created. Reason: {reason}")
    except FileNotFoundError as e:
        return False, (f"[save_video] ffmpeg execution error. Frames saved to '{temp_video_folder}', "
                       f"but video could not be created. Reason: {e}")
    except Exception as e:
        return False, (f"[save_video] unexpected error. Frames saved to '{temp_video_folder}', "
                       f"but video could not be created. Reason: {e}")

    shutil.rmtree(temp_video_folder)
    return True, f"[save_video] wrote '{output_file}' ({len(pngs)} frames @ {framerate} fps)"

# ---------------
#  AUGMENTATION
# ---------------

def _deepdream_init_from_image(path: str, size: int, batch_size: int = 1, device: str = "cuda:0"):
    # Load image, resize to (size, size), convert to float tensor in [0,1]
    with Image.open(path).convert("RGB") as im:
        im = im.resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0           # H W C in [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)      # 1 3 H W
    if batch_size != 1:
        t = t.repeat(batch_size, 1, 1, 1)

    # Pinned → GPU non_blocking for parity with new_init
    t = t.pin_memory().to(device, non_blocking=True)
    t = t.detach().clone()
    t.requires_grad_()
    return t

def _subpixel_jitter_(img, max_shift_px=0.5):
    B,C,H,W = img.shape
    # random tiny translations in NDC
    tx = (torch.rand(B,1,1,1, device=img.device)*2-1) * (max_shift_px*2/H)
    ty = (torch.rand(B,1,1,1, device=img.device)*2-1) * (max_shift_px*2/W)
    theta = torch.zeros(B,2,3, device=img.device)
    theta[:,0,2] = tx.squeeze()
    theta[:,1,2] = ty.squeeze()
    grid = F.affine_grid(theta, size=img.size(), align_corners=False)
    img.copy_(F.grid_sample(img, grid, mode='bilinear',
                            padding_mode='reflection', align_corners=False))

def _edge_aware_chroma_smooth_(img: torch.Tensor,
                               ksize: int = 3,
                               edge_k: float = 12.0,
                               y_gamma: float = 0.5,
                               amount: float = 1.0):
    """
    img: [B,3,H,W] leaf tensor, modified in-place
    ksize: blur kernel (3 or 5 recommended)
    edge_k: edge sensitivity (larger → less blur near edges)
    y_gamma: reduces blur in dark regions (0.4–0.7 good)
    amount: blend toward blurred chroma (0.5–1.5 typical)
    """
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    Y, U, V = _rgb_to_yuv(img)

    # Sobel grad on Y for edges
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=dtype, device=device).view(1,1,3,3) / 8.0
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=dtype, device=device).view(1,1,3,3) / 8.0
    gX = F.conv2d(Y, sobel_x, padding=1)
    gY = F.conv2d(Y, sobel_y, padding=1)
    grad = (gX.pow(2) + gY.pow(2)).sqrt()

    # normalize + build blur weight: low near edges & in darks
    grad_n = grad / (grad.mean() + 1e-6)
    Y01 = (Y + 1.0) * 0.5  # assume leaf roughly in [-1,1]; clamp guards
    Y01 = Y01.clamp(0, 1)
    blur_w = torch.exp(-edge_k * grad_n).clamp(0, 1) * (Y01.pow(y_gamma))  # 0..1

    # small blur for U/V (avg pool ≈ gentle gaussian)
    pad = ksize // 2
    Ub = F.avg_pool2d(U, kernel_size=ksize, stride=1, padding=pad)
    Vb = F.avg_pool2d(V, kernel_size=ksize, stride=1, padding=pad)

    U2 = U + (Ub - U) * (blur_w * amount)
    V2 = V + (Vb - V) * (blur_w * amount)

    img_rgb = _yuv_to_rgb(Y, U2, V2)
    img.copy_(img_rgb)   # in-place update

def _deghost_rollsnap_(img: torch.Tensor, max_shift: int = 3, alpha: float = 0.2):
    """
    img: [B,3,H,W] leaf tensor, modified in-place
    max_shift: search window in pixels (±max_shift)
    alpha: blend weight toward the aligned roll (0.1–0.35 works well)
    """
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype
    # luminance for correlation (cheap & stable)
    Y = (0.299*img[:,0] + 0.587*img[:,1] + 0.114*img[:,2]).to(dtype)

    best_score, best_shift = None, (0, 0)
    # small search grid; keep it cheap
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dy == 0 and dx == 0:
                continue
            Ys = torch.roll(Y, shifts=(dy, dx), dims=(1, 2))
            # normalized correlation
            num = (Y * Ys).mean()
            den = (Y.pow(2).mean().sqrt() * Ys.pow(2).mean().sqrt() + 1e-8)
            score = (num / den).item()
            if (best_score is None) or (score > best_score):
                best_score, best_shift = score, (dy, dx)

    # only snap if there’s a clear positive correlation
    if best_score is not None and best_score > 0.02:
        dy, dx = best_shift
        rolled = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
        img.add_((rolled - img) * alpha)  # in-place

# --------------------
# - DUMP & LOAD JSON -
# --------------------

def _safe_args_dict(ns):
    exclude = {
        "output_folder", "use_arch", "use_model", "model_name",
        "auto_tfeat", "load_json"
    }
    out = {}
    for k, v in vars(ns).items():
        if k in exclude:
            continue
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = f"<non-serializable:{type(v).__name__}>"
    return out

def _move_probe_for_layer(steps_folder: str, layer_idx: int, dest_run_dir: str):
    try:
        src = os.path.join(steps_folder, f"__PROBE_L{layer_idx}")
        if os.path.isdir(src):
            dst = os.path.join(dest_run_dir, f"__PROBE_L{layer_idx}")
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(Fore.CYAN + f"[probe] moved {src} → {dst}" + Fore.RESET)
    except Exception as e:
        print(Fore.YELLOW + f"[warn] could not move probe folder: {e}" + Fore.RESET)

def _write_args_json_for_image(png_path: str, ns):
    try:
        meta = _safe_args_dict(ns)
        meta["__output_image"] = os.path.basename(png_path)
        jpath = png_path.rsplit(".", 1)[0] + ".args.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    except Exception as e:
        print(Fore.YELLOW + f"[warn] could not write args json for '{png_path}': {e}" + Fore.RESET)

def _override_args_from_json(ns, json_path: str):
    if not os.path.isfile(json_path):
        raise SystemExit(Fore.RED + f"[error] --load_json not found: {json_path}" + Fore.RESET)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(Fore.RED + f"[error] --load_json must be a JSON object: {json_path}" + Fore.RESET)

    for k, v in data.items():
        if k == "load_json":
            continue
        if hasattr(ns, k):
            setattr(ns, k, v)
        else:
            print(Fore.YELLOW + f"[warn] --load_json key ignored (unknown arg): {k}" + Fore.RESET)
    print(Fore.CYAN + Style.BRIGHT + f"[cfg] Loaded and applied overrides from JSON: {json_path}" + Fore.RESET)
    return ns

# ----------------------
# ---    HELPERS    ---
# ----------------------

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

def _parse_csv_ints(s: str):
    s = s.strip()
    return [int(x) for x in s.split(",")] if s else []

def _parse_csv_floats(s: str):
    s = s.strip()
    return [float(x) for x in s.split(",")] if s else []

def _parse_exclude_list(s: str):
    s = s.strip()
    if not s:
        return []
    parts = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a,b = token.split("-",1)
            a,b = int(a), int(b)
            parts.extend(list(range(min(a,b), max(a,b)+1)))
        else:
            parts.append(int(token))
    return sorted(set(parts))
    
def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    an = a / (a.norm(dim=-1, keepdim=True) + eps)
    bn = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (an * bn).sum(dim=-1)

def _rgb_to_yuv(img):
    R, G, B = img[:,0:1], img[:,1:2], img[:,2:3]
    Y = 0.299*R + 0.587*G + 0.114*B
    U = -0.14713*R - 0.28886*G + 0.436*B
    V =  0.615*R - 0.51499*G - 0.10001*B
    return Y, U, V

def _yuv_to_rgb(Y, U, V):
    R = Y + 1.13983*V
    G = Y - 0.39465*U - 0.58060*V
    B = Y + 2.03211*U
    return torch.cat([R, G, B], dim=1)

def _sha1_of_file(path: str, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b: break
            h.update(b)
    return h.hexdigest()
    
def parse_range(range_str):
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return list(map(int, range_str.split(',')))

def clamp_range(vals, minval, maxval):
    # vals: list[int]
    return [max(min(val, maxval), minval) for val in vals]

def _effective_query_mode(args_local):
    has_text = (isinstance(args_local.prompt, str) and args_local.prompt.strip()) \
               or (getattr(args_local, "auto_tfeat", None) is not None)
    return args_local.query_mode if args_local.query_mode else ("cls" if has_text else "mean")

def _probe_cache_key(args_local, layer_idx, clipmodel, patch_size):
    key = {
        "arch": args_local.use_arch,
        "model": clipmodel,
        "layer": int(layer_idx),
        "auto_head_size": int(args_local.auto_head_size),
        "auto_head_probe_steps": int(args_local.auto_head_probe_steps),
        "head_agg": str(args_local.head_agg),
        "head_topk_patches": int(args_local.head_topk_patches),
        "query_mode": _effective_query_mode(args_local),
        "prompt": str(args_local.prompt or ""),
        "auto_prompt_sha": getattr(args_local, "auto_prompt_sha", None),
        "text_coeff_probe": float(args_local.text_coeff_probe),
        "probe_broad_prompts": str(getattr(args_local, "probe_broad_prompts", "")),
        "probe_neg_prompts":   str(getattr(args_local, "probe_neg_prompts",   "")),
        "sophia_k": int(args_local.sophia_k),
        "sophia_mode": str(args_local.sophia_mode),
        "sophia_h_subsample": float(args_local.sophia_h_subsample),
        "sophia_gamma": float(args_local.sophia_gamma),
        "sophia_tau": float(args_local.sophia_tau),
        "sophia_abs_hessian": bool(args_local.sophia_abs_hessian),
        "sophia_eps": float(args_local.sophia_eps),
        "sophia_u_lowpass": bool(getattr(args_local, "sophia_u_lowpass", True)),
        "sophia_u_lowpass_cutoff": float(getattr(args_local, "sophia_u_lowpass_cutoff", 0.30)),
        "sophia_num_hutch": int(getattr(args_local, "sophia_num_hutch", 4)),
        "sophia_u_dist": str(getattr(args_local, "sophia_u_dist", "rademacher")),
        "deterministic": bool(args_local.deterministic),
    }
    return key

def _probe_cache_hash(key: dict) -> str:
    s = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    
def _recompute_probe_scores(raw_results: List[dict], args_local):
    if not raw_results:
        return []
    # mins/maxes for normalization
    tmin, tmax = min(r["target_score"] for r in raw_results), max(r["target_score"] for r in raw_results)
    imin, imax = min(r["itc"] for r in raw_results),           max(r["itc"] for r in raw_results)
    smin, smax = min(r["spec_sparsity"] for r in raw_results), max(r["spec_sparsity"] for r in raw_results)
    kmin, kmax = min(r["spec_topk"]     for r in raw_results), max(r["spec_topk"]     for r in raw_results)
    mmin, mmax = min(r["spec_margin"]   for r in raw_results), max(r["spec_margin"]   for r in raw_results)
    gmin, gmax = min(r.get("glob", 0.0) for r in raw_results), max(r.get("glob", 0.0) for r in raw_results)

    def z(v, vmin, vmax): return 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin + 1e-8)

    beta               = float(args_local.auto_head_beta)
    use_head10_prior   = bool(getattr(args_local, "use_head10_prior", False))
    head10_prior_w     = float(getattr(args_local, "head10_prior_weight", 0.25))
    globalness_weight  = float(getattr(args_local, "globalness_weight",   0.25))
    spec_w_sparsity    = float(getattr(args_local, "spec_weight_sparsity", 0.5))
    spec_w_topk        = float(getattr(args_local, "spec_weight_topk",     0.3))
    spec_w_margin      = float(getattr(args_local, "spec_weight_margin",   0.2))

    out = []
    for r in raw_results:
        base = (1 - beta) * z(r["target_score"], tmin, tmax) + beta * z(r["itc"], imin, imax)
        score_adj = base
        if use_head10_prior:
            if "sim_h10" in r:
                score_adj -= head10_prior_w * max(0.0, r["sim_h10"])
            score_adj -= globalness_weight * z(r.get("glob", 0.0), gmin, gmax)

        s_part = spec_w_sparsity * z(r["spec_sparsity"], smin, smax)
        k_part = spec_w_topk     * z(r["spec_topk"],     kmin, kmax)
        m_part = spec_w_margin   * z(r["spec_margin"],   mmin, mmax)

        o = dict(r)
        o["score"]       = float(base)
        o["score_adj"]   = float(score_adj)
        o["score_final"] = float(score_adj + s_part + k_part + m_part)
        out.append(o)
    return out

def _print_probe_leaderboard(layer_idx: int, results_sorted: List[dict]):  # NEW
    print(Fore.CYAN + f"\n[probe-final] L{layer_idx} ranking (by score_final):" + Fore.RESET)
    for i, r in enumerate(results_sorted):
        line = (f"  #{i+1:02d} H{r['head']:02d}  score_final={r['score_final']:.3f}  "
                f"score_adj={r['score_adj']:.3f}  score={r['score']:.3f}  "
                f"spars={r['spec_sparsity']:.3f}  margin={r['spec_margin']:.3f}  glob={r.get('glob',0.0):.3f}")
        if "sim_h10" in r: line += f"  sim_h10={r['sim_h10']:.3f}"
        if "probe_png" in r: line += f"  [{r['probe_png']}]"
        print(line, flush=True)
        
# ----------- Weird notes ---------------       
#from cliptools import GameOfLifeBaseColorVariationLowFreq # weird math glitch fun / failed CV loss glitch
#from cliptools import GameOfLifeBaseColorVariationLowFreq as BaseColorVariationLowFreq # to try it