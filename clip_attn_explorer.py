import os, math, argparse, re
from PIL import Image
import kornia
from colorama import Fore, Style
from typing import Optional, List
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file
import json, shutil
import contextlib, collections

# Custom imports: CLIP with exposed QKV + attn weights
import attnclip as clip
from attnclip.model import convert_state_dict_inproj_to_qkv
from attnclip.model import QuickGELU
# Custom imports: SophiaViz Second Order (Hessian) optimizer
from sophiaviz_opt.sophia import SophiaViz

# Custom imports: cliptools
from cliptools import LossArraySophizViz as LossArray, ClipViTWrapper as ClipWrapper
from cliptools import ColorJitterGPT5 as ColorJitter, TileGPT as Tile
from cliptools import Clip, Jitter, RepeatBatch, CLIPAugCosineQueue
from cliptools import save_image, GaussianNoise, new_init
from cliptools import ChannelCorrelationLoss, TotalVariation
from cliptools import MeanStdLoss, ColorVariationLowFreq, FrequencySlopePenalty
from cliptools import TruePatchCorrelationLossGPT5 as TruePatchCorrelationLoss
from cliptools import fix_random_seed

# Custom imports: clipattntools
from clipattntools import _parse_csv_ints, _parse_csv_floats, _probe_cache_key, get_clip_dimensions
from clipattntools import _rgb_to_yuv, _yuv_to_rgb, _sha1_of_file, parse_range, clamp_range
from clipattntools import _probe_cache_hash, _recompute_probe_scores, _print_probe_leaderboard
from clipattntools import _combine_ga_batches, Normalization
from clipattntools import Pars, generate_target_text_embeddings
from clipattntools import FeatureScalerHook, MLPActHook, AllHeadsCaptureHook
from clipattntools import _natural_key, _build_video_from_frames
from clipattntools import _deepdream_init_from_image, attncopy
from clipattntools import _subpixel_jitter_, _edge_aware_chroma_smooth_
from clipattntools import _temporary_pos_embed, _resize_positional_embedding
from clipattntools import _encode_prompt_list, _encode_clip_feat, _encode_prompt_emb
from clipattntools import _safe_args_dict, _move_probe_for_layer, _write_args_json_for_image, _override_args_from_json

# Suppress warnings spam from torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP Attention Head Activation Max Visualization')
    # ┌─────────────────────┐
    # │   GENERAL SETTINGS  │
    # └─────────────────────┘
    # --- Defaults (scroll to very bottom -> how to disable) ---
    parser.set_defaults(sophia_abs_hessian=True)
    parser.set_defaults(mlp_saliency_use_grad=True)
    parser.set_defaults(print_head_probe=True)
    parser.set_defaults(leaf_ops=True)
    parser.set_defaults(exclude_head10=True)
    # --- General Settings ---
    parser.add_argument('--use_arch', default='ViT-L/14', help="CLIP Model Architecture (for state_dict / .safetensors loading)")
    parser.add_argument('--use_model', default="ViT-L/14", help="Name or path to CLIP Model; pickle .pt or .safetensors")
    parser.add_argument('--head_range', default="0-15", type=str, help="Which heads to visualize (if manual selection; --auto_head* overrides); e.g. '1,4,6'.")
    parser.add_argument('--layer_range', default="2", type=str, help="Which ONE layer to visualize. Note: Currently no multi-layer visualization support!")
    parser.add_argument("--fast", action='store_true', help="Set matmul precision to 'medium' (NVIDIA Ampere+ only)")
    parser.add_argument("--deterministic", action='store_true', help="Deterministic backends, fixed random seed; reproducible behavior on GPU when used with --force_reprobe.")
    parser.add_argument("--force_reprobe", action='store_true', help="Force re-computing the head scan / text ascent, even if cache exists.")
    # --- Output Settings ---
    parser.add_argument("--output_folder", default='results', help="Folder to save output to.")
    parser.add_argument("--save_intermediate", action='store_true', help="Also save intermediate images (every 10 steps)")    
    parser.add_argument("--save_video", action='store_true', help="Save animation from --save_intermediate steps; requires ffmpeg to be available")
    parser.add_argument("--fps_video", default=24, type=int, help="FPS for video.")
    parser.add_argument("--model_name", default='ViTL14', help="Name for prepending to filename (e.g. for tagging fine-tunes)")
    # --- Quality Settings ---
    parser.add_argument('--steps', default=1600, type=int, help="Number of image optimization steps; default: 1600")
    parser.add_argument('--lr', default=1.0, type=float, help="Learning Rate; default: 1.0")
    parser.add_argument("--lr_warmup_frac", default=0.10, type=float, help="Fraction of total steps used for linear LR warmup before cosine decay.")
    parser.add_argument("--lr_min_mult",  default=0.05, type=float, help="Final LR multiplier at end of cosine schedule (min_lr = base_lr * lr_min_mult).")
    parser.add_argument("--grad_rms_norm", action="store_true", help="Use RMS instead of L2 unit-norm")
    parser.add_argument("--repeat_batch", default=1, type=int, help="Repeat batches for jitter/colors; increases quality and blows up VRAM use with octave upscale.")
    parser.add_argument("--fake_n", default=4, type=int, help="Fakebatch batch size n; enable with --fakebatch")
    parser.add_argument("--fakebatch", action="store_true", help="Enable Fake (low res) repeat batch 224px; may improve quality at the expense of compute, lower VRAM overhead")
    # --- Octave (embeddings upscale) controls for Vision Transformer ---
    parser.add_argument("--octaves", default="112,168,224,336,448", type=str, help="Comma-separated target sizes (embeds interpolation); remove 448 if CUDA OOM")
    parser.add_argument("--octave_step_policy", default="equal", type=str, choices=["equal","patch_equiv"], help="How to split steps across octaves")
    parser.add_argument("--octave_step_weights", default="", type=str, help="Manual weights; empty => policy")
    parser.add_argument("--octave_tv_scales", default="0.5,0.7,1.2,1.8,1.3", type=str, help="TV multipliers per octave")
    parser.add_argument("--octave_lr_scales", default="1.1,1.0,1.3,1.0,0.8", type=str, help="LR multipliers per octave")
    #
    # ┌─────────────────────────────┐
    # │   VISUALIZATION SETTINGS    │
    # └─────────────────────────────┘
    # [IMAGE INPUT]        --- ViT: MLP saliency from a real image ---
    parser.add_argument("--mlp_saliency_img", type=str, default="", help="Path to image; compute MLP-channel saliency at the target layer")
    parser.add_argument("--mlp_topk", type=int, default=8, help="How many MLP channels (neuron idx) to use if saliency image is provided")
    parser.add_argument("--mlp_alpha", type=float, default=0.5, help="Weight of MLP-channel (--mlp_topk) objective term")
    # [DEEPDREAM]           --- ViT: Gimmick ~ DeepDream from real image :-) ---
    parser.add_argument("--deepdream", action="store_true", help="Start optimization from --mlp_saliency_img instead of random (Gaussian noise) init.")
    parser.add_argument("--overlay_alpha", type=float, default=0.8, help="Deepdream: Per-step blend toward --mlp_saliency_img.")
    parser.add_argument("--overlay_frac", type=float, default=0.40, help="Deepdream: Apply overlay for initial x percent of octave (default: 40%)")
    # [TEXT PROMPT]         --- TxT: Text Encoder prompt guidance ---
    parser.add_argument("--prompt", type=str, default="", help="Text prompt(s) to align with; separate with '|', e.g. 'a cat|a kitty|a feline'. Uses text-image cos-sim (ITC) and PromptAlignmentLoss.")
    parser.add_argument("--auto_prompt", action="store_true", help="For an image --mlp_saliency_img, auto-optimize a prompt embedding; replaces --prompt (and is more accurate than human prompt).")
    parser.add_argument("--min_cos_sim", action='store_true', help="For --auto_prompt: Minimize Cosine Similarity of Text with Image, instead of Max Cos Sim (weird ill-defined optimization goal 'away from'!)")
    parser.add_argument('--ga_batch_size', default=10, type=int, help="For --auto_prompt: Batch size for gradient ascent on the text embeddings (CLIP's 'opinion' self-prompt embedding about image)")
    parser.add_argument("--text_coeff", type=float, default=2.5, help="Factor: Strength of PromptAlignmentLoss for --auto_prompt or --prompt. SET 0 TO DISABLE. Negative (min cos sim) may cause weird results.")
    parser.add_argument("--text_coeff_probe", type=float, default=1.5, help="Text Coefficient for Head Scan Probe; only used for head selection (not visualization); set higher for more selective pressure.")
    parser.add_argument("--probe_broad_prompts", type=str, default="", help="List of alternative (positive) labels; e.g. 'a cat|a kitty|a feline', to compute spec_margin. Affects head ranking.")
    parser.add_argument("--probe_neg_prompts",   type=str, default="", help="List of negatives, e.g. 'text|letters|logo' or 'a dog|a canine|a wolf'. Affects head ranking (indirect effect on visualization).")

    # [AUTO-HEAD]           --- ViT: Auto Attention Head Selection / Probe Scan ---
    parser.add_argument("--auto_head", action="store_true", help="Auto-select the single best head at the target layer using probe scoring.")
    parser.add_argument("--auto_head_multi", action="store_true", help="Select multiple top heads instead of one (see --auto_head_multi_n and --auto_head_min_frac).")
    parser.add_argument("--auto_head_multi_n", type=int, default=3, help="Number of heads to keep when --auto_head_multi is set.")    
    # [AUTO-HEAD-FACTORS]   --- ViT: Auto Attention Head Selection Factors ---
    parser.add_argument("--spec_weight_margin",   type=float, default=0.2, help="Weight for specificity via margin vs broad/negative prompts in final probe score")
    parser.add_argument("--auto_head_beta", type=float, default=0.7, help="Blend between pooled target_score and image–text ITC in probe scoring (0=target_score only, 1=ITC only).")
    parser.add_argument("--neg_alpha", type=float, default=0.25, help="Weight for explicit negative-head/activity penalty ---- TODO.")
    parser.add_argument("--head_weights", type=str, default="auto", help="Per-head weights for joint target (comma-separated) OR 'auto' to weight by probe scores.")
    parser.add_argument("--auto_head_min_frac", type=float, default=0.6, help="Keep heads with score_final ≥ this fraction of the best when --auto_head_multi is set.")
    parser.add_argument("--auto_head_probe_steps", type=int, default=20, help="Steps per head during the probing phase.")
    parser.add_argument("--auto_head_size", type=int, default=168, help="Probe image size (pixels) for head probing; must be a multiple of the patch size.")

    # [ATTN QKV, MAP]       --- ViT: Attention Queries and Keys, Map ---
    parser.add_argument("--attn_prior", type=str, default="head10", choices=["none","center","ring","diag","head10"], help="OPTIMIZATION: Spatial prior used as KL target for attention over KEYS.")
    parser.add_argument("--use_head10_prior", action="store_true", help="HEAD PROBE: Penalize heads similar to Head 10 during ranking (Head 10 is the noisy global info router in ViT-L/14)")
    parser.add_argument("--head_agg", type=str, default="mean", choices=["mean","topk","center","attn_weighted"], help="HEAD PROBE: How to pool token scores -> head score.")
    parser.add_argument("--query_mode", type=str, default="", choices=["","cls","mean","topk"], help="PROBE/OPTIM: Used for Aw (attn over KEYS): QUERIES to read attn from. Empty=Auto; CLS if prompt, else mean.")
    parser.add_argument("--attn_entropy_target", type=str, default="high", choices=["low","high"], help="Entropy 'low' encourages peaky attention; 'high' encourages diffuse attention.")
    # [ATTN FACTORS]        --- ViT: Attention factors (IMPORTANT for above!) ---
    parser.add_argument("--head_topk_patches", type=int, default=0, help="If head_agg='topk' or query_mode='topk', number of patches to pool; 0 ⇒ auto (≈P/32).")
    parser.add_argument("--comp_k_heads", type=int, default=0, help="Use top-K strongest competitor heads; 0 ⇒ average over all other heads.")    
    parser.add_argument("--comp_alpha_heads", type=float, default=1.0, help="Weight of competitor-head penalty (encourages target head > other heads).")
    parser.add_argument("--attn_prior_lambda", type=float, default=0.02, help="Weight of KL(attn || prior) term when --attn_prior is not 'none'")
    parser.add_argument("--attn_entropy_lambda", type=float, default=0.03, help="Strength of entropy regularizer on attention map (sign/target set by --attn_entropy_target)")
    parser.add_argument("--head10_prior_weight", type=float, default=0.25, help="Penalty weight for similarity to head-10 in probe scoring (applied when --use_head10_prior)")
    parser.add_argument("--globalness_weight",   type=float, default=0.25, help="Penalty weight for globalness (high entropy) in probe scoring (applied when --use_head10_prior)")
    parser.add_argument("--spec_weight_sparsity", type=float, default=0.5, help="Weight for specificity via 1−globalness (sparsity) term in final probe score")
    parser.add_argument("--spec_weight_topk",     type=float, default=0.3, help="Weight for specificity via top-k mass term in final probe score")

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │   Good News: You might not need [below]. Defaults are usually good!  │
    # └──────────────────────────────────────────────────────────────────────┘
    # [ABLATE REG NEURONS]  --- ViT: Ablate Register Neurons (set activation value to 0) ---
    parser.add_argument("--ablate_registers", action="store_true", help="Ablate the 13 emergent Register Neurons (global information superhighway) in Layer 11+12 that feed Head 10")
    # [ATTN TRANSPLANT]     --- ViT: Transplant Attention from one layer to another (overwrite target attention) ---
    parser.add_argument('--attn_from', default="22", type=str, help="Source of Attention block to transplant (remains unmodified); e.g. '4-6', '13,19,21'")
    parser.add_argument('--attn_to', default="2", type=str, help="Target for Attention (will be replaced); number of targets must match number of sources")
    parser.add_argument("--attn_move_late", action="store_true", help="Enable attention transplantation AFTER head query, before visualization") # Choose one, NOT both!
    parser.add_argument("--attn_move_init", action="store_true", help="Enable attention transplantation BEFORE any computations are carried out") # Choose one, NOT both!
    # [REGULARIZATION LOSS] --- Image Regularization: Auxiliary Losses (Regularization Loss) ---
    parser.add_argument('--coeff', default=0.0001, type=float, help="Total Variation Loss factor for tv*coeff. Try 0.001 for smoother (but potentially less accurate) image.")
    parser.add_argument("--mean_std", action="store_true", help="Enable Mean/Std Correction Loss")
    parser.add_argument('--std_coeff', default=0.2, type=float, help="Mean/Std Correction Loss Loss factor")
    parser.add_argument('--chan', default=0.1, type=float, help="Color Channel Correlation Penalty Loss factor")
    parser.add_argument("--chan_corr", action="store_true", help="Enable Color Channel Correlation Penalty Loss")
    parser.add_argument('--col', default=0.3, type=float, help="ColorVariationLowFreq Penalty Loss factor")
    parser.add_argument("--col_var", action="store_true", help="Enable ColorVariationLowFreq Penalty Loss")
    parser.add_argument('--freq', default=0.2, type=float, help="Frequency Slope (Color) Penalty Loss factor")
    parser.add_argument("--freq_var", action="store_true", help="Enable Frequency Slope Penalty Loss")
    parser.add_argument('--decorr', default=0.1, type=float, help="Patch Correlation Penalty Loss factor")
    parser.add_argument("--dec_pen", action="store_true", help="Enable Patch Correlation Penalty Loss")
    # [REGULARIZATION]      --- Leaf Tensor Ops (Regularization, injected AFTER optimization) ---
    parser.add_argument("--leaf_every", type=int, default=20, help="Inject Leaf Ops every n Steps")
    parser.add_argument('--leaf_gauss', default=0.0005, type=float, help="Gaussian Noise factor")
    parser.add_argument('--leaf_col_amp', default=0.03, type=float, help="Color Jitter Amplitude factor")
    parser.add_argument('--leaf_col_std', default=0.04, type=float, help="Color Jitter Scale factor")
    parser.add_argument('--leaf_chroma', default=1.0, type=float, help="Strength factor for Chroma (color) smoothing")
    parser.add_argument('--leaf_chroma_edge', default=12, type=int, help="Smooth color: Edge Kernel size (increase to preserve edges more)")
    parser.add_argument('--leaf_chroma_col', default=3, type=int, help="Smooth color: Kernel size (increase for smoother, e.g. 4-6)")
    # [OPTIMIZER SETTINGS]  --- SophiaViz Optimizer ---
    parser.add_argument("--sophia_k", type=int, default=10, help="Hessian refresh period (every n steps). Smaller = fresher curvature but noisier; larger = smoother but staler")
    parser.add_argument("--sophia_mode", type=str, default="hutchinson", choices=["hutchinson","grad_sqr"], help="How to estimate diag curvature: Hutchinson (stochastic u⊙Hu) or grad_sqr (g²)")
    parser.add_argument("--sophia_h_subsample", type=float, default=0.25, help="Bernoulli subsample probability for Hutchinson vectors")
    parser.add_argument("--sophia_gamma", type=float, default=0.012, help="Per-coordinate clipping factor; stabilizes noisy curvature estimates")
    parser.add_argument("--sophia_tau", type=float, default=0.00001, help="Curvature floor τ; prevents division by very small Hessian entries")
    parser.add_argument("--sophia_u_lowpass", action="store_true", help="Enable low-pass for filter Hutchinson vectors u")
    parser.add_argument("--sophia_u_lowpass_cutoff", type=float, default=0.30, help="Normalized cutoff for u low-pass (0–0.5-ish). Default: 0.30")
    parser.add_argument("--sophia_num_hutch", type=int, default=1, help="Number of iid Hutchinson probes to average per refresh.")
    parser.add_argument("--sophia_u_dist", type=str, default="rademacher", choices=["gaussian", "rademacher"], help="Distribution for Hutchinson vectors u. Default: rademacher")
    parser.add_argument("--sophia_eps", type=float, default=1e-12, help="Numerical floor for divisions; avoids blow-ups on near-zero curvature.")

    # [OVERRIDE DEFAULTS]   --- Negations (disable known-good defaults) ---
    parser.add_argument("--no_exclude_head10", dest="exclude_head10", action="store_false", help="Include Head 10 in auto-head selection. Noisy, always wins after register neurons emergence in Layers 11+12.")
    parser.add_argument("--sophia_no_abs_hessian", dest="sophia_abs_hessian", action="store_false", help="Disable |u⊙Hu| PSD surrogate (absolute Hessian). Includes sign-flips in Hutchinson; noisy.")
    parser.add_argument("--mlp_saliency_no_use_grad", dest="mlp_saliency_use_grad", action="store_false", help="Disable grad*act for MLP-channel saliency; less targeted (uses plain activations, ignores CLS from text).")
    parser.add_argument("--no_print_head_probe", dest="print_head_probe", action="store_false", help="Disable intermediate auto head probe metrics print (marginal overhead (fwd pass) reduction)")
    parser.add_argument("--no_leaf_ops", dest="leaf_ops", action="store_false", help="Disable Leaf Tensor (post) augmentation / image regularization.")

    parser.add_argument("--load_json", type=str, default="", help="Path to .json; load/override all CLI args EXCEPT --output_folder --use_arch --use_model --model_name (e.g. to test fine-tune vs. pre-trained).")
    
    return parser.parse_args()

args = parse_arguments()
steps_folder = args.output_folder
os.makedirs(steps_folder, exist_ok=True)
repeats = args.repeat_batch
iterations = args.steps
clipmodel = args.use_model
clipname = args.model_name

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.deterministic:
    fix_random_seed()
    if not args.force_reprobe:
        print(Fore.RED + Style.BRIGHT + "\n")
        print("┌──────────────────────────────────────────────────┐")
        print("│                !!!! WARNING !!!!                 │")
        print("│ Determinism not possible if using cache.         │")
        print("│ Always set --force_reprobe for true determinism! │")
        print("└──────────────────────────────────────────────────┘\n" + Fore.RESET)
            
if args.fast:
    torch.set_float32_matmul_precision("medium")

if getattr(args, "load_json", "") and args.load_json.strip():
    args = _override_args_from_json(args, args.load_json.strip())

def load_clip_model(device: str = 'cuda') -> torch.nn.Module:
    if clipmodel.endswith(".safetensors"):
        print(f"Detected .safetensors file. Loading {args.use_arch} and applying file as state_dict...")
        model, preprocess = clip.load(args.use_arch, device=device, jit=False)
        state_dict = load_file(clipmodel)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            msg = str(e)
            if ("Missing key(s) in state_dict" in msg or "Unexpected key(s) in state_dict" in msg):
                print("State dict format mismatch, attempting QKV conversion...")
                state_dict = convert_state_dict_inproj_to_qkv(state_dict)
                model.load_state_dict(state_dict)
                print("OK!")
            else:
                raise
    else:
        print("Detected non-.safetensors file or name. Attempting to load model...")
        model, preprocess = clip.load(clipmodel, device=device, jit=False)

    premodel = model
    model = ClipWrapper(model).to(device).float()

    base_pos_embed = premodel.visual.positional_embedding.detach().clone()
    patch_size = getattr(premodel.visual, 'patch_size', None)
    if patch_size is None and hasattr(premodel.visual, 'conv1'):
        patch_size = premodel.visual.conv1.kernel_size[0]
    return model, premodel, preprocess, base_pos_embed, patch_size

def _compute_mlp_topk_from_image(premodel, preprocess, layer_idx: int, image_path: str,
                                 prompt: str, topk: int, use_grad: bool, device: str = device):
    block = premodel.visual.transformer.resblocks[layer_idx]
    # set up hook that also captures grads if needed
    captured = {"out": None, "grad": None}

    def fwd_hook(module, inp, out):
        captured["out"] = out
        if use_grad and prompt and out.requires_grad:
            out.retain_grad()

    def bwd_hook(module, grad_in, grad_out):
        captured["grad"] = grad_out[0]

    gelu = None
    for m in block.mlp.modules():
        if isinstance(m, QuickGELU):
            gelu = m
            break
    if gelu is None:
        raise RuntimeError("QuickGELU not found in block.mlp")

    h1 = gelu.register_forward_hook(fwd_hook)
    h2 = gelu.register_full_backward_hook(bwd_hook) if use_grad and prompt else None

    with Image.open(image_path).convert("RGB") as im:
        x = preprocess(im).unsqueeze(0).to(device)

    premodel = premodel.to(device)
    premodel.eval()

    # forward (and backward if needed)
    with torch.set_grad_enabled(bool(use_grad and prompt)):
        img_feat = premodel.encode_image(x)  # triggers hook
        if use_grad and prompt:
            tokens = clip.tokenize([prompt]).to(device)
            txt = premodel.encode_text(tokens).float()
            txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
            img_feat = img_feat.float()
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
            sim = (img_feat * txt).sum()
            sim.backward()

    # aggregate saliency
    acts = captured["out"]
    if acts is None:
        h1.remove(); 
        if h2: h2.remove()
        raise RuntimeError("MLP saliency capture failed (no activations).")

    # unify to [..., hidden] with hidden=4*width
    hidden = acts.shape[-1]
    if acts.dim() == 3:
        reduce_axes = tuple(range(acts.dim()-1))  # mean over all but last
    else:
        raise RuntimeError(f"Unexpected MLP activation ndim={acts.dim()}")

    if use_grad and prompt and captured["grad"] is not None:
        sal = (acts * captured["grad"]).abs().mean(dim=reduce_axes)  # grad*act
    else:
        sal = acts.abs().mean(dim=reduce_axes)  # plain activation strength

    k = min(int(topk), int(hidden))
    top_vals, top_idx = torch.topk(sal, k=k, dim=-1)
    top_idx = top_idx.detach().cpu().tolist()

    h1.remove()
    if h2: h2.remove()

    return top_idx  # list[int]

def _pooled_head_score(acts_all, target_head, head_agg, head_topk_patches, query_mode, block):
    # acts_all: [B, Seq, H, Dh]
    with torch.no_grad():
        B, Seq, Hh, Dh = acts_all.shape
        target = acts_all[:, :, target_head, :].norm(dim=-1)  # [B, Seq]

        # keys grid handling (drop CLS on keys if grid-like)
        g_sq = int(round((Seq - 1) ** 0.5))
        dropped_cls = (g_sq * g_sq + 1 == Seq)
        if dropped_cls:
            target_ = target[:, 1:]    # [B, P]
        else:
            target_ = target           # [B, Seq]

        P = target_.shape[1]
        Aw = None
        if block is not None and getattr(block, "attn_probs", None) is not None:
            probs_flat = block.attn_probs#.detach()  # [B*H, Q, K]
            Q, K = probs_flat.shape[-2], probs_flat.shape[-1]
            attn_probs = probs_flat.view(B, Hh, Q, K)
            if query_mode == "cls" and Q >= 1:
                Aw = attn_probs[:, target_head, 0, :]
            elif query_mode == "mean":
                if Q == Seq and dropped_cls:
                    Aw = attn_probs[:, target_head, 1:, :].mean(dim=1)
                else:
                    Aw = attn_probs[:, target_head].mean(dim=1)
            elif query_mode == "topk":
                kq = head_topk_patches if head_topk_patches > 0 else max(1, P // 32)
                _, top_idx = torch.topk(target_, k=kq, dim=1)
                q_idx = top_idx + 1 if (Q == Seq and dropped_cls) else top_idx
                gather_idx = q_idx.unsqueeze(-1).expand(-1, -1, K)
                Ah = attn_probs[:, target_head]              # [B,Q,K]
                Aw = torch.gather(Ah, dim=1, index=gather_idx).mean(dim=1)
            if Aw is not None and Aw.size(1) == Seq and dropped_cls:
                Aw = Aw[:, 1:]
            if Aw is not None:
                Aw = Aw / (Aw.sum(dim=-1, keepdim=True) + 1e-8)

        if head_agg == "mean":
            return target_.mean().item()
        elif head_agg == "topk":
            k = head_topk_patches if head_topk_patches > 0 else max(1, P // 32)
            vals, _ = torch.topk(target_, k=k, dim=1)
            return vals.mean().item()
        elif head_agg == "center":
            g = int(round(P ** 0.5))
            if g * g != P:  # fallback
                return target_.mean().item()
            yy, xx = torch.meshgrid(torch.linspace(-1,1,g,device=target_.device),
                                    torch.linspace(-1,1,g,device=target_.device), indexing="ij")
            m = torch.exp(-(xx**2 + yy**2)/(2*(0.6**2))); m = (m/m.sum()).reshape(1, P)
            return (target_ * m).sum(dim=1).mean().item()
        elif head_agg == "attn_weighted" and Aw is not None:
            return (target_ * Aw).sum(dim=1).mean().item()
        return target_.mean().item()

class HeadProbeVisualizer:
    def __init__(self, model, loss_array: LossArray, target_layer, target_head,
                all_heads_hook, pre_aug=None, post_aug=None,
                steps: int = 100, lr: float = 0.1,
                save_every: int = 50, saver: bool = True, print_every: int = 25,
                head_agg: str = "mean", head_topk_patches: int = 0,
                query_mode: str | None = None, prompt_active: bool = False,
                sophia_k: int = 10, sophia_mode: str = "hutchinson", sophia_h_subsample: float = 0.25,
                sophia_gamma: float = 0.01, sophia_tau: float = 0.0, sophia_abs_hessian: bool = True, sophia_eps: float = 1e-12,
                u_lowpass: bool = True, u_lowpass_cutoff: float = 0.30, num_hutch: int = 4, u_dist: str = "rademacher",
                progress_hook=None):
                
        self.model = model
        self.loss = loss_array
        self.target_layer = target_layer
        self.target_head = target_head
        self.all_heads_hook = all_heads_hook
        self.pre_aug = pre_aug
        self.post_aug = post_aug
        self.steps = int(steps)
        self.lr = float(lr)
        self.save_every = int(save_every)
        self.saver = bool(saver)
        self.print_every = int(print_every)
        self.head_agg = head_agg
        self.head_topk_patches = head_topk_patches
        self.query_mode = (query_mode or ("cls" if prompt_active else "mean"))

        self.sophia_cfg = dict(
            k=sophia_k,
            hessian_mode=sophia_mode,
            h_subsample=sophia_h_subsample,
            gamma=sophia_gamma,
            tau=sophia_tau,
            abs_hessian=sophia_abs_hessian,
            eps=sophia_eps,
            u_lowpass=bool(u_lowpass),
            u_lowpass_cutoff=float(u_lowpass_cutoff),
            num_hutch=int(num_hutch),
            u_dist=str(u_dist),
        )

        self.progress_hook = progress_hook

    def _simple_target_score(self, acts_all):
        target = acts_all[:, :, self.target_head, :].norm(dim=-1)  # [B, Seq]
        return target.mean()

    def _objective(self, img: torch.Tensor):
        augmented = self.pre_aug(img) if self.pre_aug is not None else img
        _ = self.model(augmented)  # triggers hooks
        acts_all = self.all_heads_hook.activations
        if acts_all is None:
            raise RuntimeError("No activations from all heads hook (probe).")

        # comp_alpha_heads = 0.0 during probe (no competitor penalty)
        head_score = self._simple_target_score(acts_all)
        tv = self.loss(augmented, track_stats_override=True)

        # maximize head_score -> minimize negative
        return -(head_score) + tv

    def __call__(self, img: torch.Tensor, layer: int, clipname: str):
        if not img.is_cuda or img.device != device:
            img = img.to(device)
        if not img.requires_grad:
            img.requires_grad_()

        opt = SophiaViz([img], lr=self.lr, betas=(0.96, 0.99), weight_decay=0.0,
                        capturable=False, maximize=False, **self.sophia_cfg)

        print(Fore.MAGENTA + Style.BRIGHT + f"\nRunning fast head scan: HEAD {self.target_head}" + Fore.RESET)
        print(f'#i\tLoss\t[probe@L{layer}H{self.target_head}]', flush=True)
        for i in range(self.steps + 1):
            opt.zero_grad(set_to_none=True)
            loss = self._objective(img)

            if i % self.print_every == 0:
                print(f'{i}\t{loss.item():.3f}', flush=True)

            loss.backward()

            with torch.no_grad():
                g = img.grad
                gn = g.detach().float().norm(p=2)
                if torch.isfinite(gn) and gn > 0:
                    g.div_(gn + 1e-8)

            opt.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            if (self.progress_hook is not None) and (i % self.print_every == 0):
                try:
                    self.progress_hook(i, img)
                except Exception:
                    pass

            if self.saver and (i % self.save_every == 0):
                pf = f'{steps_folder}/steps_scan'
                os.makedirs(pf, exist_ok=True)
                save_image(img.data, f'{pf}/{clipname}_{img.shape[2]}_{i}_H{self.target_head}_L{layer}.png')

        opt.state = collections.defaultdict(dict)
        return img

def _probe_heads_for_prompt(model, premodel, block, layer_idx, tfeat, args_local, base_pos_embed, patch_size, device):
    """
    Returns list of dicts per head with:
      'head', 'target_score', 'itc', 'spec_sparsity', 'spec_topk', 'spec_margin',
      optional 'sim_h10', 'glob',
      plus 'score', 'score_adj', 'score_final' (filled after normalization),
      and 'probe_png' path for dumping sidecar TXT.
    Live metrics are printed only if --print_head_probe is set (to avoid extra compute).
    """
    H = block.attn.num_heads
    size = args_local.auto_head_size
    steps = args_local.auto_head_probe_steps
    beta = args_local.auto_head_beta

    grid_hw = int(size // patch_size)
    new_pos = _resize_positional_embedding(base_pos_embed, grid_hw)

    broad_str = getattr(args_local, "probe_broad_prompts", "")
    neg_str   = getattr(args_local, "probe_neg_prompts", "")
    t_broad, broad_names = _encode_prompt_list(premodel, broad_str, device)
    t_negs,  neg_names   = _encode_prompt_list(premodel, neg_str,   device)

    use_text_emb = (tfeat is not None)
    
    results = []
    probe_folder = f'{steps_folder}/__PROBE_L{layer_idx}'
    os.makedirs(probe_folder, exist_ok=True)

    for h in range(H):
        image = new_init(size, 1)

        loss = LossArray()
        loss += TotalVariation(2, size, 0.0001 * 1.0 * 0.25)

        if use_text_emb:
            from cliptools import PromptAlignmentLoss
            loss += PromptAlignmentLoss(
                clip_model=premodel, image=None,
                target_text_emb=tfeat,
                text_coeff=args_local.text_coeff_probe
            )
        pre = torch.nn.Sequential(
            RepeatBatch(1),
            ColorJitter(1, shuffle_every=True),
            GaussianNoise(1, True, 0.5, steps),
            Tile(1), Jitter()
        )
        post = Clip()

        all_heads_hook = AllHeadsCaptureHook(block)

        try:
            # -------- live progress hook --------
            progress_hook = None
            if getattr(args_local, "print_head_probe", False):
                @torch.no_grad()
                def _progress_hook(step_i: int, img: torch.Tensor):
                    # populate hooks on current image
                    _ = model(pre(img))
                    acts_all = all_heads_hook.activations
                    if acts_all is None:
                        return

                    # pooled target score (quick readout)
                    tgt_now = _pooled_head_score(
                        acts_all, h,
                        args_local.head_agg, args_local.head_topk_patches,
                        (args_local.query_mode or ("cls" if use_text_emb else "mean")),
                        block
                    )

                    im_feat = premodel.encode_image(post(img)).float()
                    im_feat = im_feat / (im_feat.norm(dim=-1, keepdim=True) + 1e-8)
                    itc_now = float((im_feat * tfeat).sum().item())

                    broad_best_name, broad_best_val = None, None
                    if t_broad:
                        vals = [(name, float((im_feat * tb).sum().item())) for tb, name in zip(t_broad, broad_names)]
                        vals.sort(key=lambda x: x[1], reverse=True)
                        broad_best_name, broad_best_val = vals[0]

                    neg_best_name, neg_best_val = None, None
                    if t_negs:
                        valsn = [(name, float((im_feat * tn).sum().item())) for tn, name in zip(t_negs, neg_names)]
                        valsn.sort(key=lambda x: x[1], reverse=True)
                        neg_best_name, neg_best_val = valsn[0]

                    margins = []
                    if broad_best_val is not None: margins.append(itc_now - broad_best_val)
                    if neg_best_val   is not None: margins.append(itc_now - neg_best_val)
                    spec_margin_now = float(min(margins)) if margins else 0.0

                    # sparsity/globalness
                    cand = acts_all[:, :, h, :].norm(dim=-1)  # [B, Seq]
                    B, Seq = cand.shape
                    g_sq = int(round((Seq - 1) ** 0.5))
                    dropped = (g_sq * g_sq + 1 == Seq)
                    cand_vec = cand[:, 1:] if dropped else cand

                    eps = 1e-8
                    p = cand_vec.clamp_min(0)
                    p = p / (p.sum(dim=-1, keepdim=True) + eps)       # [B, P]
                    ent = -(p * (p.clamp_min(eps).log())).sum(dim=-1) # [B]
                    glob_now = float((ent / (math.log(p.shape[1]) + eps)).mean().item())  # [0,1]
                    sparsity_now = 1.0 - glob_now

                    # head-10 similarity
                    sim_h10_now = None
                    if 10 < acts_all.shape[2]:
                        q = acts_all[:, :, 10, :].norm(dim=-1)
                        q = q[:, 1:] if dropped else q
                        q = q.clamp_min(0)
                        q = q / (q.sum(dim=-1, keepdim=True) + eps)
                        num = (p * q).sum(dim=-1)
                        den = (p.norm(dim=-1) * q.norm(dim=-1) + eps)
                        sim_h10_now = float((num / den).mean().item())

                    live = [
                        f"[probe] L{layer_idx} H{h} i={step_i}",
                        f"tgt={tgt_now:.3f}",
                        f"itc={itc_now:.3f}",
                        f"spars={sparsity_now:.3f}",
                        f"glob={glob_now:.3f}",
                        f"margin={spec_margin_now:.3f}",
                    ]
                    if sim_h10_now is not None:
                        live.append(f"sim_h10={sim_h10_now:.3f}")
                    if broad_best_name is not None:
                        live.append(f"broad_top=\"{broad_best_name}\":{broad_best_val:.3f}")
                    if neg_best_name is not None:
                        live.append(f"neg_top=\"{neg_best_name}\":{neg_best_val:.3f}")
                    print("  ".join(live), flush=True)

                progress_hook = _progress_hook

            with _temporary_pos_embed(premodel, new_pos):
                viz = HeadProbeVisualizer(
                    model, loss_array=loss, target_layer=layer_idx, target_head=h,
                    all_heads_hook=all_heads_hook,
                    pre_aug=pre, post_aug=post,
                    steps=steps, lr=0.7,
                    save_every=10, saver=False, print_every=10,
                    head_agg=args_local.head_agg,
                    head_topk_patches=args_local.head_topk_patches,
                    query_mode=(args_local.query_mode or None),
                    prompt_active=bool(use_text_emb),
                    sophia_k=args_local.sophia_k, sophia_mode=args_local.sophia_mode,
                    sophia_h_subsample=args_local.sophia_h_subsample,
                    sophia_gamma=args_local.sophia_gamma, sophia_tau=args_local.sophia_tau,
                    sophia_abs_hessian=args_local.sophia_abs_hessian, sophia_eps=args_local.sophia_eps,
                    u_lowpass=args_local.sophia_u_lowpass,
                    u_lowpass_cutoff=args_local.sophia_u_lowpass_cutoff,
                    num_hutch=args_local.sophia_num_hutch,
                    u_dist=args_local.sophia_u_dist,
                    progress_hook=progress_hook
                )
                image.data = viz(image, layer=layer_idx, clipname="PROBE")

                _ = model(pre(image))
                acts_all = all_heads_hook.activations
                if acts_all is None:
                    raise RuntimeError("Probe metrics requested before hooks were populated (end state).")

                tgt = _pooled_head_score(
                    acts_all, h,
                    args_local.head_agg, args_local.head_topk_patches,
                    (args_local.query_mode or ("cls" if use_text_emb else "mean")),
                    block
                )

                with torch.no_grad():
                    im_feat = premodel.encode_image(post(image)).float()
                    im_feat = im_feat / (im_feat.norm(dim=-1, keepdim=True) + 1e-8)
                    itc = float((im_feat * tfeat).sum().item())

                    itc_broad_max = max([(im_feat * tb).sum().item() for tb in t_broad], default=-1e9)
                    itc_neg_max   = max([(im_feat * tn).sum().item() for tn in t_negs ], default=-1e9)
                    margins = []
                    if t_broad: margins.append(itc - itc_broad_max)
                    if t_negs:  margins.append(itc - itc_neg_max)
                    spec_margin = float(min(margins)) if margins else 0.0

                # sparsity/globalness + head10 similarity
                sim_h10 = None
                try:
                    cand = acts_all[:, :, h, :].norm(dim=-1)  # [B, Seq]
                    B, Seq = cand.shape
                    g_sq = int(round((Seq - 1) ** 0.5))
                    dropped = (g_sq * g_sq + 1 == Seq)
                    cand_vec = cand[:, 1:] if dropped else cand

                    eps = 1e-8
                    p = cand_vec.clamp_min(0)
                    p = p / (p.sum(dim=-1, keepdim=True) + eps)           # [B, P]
                    ent = -(p * (p.clamp_min(eps).log())).sum(dim=-1)     # [B]
                    glob = float((ent / (math.log(p.shape[1]) + eps)).mean().item())  # [0,1]
                    spec_sparsity = 1.0 - glob

                    k = min(getattr(args_local, "spec_topk_k", 16), p.shape[1])
                    spec_topk = float(torch.topk(p, k=k, dim=1).values.sum(dim=1).mean().item())

                    if 10 < acts_all.shape[2]:
                        q = acts_all[:, :, 10, :].norm(dim=-1)
                        q = q[:, 1:] if dropped else q
                        q = q.clamp_min(0)
                        q = q / (q.sum(dim=-1, keepdim=True) + eps)
                        num = (p * q).sum(dim=-1)
                        den = (p.norm(dim=-1) * q.norm(dim=-1) + eps)
                        sim_h10 = float((num / den).mean().item())
                except Exception:
                    glob = 0.0
                    spec_sparsity = 0.0
                    spec_topk = 0.0

            png_path = f"{probe_folder}/H{h}_{size}.png"
            save_image(image, png_path)

            rec = {
                "head": h,
                "target_score": float(tgt),
                "itc": float(itc),
                "spec_sparsity": float(spec_sparsity),
                "spec_topk": float(spec_topk),
                "spec_margin": float(spec_margin),
                "probe_png": png_path,
                "glob": float(glob)
            }
            if sim_h10 is not None: rec["sim_h10"] = sim_h10
            results.append(rec)

            raw_line = (Fore.YELLOW + Style.BRIGHT + f"[probe-end] L{layer_idx} H{h}  tgt={tgt:.3f} itc={itc:.3f} "
                        f"spars={spec_sparsity:.3f} glob={glob:.3f} margin={spec_margin:.3f}" + Fore.RESET)
            if sim_h10 is not None:
                raw_line += Fore.RED + Style.BRIGHT + f" sim_h10={sim_h10:.3f}" + Fore.RESET
            print(raw_line, flush=True)
        finally:
            with contextlib.suppress(Exception):
                all_heads_hook.clear()
                all_heads_hook.remove()

    if not results:
        return results

    tmin, tmax = min(r["target_score"] for r in results), max(r["target_score"] for r in results)
    imin, imax = min(r["itc"] for r in results),         max(r["itc"] for r in results)
    smin, smax = min(r["spec_sparsity"] for r in results), max(r["spec_sparsity"] for r in results)
    kmin, kmax = min(r["spec_topk"]     for r in results), max(r["spec_topk"]     for r in results)
    mmin, mmax = min(r["spec_margin"]   for r in results), max(r["spec_margin"]   for r in results)
    gmin, gmax = min(r.get("glob",0.0)  for r in results), max(r.get("glob",0.0)  for r in results)

    def z(v, vmin, vmax): return 0.0 if vmax<=vmin else (v - vmin) / (vmax - vmin + 1e-8)

    use_head10_prior   = getattr(args_local, "use_head10_prior", False)
    head10_prior_w     = float(getattr(args_local, "head10_prior_weight", 0.25))
    globalness_weight  = float(getattr(args_local, "globalness_weight",   0.25))
    spec_w_sparsity    = float(getattr(args_local, "spec_weight_sparsity", 0.5))
    spec_w_topk        = float(getattr(args_local, "spec_weight_topk",     0.3))
    spec_w_margin      = float(getattr(args_local, "spec_weight_margin",   0.2))

    for r in results:
        base = (1 - beta) * z(r["target_score"], tmin, tmax) + beta * z(r["itc"], imin, imax)

        score_adj = base
        if use_head10_prior:
            if "sim_h10" in r:
                score_adj -= head10_prior_w * max(0.0, r["sim_h10"])
            score_adj -= globalness_weight * z(r.get("glob", 0.0), gmin, gmax)

        s_part = spec_w_sparsity * z(r["spec_sparsity"], smin, smax)
        k_part = spec_w_topk     * z(r["spec_topk"],     kmin, kmax)
        m_part = spec_w_margin   * z(r["spec_margin"],   mmin, mmax)
        score_final = score_adj + s_part + k_part + m_part

        r["score"] = float(base)
        r["score_adj"] = float(score_adj)
        r["score_final"] = float(score_final)

    # TXT dump next to each PNG preview
    for r in results:
        txt_path = r["probe_png"].rsplit(".", 1)[0] + ".txt"
        lines = []
        lines.append(f"layer={layer_idx}")
        lines.append(f"head={r['head']}")
        lines.append(f"target_score={r['target_score']:.6f}")
        lines.append(f"itc={r['itc']:.6f}")
        lines.append(f"glob={r.get('glob', 0.0):.6f}")          # higher = more global
        lines.append(f"spec_sparsity={r['spec_sparsity']:.6f}") # 1 - glob
        lines.append(f"spec_topk={r['spec_topk']:.6f}")
        lines.append(f"spec_margin={r['spec_margin']:.6f}")
        if "sim_h10" in r:   lines.append(f"sim_h10={r['sim_h10']:.6f}")
        lines.append(f"score={r['score']:.6f}")
        lines.append(f"score_adj={r['score_adj']:.6f}")
        lines.append(f"score_final={r['score_final']:.6f}")
        lines.append(f"auto_head_beta={beta:.6f}")
        lines.append(f"spec_weight_sparsity={spec_w_sparsity:.6f}")
        lines.append(f"spec_weight_topk={spec_w_topk:.6f}")
        lines.append(f"spec_weight_margin={spec_w_margin:.6f}")
        lines.append(f"use_head10_prior={int(use_head10_prior)}")
        lines.append(f"head10_prior_weight={head10_prior_w:.6f}")
        lines.append(f"globalness_weight={globalness_weight:.6f}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    results_sorted = sorted(results, key=lambda r: r["score_final"], reverse=True)
    print(Fore.CYAN + f"\n[probe-final] L{layer_idx} ranking (by score_final):" + Fore.RESET)
    for i, r in enumerate(results_sorted):
        line = (f"  #{i+1:02d} H{r['head']:02d}  score_final={r['score_final']:.3f}  "
                f"score_adj={r['score_adj']:.3f}  score={r['score']:.3f}  "
                f"spars={r['spec_sparsity']:.3f}  margin={r['spec_margin']:.3f}  glob={r.get('glob',0.0):.3f}")
        if "sim_h10" in r: line += f"  sim_h10={r['sim_h10']:.3f}"
        line += f"  [{r['probe_png']}]"
        print(line, flush=True)

    return results

class ImageNetVisualizer:
    def __init__(self, model, loss_array: LossArray, target_layer, target_head, all_heads_hook,
                 pre_aug: nn.Module = None, post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1,
                 save_every: int = 200, saver: bool = True, print_every: int = 5,
                 grad_rms_norm: bool = False, lr_scale: float = 1.0, lr_warmup_frac: float = 0.10, lr_min_mult: float = 0.05,
                 sophia_k: int = 10, sophia_mode: str = "hutchinson", sophia_h_subsample: float = 0.25,
                 sophia_gamma: float = 0.01, sophia_tau: float = 0.0, sophia_abs_hessian: bool = True, sophia_eps: float = 1e-12,
                 u_lowpass: bool = True, u_lowpass_cutoff: float = 0.30, num_hutch: int = 4, u_dist: str = "rademacher",
                 head_agg: str = "attn_weighted", head_topk_patches: int = 0, comp_alpha_heads: float = 1.0, comp_k_heads: int = 0,
                 query_mode: Optional[str] = None, prompt_active: bool = False,
                 attn_entropy_lambda: float = 0.0, attn_entropy_target: str = "low",
                 attn_prior: str = "none", attn_prior_lambda: float = 0.0,
                 neg_heads: Optional[List[int]] = None, neg_alpha: float = 0.0,
                 mlp_hook: Optional[MLPActHook] = None,
                 mlp_channels: Optional[List[int]] = None,
                 mlp_alpha: float = 0.0,
                 target_heads: Optional[List[int]] = None,
                 head_weights: Optional[List[float]] = None,
                 content_ref_feat: Optional[torch.Tensor] = None,   # 1xD CLIP feature of the start image
                 content_lambda: float = 0.0,                       # weight for CLIP content preservation
                 pixel_ref_img: Optional[torch.Tensor] = None,      # 1x3xHxW start image (current octave size)
                 pixel_lambda: float = 0.0,                         # weight for pixel MSE anchor
                 run_dir: str = "none",
                 **_):
        self.loss = loss_array
        self.model = model
        self.saver = saver
        self.target_layer = target_layer
        self.target_head = target_head
        self.all_heads_hook = all_heads_hook
        self.pre_aug = pre_aug
        self.post_aug = post_aug
        self.save_every = save_every
        self.print_every = print_every
        self.steps = int(steps)
        self.lr = float(lr)
        self.u_lowpass=u_lowpass
        self.u_lowpass_cutoff=u_lowpass_cutoff
        self.num_hutch=num_hutch
        self.u_dist=u_dist
        self.head_weights = (list(head_weights) if head_weights else None)
        # Scheduler/Grad
        self.grad_rms_norm = bool(grad_rms_norm)
        self.lr_scale = float(lr_scale)
        self.lr_warmup_frac = float(lr_warmup_frac)
        self.lr_min_mult = float(lr_min_mult)
        # SophiaViz Optimizer
        self.sophia_k = sophia_k
        self.sophia_mode = sophia_mode
        self.sophia_h_subsample = sophia_h_subsample
        self.sophia_gamma = sophia_gamma
        self.sophia_tau = sophia_tau
        self.sophia_abs_hessian = sophia_abs_hessian
        self.sophia_eps = sophia_eps
        # Pooling/Competition & Attention Shaping
        self.head_agg = head_agg
        self.head_topk_patches = head_topk_patches
        self.target_heads = target_heads
        self.comp_alpha_heads = comp_alpha_heads
        self.comp_k_heads = comp_k_heads
        self.prompt_active = prompt_active
        self.query_mode = (query_mode or ("cls" if prompt_active else "mean"))
        self.attn_entropy_lambda = attn_entropy_lambda
        self.attn_entropy_target = attn_entropy_target
        self.attn_prior = attn_prior
        self.attn_prior_lambda = attn_prior_lambda
        self.neg_heads = (neg_heads or [])
        self.neg_alpha = float(neg_alpha)
        # MLP Content Term
        self.mlp_hook = mlp_hook
        self.mlp_channels = list(mlp_channels) if mlp_channels else []
        self.mlp_alpha = float(mlp_alpha)
        # DeepDream Anchors
        self.content_ref_feat = content_ref_feat
        self.content_lambda   = float(content_lambda)
        self.pixel_ref_img    = pixel_ref_img
        self.pixel_lambda     = float(pixel_lambda)
        # Internal cache for grad center mask
        self._center_mask = None
        self._last_attn = 0.0
        self.run_dir = run_dir
   
    @staticmethod
    def _center_mask_1d(P: int, device):
        g = int(round(P ** 0.5))
        if g * g != P:
            return None
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, g, device=device),
                                torch.linspace(-1, 1, g, device=device),
                                indexing="ij")
        sigma = 0.6
        m = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
        s = m.sum()
        if torch.isfinite(s) and s > 0:
            return (m / s).reshape(1, P)
        return None

    @staticmethod
    def _prior_mask_1d(P: int, device, shape: str):
        g = int(round(P ** 0.5))
        if g * g != P or shape == "none":
            return None
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, g, device=device),
                                torch.linspace(-1, 1, g, device=device),
                                indexing="ij")
        if shape == "center":
            s = 0.6; m = torch.exp(-(xx**2 + yy**2) / (2 * s * s))
        elif shape == "ring":
            s1, s2 = 0.35, 0.9
            inner = torch.exp(-(xx**2 + yy**2) / (2 * s1 * s1))
            outer = torch.exp(-(xx**2 + yy**2) / (2 * s2 * s2))
            m = (outer - inner).clamp_min(0)
        elif shape == "diag":
            s = 0.3
            m = torch.exp(-((yy - xx)**2) / (2 * s * s)) + torch.exp(-((yy + xx)**2) / (2 * s * s))
        else:
            return None
        s = m.sum()
        if torch.isfinite(s) and s > 0:
            return (m / s).reshape(1, P)
        return None

    # build normalized weight tensor aligned with selected heads
    def _head_weights_tensor(self, device, Hsel: int):
        if (self.head_weights is None) or (Hsel <= 0):
            w = torch.full((Hsel,), 1.0 / max(1, Hsel), device=device)
        else:
            w = torch.tensor(self.head_weights, dtype=torch.float32, device=device)
            if w.numel() != Hsel:
                # fallback to uniform if mismatch
                w = torch.full((Hsel,), 1.0 / max(1, Hsel), device=device)
            s = torch.clamp(w.sum(), min=1e-8)
            w = w / s
        return w.view(1, Hsel, 1)

    def _compute_objective(self, img: torch.Tensor, track_stats_override: bool):
        augmented = self.pre_aug(img) if self.pre_aug is not None else img
        _ = self.model(augmented)  # trigger hooks

        acts_all = self.all_heads_hook.activations  # [B, Seq, H, Dh]
        if acts_all is None:
            raise RuntimeError("No activations from all heads hook")

        # build token scores for either single head or pooled multi-head
        if self.target_heads and len(self.target_heads) > 1:
            tgt_heads = self.target_heads
            target_set = set(tgt_heads)
            target_head_acts = acts_all[:, :, tgt_heads, :]              # [B, Seq, |Hsel|, Dh]
            Hsel = len(tgt_heads)
            w = self._head_weights_tensor(acts_all.device, Hsel)         # [1, |Hsel|, 1]
            head_norms = target_head_acts.norm(dim=-1)                   # [B, Seq, |Hsel|]
            w_seq = w.squeeze(-1)                                        # [1, |Hsel|]
            token_scores_full = (head_norms * w_seq).sum(dim=2)          # [B, Seq]
        else:
            target_set = {self.target_head}
            target_head_acts = acts_all[:, :, self.target_head, :]        # [B, Seq, Dh]
            token_scores_full = target_head_acts.norm(dim=-1)             # [B, Seq]

        B, Seq = token_scores_full.shape
        g_sq = int(round((Seq - 1) ** 0.5))
        if g_sq * g_sq + 1 == Seq:
            token_scores = token_scores_full[:, 1:]                   # drop CLS on keys
            dropped_cls_on_keys = True
        else:
            token_scores = token_scores_full
            dropped_cls_on_keys = False
        P = token_scores.shape[1]

        # attention probs from the SAME block (if any)
        Aw = None
        block = getattr(self.all_heads_hook, "block", None)
        if block is not None and getattr(block, "attn_probs", None) is not None:
            probs_flat = block.attn_probs#.detach()                    # [B*H, Q, K]
            Hh = acts_all.shape[2]
            Q, K = probs_flat.shape[-2], probs_flat.shape[-1]
            attn_probs = probs_flat.view(B, Hh, Q, K)                 # [B,H,Q,K]

            qm = self.query_mode
            # Compute Aw for single head, or mean over selected heads in joint mode
            if self.target_heads and len(self.target_heads) > 1:
                heads = torch.as_tensor(self.target_heads, device=attn_probs.device, dtype=torch.long)
                Hsel = int(heads.numel())
                w = self._head_weights_tensor(attn_probs.device, Hsel)   # [1, Hsel, 1]
                w_headsK  = w                    # for tensors shaped [B, Hsel, K]
                w_headsQK = w.view(1, Hsel, 1, 1)  # for tensors shaped [B, Hsel, Q, K]

                if qm == "cls" and Q >= 1:
                    # [B, Hsel, K] * [1, Hsel, 1] -> [B, Hsel, K] -> sum over heads
                    S = attn_probs[:, heads, 0, :]                         # [B, Hsel, K]
                    Aw = (S * w_headsK).sum(dim=1)                         # [B, K]
                elif qm == "mean":
                    if Q == Seq and g_sq * g_sq + 1 == Seq:
                        # drop CLS on queries, mean over query dim (dim=2)
                        S = attn_probs[:, heads, 1:, :].mean(dim=2)        # [B, Hsel, K]
                    else:
                        # mean over query dim (dim=2)
                        S = attn_probs[:, heads].mean(dim=2)               # [B, Hsel, K]
                    Aw = (S * w_headsK).sum(dim=1)                         # [B, K]

                elif qm == "topk":
                    # choose top-k queries based on token_scores, then weight heads and gather those queries
                    kq = self.head_topk_patches if self.head_topk_patches > 0 else max(1, P // 32)
                    _, top_idx = torch.topk(token_scores, k=kq, dim=1)     # [B, kq]
                    q_idx = top_idx + 1 if (Q == Seq and g_sq * g_sq + 1 == Seq) else top_idx
                    gather_idx = q_idx.unsqueeze(-1).expand(-1, -1, K)     # [B, kq, K]

                    Ah = attn_probs[:, heads]                               # [B, Hsel, Q, K]
                    Awh = (Ah * w_headsQK).sum(dim=1)                       # [B, Q, K]  (weighted over heads)
                    Aw  = torch.gather(Awh, dim=1, index=gather_idx).mean(dim=1)  # [B, K]

            else:
                if qm == "cls" and Q >= 1:
                    Aw = attn_probs[:, self.target_head, 0, :]
                elif qm == "mean":
                    if Q == Seq and g_sq * g_sq + 1 == Seq:
                        Aw = attn_probs[:, self.target_head, 1:, :].mean(dim=1)
                    else:
                        Aw = attn_probs[:, self.target_head].mean(dim=1)
                elif qm == "topk":
                    kq = self.head_topk_patches if self.head_topk_patches > 0 else max(1, P // 32)
                    _, top_idx = torch.topk(token_scores, k=kq, dim=1)
                    q_idx = top_idx + 1 if (Q == Seq and g_sq * g_sq + 1 == Seq) else top_idx
                    gather_idx = q_idx.unsqueeze(-1).expand(-1, -1, K)
                    Ah = attn_probs[:, self.target_head]                  # [B,Q,K]
                    Aw = torch.gather(Ah, dim=1, index=gather_idx).mean(dim=1)

            if Aw is not None and Aw.size(1) == Seq and dropped_cls_on_keys:
                Aw = Aw[:, 1:]
            if Aw is not None:
                Aw = Aw / (Aw.sum(dim=-1, keepdim=True) + 1e-8)

        # --- build a dynamic prior from head-10 (smoothed) ---
        p10 = None
        if self.attn_prior == "head10" and block is not None and getattr(block, "attn_probs", None) is not None:
            # attn_probs: [B*H, Q, K] -> [B, H, Q, K]
            probs_flat = block.attn_probs
            B, Seq, Hh, Dh = acts_all.shape
            Q, K = probs_flat.shape[-2], probs_flat.shape[-1]
            attn_probs = probs_flat.view(B, Hh, Q, K)

            head10_idx = min(10, Hh - 1)
            if self.query_mode == "cls" and Q >= 1:
                p10 = attn_probs[:, head10_idx, 0, :]                    # [B, K]
            elif self.query_mode == "mean":
                if Q == Seq and (int(round((Seq - 1) ** 0.5)) ** 2 + 1 == Seq):
                    p10 = attn_probs[:, head10_idx, 1:, :].mean(dim=1)   # drop CLS on queries if grid-like
                else:
                    p10 = attn_probs[:, head10_idx].mean(dim=1)          # [B, K]
            elif self.query_mode == "topk":
                # topk handling using the target head’s token_scores as query selector
                kq = self.head_topk_patches if self.head_topk_patches > 0 else max(1, P // 32)
                _, top_idx = torch.topk(token_scores, k=kq, dim=1)       # [B, kq]
                q_idx = top_idx + 1 if (Q == Seq and (int(round((Seq - 1) ** 0.5)) ** 2 + 1 == Seq)) else top_idx
                gather_idx = q_idx.unsqueeze(-1).expand(-1, -1, K)       # [B,kq,K]
                Ah10 = attn_probs[:, head10_idx]                         # [B,Q,K]
                p10 = torch.gather(Ah10, dim=1, index=gather_idx).mean(dim=1)  # [B,K]

            # Drop CLS on KEYS if target
            if p10 is not None and p10.size(1) == Seq and dropped_cls_on_keys:
                p10 = p10[:, 1:]  # [B, P]

            # --- smooth head-10 map ---
            if p10 is not None:
                # reshape to grid, blur via avg-pool, then upsample
                g = int(round(P ** 0.5))
                p10g = p10.reshape(B, 1, g, g)
                ksize = max(2, g // 8)       # gentle low-pass; tune  g//6..g//10
                if ksize % 2 == 0:
                    ksize += 1
                pad = ksize // 2
                p10g = F.avg_pool2d(F.pad(p10g, (pad, pad, pad, pad), mode="reflect"), ksize, stride=1)
                p10 = F.interpolate(p10g, size=(g, g), mode="bilinear", align_corners=False).reshape(B, P)
                p10 = p10.clamp_min(0)
                p10 = p10 / (p10.sum(dim=-1, keepdim=True) + 1e-8)  # stochastic prior

        center_mask_1d = self._center_mask_1d(P, token_scores.device)
        prior_mask_1d  = self._prior_mask_1d(P, token_scores.device, self.attn_prior)

        # pooling
        if self.head_agg == "mean":
            target_score = token_scores.mean()
        elif self.head_agg == "topk":
            k = self.head_topk_patches if self.head_topk_patches > 0 else max(1, P // 32)
            vals, _ = torch.topk(token_scores, k=k, dim=1)
            target_score = vals.mean()
        elif self.head_agg == "center" and center_mask_1d is not None:
            target_score = (token_scores * center_mask_1d).sum(dim=1).mean()
        elif self.head_agg == "attn_weighted" and Aw is not None:
            target_score = (token_scores * Aw).sum(dim=1).mean()
        else:
            target_score = token_scores.mean()

        # Competitors exclude the entire target set (single head or multi-head)
        other_heads_mask = [k for k in range(acts_all.shape[2]) if k not in target_set]
        if len(other_heads_mask) == 0:
            comp_score = torch.zeros((), device=acts_all.device)
        else:
            other_acts = acts_all[:, :, other_heads_mask, :].norm(dim=-1).mean(dim=1)    # [B, H_other]
            if self.comp_k_heads and self.comp_k_heads > 0:
                kk = min(self.comp_k_heads, other_acts.size(1))
                comp_vals, _ = torch.topk(other_acts, k=kk, dim=1)
                comp_score = comp_vals.mean()
            else:
                comp_score = other_acts.mean()

        attn_loss = -(target_score - self.comp_alpha_heads * comp_score)

        # explicit negative-head penalty
        if self.neg_heads and self.neg_alpha > 0.0:
            neg_acts = acts_all[:, :, self.neg_heads, :].norm(dim=-1).mean()
            attn_loss = attn_loss + self.neg_alpha * neg_acts

        # entropy shaping
        if self.attn_entropy_lambda != 0.0 and Aw is not None:
            p = Aw.clamp_min(1e-8)
            Hent = -(p * p.log()).sum(dim=-1).mean()
            if self.attn_entropy_target == "low":
                attn_loss = attn_loss + self.attn_entropy_lambda * Hent
            else:
                attn_loss = attn_loss - self.attn_entropy_lambda * Hent

        # prior KL
        if self.attn_prior_lambda != 0.0 and Aw is not None:
            kl = None
            if self.attn_prior == "head10" and p10 is not None:
                p  = Aw.clamp_min(1e-8)
                pi = p10.detach()  # stop-grad teacher
                kl = (p * (p.log() - pi.log())).sum(dim=-1).mean()
            else:
                prior_mask_1d = self._prior_mask_1d(P, token_scores.device, self.attn_prior)
                if prior_mask_1d is not None:
                    p  = Aw.clamp_min(1e-8)
                    pi = prior_mask_1d.expand_as(p).clamp_min(1e-8)
                    kl = (p * (p.log() - pi.log())).sum(dim=-1).mean()

            if kl is not None:
                attn_loss = attn_loss + self.attn_prior_lambda * kl

        # MLP-channel bonus (maximize content)
        if self.mlp_hook is not None and len(self.mlp_channels) > 0 and self.mlp_alpha > 0.0:
            mlp_act = self.mlp_hook.activations
            if mlp_act is None:
                raise RuntimeError("MLPActHook produced no activations")
            idx = torch.tensor(self.mlp_channels, device=mlp_act.device, dtype=torch.long)
            sel = torch.index_select(mlp_act, dim=-1, index=idx)  # [..., k]
            mlp_score = sel.abs().mean()
            attn_loss = attn_loss - self.mlp_alpha * mlp_score  # lower loss when content ↑

        tv_loss = self.loss(augmented, track_stats_override=track_stats_override)

        # DeepDream content anchors
        content_loss = torch.zeros((), device=img.device)
        # CLIP feature anchor (keeps high-level content informed by start image)
        if (self.content_ref_feat is not None) and (self.content_lambda > 0.0):
            im_feat = self.model(self.post_aug(img)).float()
            im_feat = im_feat / (im_feat.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sim = (im_feat * self.content_ref_feat).sum()
            content_loss = content_loss + self.content_lambda * (1.0 - cos_sim)

        # Pixel anchor (keeps color/low-freqs from washing out)
        if (self.pixel_ref_img is not None) and (self.pixel_lambda > 0.0):
            pref = self.pixel_ref_img.expand(img.shape[0], -1, -1, -1)
            if getattr(args, "deepdream", False):
                # luma-only MSE to avoid global hue seesaw
                Y, _, _ = _rgb_to_yuv(img)
                Yref, _, _ = _rgb_to_yuv(pref)
                content_loss = content_loss + self.pixel_lambda * F.mse_loss(Y, Yref)
            else:
                content_loss = content_loss + self.pixel_lambda * F.mse_loss(img, pref)


        total = attn_loss + tv_loss + content_loss

        if track_stats_override:
            try:
                self._last_attn = float(attn_loss.detach().item())
            except Exception:
                pass

        return total

    @torch.no_grad()
    def _lr_for_step(self, i: int):
        warmup = max(1, int(round(self.steps * self.lr_warmup_frac)))
        base_lr = self.lr * self.lr_scale
        min_lr = base_lr * self.lr_min_mult
        if i < warmup:
            return base_lr * float(i + 1) / float(max(1, warmup))
        denom = max(1, self.steps - warmup)
        progress = float(i - warmup) / float(denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    def _hvp_closure(self, img: torch.Tensor) -> torch.Tensor:
        return self._compute_objective(img, track_stats_override=False)

    def __call__(self, img: torch.Tensor = None, layer: int = None, clipname: str = None):
        if not img.is_cuda or img.device != device:
            img = img.to(device)
        if not img.requires_grad:
            img.requires_grad_()

        optimizer = SophiaViz(
            [img],
            lr=self.lr,
            betas=(0.96, 0.99),
            weight_decay=0.0,
            k=self.sophia_k,
            hessian_mode=self.sophia_mode,
            h_subsample=self.sophia_h_subsample,
            gamma=self.sophia_gamma,
            tau=self.sophia_tau,
            abs_hessian=self.sophia_abs_hessian,
            eps=self.sophia_eps,
            u_lowpass=self.u_lowpass,
            u_lowpass_cutoff=self.u_lowpass_cutoff,
            num_hutch=self.num_hutch,
            u_dist=self.u_dist,
            maximize=False,
            capturable=False,
        )

        print(Fore.YELLOW + Style.BRIGHT + f"\n#i\tTotal\tAttn\tLR\t{self.loss.header()}" + Fore.RESET, flush=True)
        for i in range(self.steps + 1):
            current_lr = self._lr_for_step(i)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            
            optimizer.zero_grad(set_to_none=True)
            total_loss = self._compute_objective(img, track_stats_override=True)

            total_loss.backward()

            # grad normalization (RMS or L2)
            gimg = img.grad
            if self.grad_rms_norm:
                gs = gimg.detach().float().std()
                if torch.isfinite(gs) and gs > 0:
                    gimg.div_(gs + 1e-8)
            else:
                gn = gimg.detach().float().norm(p=2)
                if torch.isfinite(gn) and gn > 0:
                    gimg.div_(gn + 1e-8)

            # soft center bias on gradients
            with torch.no_grad():
                Bimg, C, H, W = img.shape
                if (self._center_mask is None) or (self._center_mask.shape[-2:] != (H, W)):
                    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=img.device),
                                            torch.linspace(-1, 1, W, device=img.device),
                                            indexing="ij")
                    sigma_pix = 0.6
                    m = torch.exp(-(xx**2 + yy**2) / (2 * sigma_pix * sigma_pix)).clamp_min(1e-3)
                    self._center_mask = m[None, None, :, :]
                img.grad.mul_(self._center_mask)

            optimizer.step(hvp_closure=lambda: self._hvp_closure(img))

            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            
            if i % self.print_every == 0:
                tot = float(total_loss.detach().item())
                print(f"{i}\t{tot:.2f}\t{self._last_attn:.3f}\t{current_lr:.2f}\t{self.loss}", flush=True)

            if self.saver and i % self.save_every == 0:
                temp_folder = os.path.join(self.run_dir, "steps") if self.run_dir else os.path.join(steps_folder, f"steps_H{self.target_head}_L{self.target_layer}")
                os.makedirs(temp_folder, exist_ok=True)
                save_image(img.data, f'{temp_folder}/{clipname}_{img.shape[2]}_{i}_H{self.target_head}_L{self.target_layer}.png')

            # DeepDream: Overlay original image to prime
            if args.deepdream:
                if i < (self.steps*args.overlay_frac):
                    if self.pixel_ref_img is not None and (args.overlay_alpha > 0.0):
                        with torch.no_grad():
                            pref = self.pixel_ref_img.expand_as(img)
                            Y, U, V = _rgb_to_yuv(img)
                            Yref, _, _ = _rgb_to_yuv(pref)
                            Y.lerp_(Yref, args.overlay_alpha)
                            img.copy_(_yuv_to_rgb(Y, U, V))

            if args.leaf_ops and not args.deepdream:
                if (i % args.leaf_every) == 0 and i < (self.steps*0.75) and i > 10:
                    with torch.no_grad():
                        # --- GAUSSIAN NOISE ---
                        img.add_(torch.randn_like(img) * args.leaf_gauss)                     
                        # --- COLOR JITTER ---
                        B = img.shape[0]
                        mean_amp = args.leaf_col_amp       # shift amplitude: mean ∈ [-0.03, +0.03]
                        std_log_amp = args.leaf_col_std    # scale via exp(u), u ∈ [-0.04, +0.04] -> std ∈ [~0.96, ~1.04]
                        # sample per-channel offsets/scales, broadcast to HxW
                        col_jit_mean = (torch.rand((B, 3, 1, 1), device=img.device, dtype=img.dtype).sub_(0.5).mul_(2.0).mul_(mean_amp))
                        col_jit_std  = torch.exp(torch.rand((B, 3, 1, 1), device=img.device, dtype=img.dtype).sub_(0.5).mul_(2.0).mul_(std_log_amp))
                        img.mul_(col_jit_std).add_(col_jit_mean) # not .div_, oops
                        # -- CHROMA CORRECTION --
                        if img.shape[2] < 400:
                            _edge_aware_chroma_smooth_(img, ksize=args.leaf_chroma_col, edge_k=args.leaf_chroma_edge, y_gamma=0.55, amount=args.leaf_chroma)
                        else:
                            _edge_aware_chroma_smooth_(img, ksize=(args.leaf_chroma_col+6), edge_k=(args.leaf_chroma_edge+4), y_gamma=0.55, amount=args.leaf_chroma)
                        # -- UNSHARP MASK --
                        # micro unsharp on Y only
                        Y, U, V = _rgb_to_yuv(img)
                        Yb = F.avg_pool2d(Y, 3, 1, 1)
                        img.copy_(_yuv_to_rgb(Y + (Y - Yb) * 0.05, U, V))                        
                        # -- Corrective Jitter --
                        _subpixel_jitter_(img, max_shift_px=0.5)

            # post-augmentation; also clips to valid range for previous ops
            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

        # free optimizer state
        optimizer.state = collections.defaultdict(dict)
        return img

def generate_visualizations(model, premodel, clipname, layer_range, head_range,
                            image_size, tv, lr, steps, print_every, save_every, saver, coefficient,
                            octave_sizes, steps_per_octave, tv_scales, lr_scales,
                            lr_warmup_frac, lr_min_mult, grad_rms_norm,
                            base_pos_embed, patch_size, sophia_args, neg_heads, neg_alpha,
                            mlp_channels: Optional[List[int]] = None,
                            mlp_alpha: float = 0.0, joint_target_heads: Optional[List[int]] = None,
                            joint_head_weights: Optional[List[float]] = None,
                            content_ref_feat: Optional[torch.Tensor] = None, content_lambda: float = 0.0,
                            pixel_ref_img: Optional[torch.Tensor] = None, pixel_lambda: float = 0.0,
                            ):

    for layer in layer_range:
        block = premodel.visual.transformer.resblocks[layer]
        mlp_hook = MLPActHook(block) if (mlp_channels and len(mlp_channels) > 0 and mlp_alpha > 0.0) else None

        try:
            if joint_target_heads and len(joint_target_heads) > 1:
                iter_heads = [joint_target_heads[0]]
                head_naming_tag = f"Hmulti_{'-'.join(map(str, joint_target_heads))}"
                target_heads_for_viz = joint_target_heads
            else:
                iter_heads = head_range
                head_naming_tag = None
                target_heads_for_viz = None

            for head in iter_heads:
                all_heads_hook = AllHeadsCaptureHook(block)
                tag = (head_naming_tag if head_naming_tag is not None else f"H{head}")
                print(Fore.MAGENTA + Style.BRIGHT + f"\nGenerating visualization for Layer {layer}, {tag}..." + Fore.RESET)

                # Set-up folder for saving stuff in
                qtag = (args.query_mode if (args.query_mode and args.query_mode.strip()) else "auto")
                run_dir = os.path.join(
                    steps_folder,
                    f"{clipname}_{tag}_L{layer}_agg-{args.head_agg}-q-{qtag}-prio-{args.attn_prior}"
                )
                os.makedirs(os.path.join(run_dir, "steps"), exist_ok=True)
                moved_probe = False # We only know destination folder after this (head probe) has run :-)

                smallest = octave_sizes[0]
                if args.deepdream:
                    if args.mlp_saliency_img.strip():
                        try:
                            image = _deepdream_init_from_image(args.mlp_saliency_img, smallest, 1, device=device)
                            print(Fore.GREEN + Style.BRIGHT + "[deepdream] Using --mlp_saliency_img as start tensor." + Fore.RESET)
                        except Exception as e:
                            print(Fore.RED + Style.BRIGHT + f"[deepdream] Failed to load image: {e}. Falling back to random init." + Fore.RESET)
                            image = new_init(smallest, 1)
                    else:
                        print(Fore.RED + Style.BRIGHT + "[deepdream] --mlp_saliency_img not provided; falling back to random init." + Fore.RESET)
                        image = new_init(smallest, 1)
                else:
                    image = new_init(smallest, 1)

                for oi, cur_size in enumerate(octave_sizes):
                    grid_hw = int(cur_size // patch_size)
                    new_pos = _resize_positional_embedding(base_pos_embed, grid_hw)

                    if oi > 0:
                        with torch.no_grad():
                            image = F.interpolate(image, size=(cur_size, cur_size), mode='bilinear', align_corners=False)

                    if args.deepdream:
                        add_noise = (oi == 0)
                        pre, post = (
                            torch.nn.Sequential(
                                RepeatBatch(repeats),
                                ColorJitter(repeats, shuffle_every=True, mean=0.00001, std=0.00001, use_fixed_random_seed=False),
                                GaussianNoise(repeats, add_noise, 0.00001 if add_noise else 0.0, iterations),
                                Tile(cur_size // cur_size),
                                Jitter()
                            ),
                            Clip()
                        )     
                    else:
                        add_noise = (oi == 0)
                        pre, post = (
                            torch.nn.Sequential(
                                RepeatBatch(repeats),
                                ColorJitter(repeats, shuffle_every=True),
                                GaussianNoise(repeats, add_noise, 0.5 if add_noise else 0.0, iterations),
                                Tile(cur_size // cur_size),
                                Jitter()
                            ),
                            Clip()
                        )

                    # DeepDream anchors
                    content_ref_feat = None
                    pixel_ref_img = None
                    if args.deepdream and args.mlp_saliency_img.strip():
                        try:
                            with Image.open(args.mlp_saliency_img).convert("RGB") as im:
                                im = im.resize((cur_size, cur_size), Image.BICUBIC)
                                arr = np.asarray(im, dtype=np.float32) / 255.0
                            ref = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(image.device)
                            pixel_ref_img = ref.detach()
                            with _temporary_pos_embed(premodel, new_pos):
                                content_ref_feat = _encode_clip_feat(premodel, post, pixel_ref_img).detach()
                        except Exception as e:
                            print(Fore.RED + Style.BRIGHT + f"[deepdream] Anchor build failed at size {cur_size}: {e}" + Fore.RESET)
                    
                    dd_clip_w  = 0.5 if args.deepdream else 0.0     # CLIP cosine anchor
                    dd_pixel_w = 0.25 if args.deepdream else 0.0    # pixel MSE

                    loss = LossArray()
                    
                    if args.decorr:
                        loss += TruePatchCorrelationLoss(patch=7, stride=3, coefficient=args.dec_pen, luma_only=True, local_sigma_px=2.5*7, local_radius_px=6*3)
                    if args.chan_corr:
                        loss += ChannelCorrelationLoss(coefficient=args.chan)         
                    if args.col_var:
                        loss += ColorVariationLowFreq(p=2, size=cur_size, coefficient=args.col, sigma=2.0)
                    if args.freq_var:
                        loss += FrequencySlopePenalty(alpha=1.2, spike_l2=0.0, coefficient=args.freq, shift_frac=0.125, use_hann=True, bin_jitter=True)   
                    if args.fakebatch:
                        loss += CLIPAugCosineQueue(premodel, post, queue_size=args.fake_n, coefficient=0.5)
                    if args.mean_std:
                        loss += MeanStdLoss(mean_target=0.5, std_target=0.25, coefficient=args.std_coeff)
                    
                    loss += TotalVariation(2, cur_size, coefficient * tv * tv_scales[oi])

                    # --- Text Guidance from PROMPT or AUTO_PROMPT ---
                    prompt_active = False
                    tfeat_for_loss = None

                    if getattr(args, "auto_tfeat", None) is not None:
                        prompt_active = True
                        tfeat_for_loss = args.auto_tfeat
                    elif isinstance(args.prompt, str) and args.prompt.strip():
                        prompt_active = True
                        with torch.no_grad():
                            tokens = clip.tokenize([args.prompt]).to(image.device)
                            tfeat_tmp = premodel.encode_text(tokens).float()
                            tfeat_for_loss = tfeat_tmp / (tfeat_tmp.norm(dim=-1, keepdim=True) + 1e-8)

                    if prompt_active and (tfeat_for_loss is not None):
                        from cliptools import PromptAlignmentLoss
                        loss += PromptAlignmentLoss(
                            clip_model=premodel, image=None,
                            target_text_emb=tfeat_for_loss,
                            text_coeff=args.text_coeff
                        )

                    visualizer = ImageNetVisualizer(
                        model, loss_array=loss, target_layer=layer, target_head=head,
                        all_heads_hook=all_heads_hook,
                        pre_aug=pre, post_aug=post, print_every=print_every,
                        lr=(lr * (0.5 if args.deepdream else 1.0)),
                        steps=steps_per_octave[oi], save_every=save_every, saver=saver, coefficient=coefficient,
                        grad_rms_norm=grad_rms_norm,
                        lr_scale=lr_scales[oi],
                        lr_warmup_frac=lr_warmup_frac,
                        lr_min_mult=lr_min_mult,
                        head_agg=args.head_agg,
                        head_topk_patches=args.head_topk_patches,
                        comp_alpha_heads=args.comp_alpha_heads,
                        comp_k_heads=args.comp_k_heads,
                        query_mode=(args.query_mode or None),
                        prompt_active=prompt_active,
                        attn_entropy_lambda=(0.5*args.attn_entropy_lambda if args.deepdream else args.attn_entropy_lambda),
                        attn_entropy_target=args.attn_entropy_target,
                        attn_prior=args.attn_prior,
                        attn_prior_lambda=(0.5*args.attn_prior_lambda if args.deepdream else args.attn_prior_lambda),
                        neg_heads=neg_heads, neg_alpha=neg_alpha,
                        mlp_hook=mlp_hook, mlp_channels=mlp_channels or [], mlp_alpha=mlp_alpha,
                        target_heads=target_heads_for_viz,
                        head_weights=joint_head_weights,
                        content_ref_feat=content_ref_feat, content_lambda=dd_clip_w,
                        pixel_ref_img=pixel_ref_img, pixel_lambda=dd_pixel_w,
                        run_dir=run_dir,
                        **sophia_args
                    )

                    with _temporary_pos_embed(premodel, new_pos):
                        image.data = visualizer(image, layer=layer, clipname=clipname)

                    #save_image(image, f'{steps_folder}/{clipname}_H{tag}_L{layer}_{cur_size}.png')
                    out_png = os.path.join(run_dir, f"{clipname}_{tag}_L{layer}_{cur_size}.png")
                    save_image(image, out_png)
                    _write_args_json_for_image(out_png, args)

                    if not moved_probe:
                        _move_probe_for_layer(steps_folder, layer, run_dir)
                        moved_probe = True

        finally:
            with contextlib.suppress(Exception):
                all_heads_hook.clear()
                all_heads_hook.remove()

def main():
    args_local = args

    if args.attn_move_init and args.attn_move_late:
        raise SystemExit(Fore.RED + "[error] Choose only one of --attn_move_init **OR** --attn_move_late." + Fore.RESET)

    if getattr(args_local, "force_reprobe", False):
        cache_root = os.path.join(steps_folder, "temp")
        print(Fore.CYAN + Style.BRIGHT + f"[cache] --force_reprobe set: clearing '{cache_root}' and bypassing cache reads." + Fore.RESET)
        with contextlib.suppress(Exception):
            shutil.rmtree(cache_root)
        os.makedirs(cache_root, exist_ok=True)

    model, premodel, preprocess, base_pos_embed, patch_size = load_clip_model()
    input_dims, num_layers, num_features, num_heads = get_clip_dimensions(premodel, preprocess)
    image_size = input_dims
    print(f"\nNative input dimension for {clipmodel}:" + Fore.GREEN + Style.BRIGHT + f" {input_dims}" + Fore.RESET)
    print("Layers:" + Fore.GREEN + Style.BRIGHT + f"0-{num_layers-1} with 0-{num_features-1} Features / Layer, 0-{num_heads-1} Attn Heads, Patch Size: {patch_size}" + Fore.RESET)

    layer_range   = clamp_range(parse_range(args_local.layer_range),  0, num_layers-1)
    head_range    = clamp_range(parse_range(args_local.head_range),    0, num_heads-1)
    attn_from     = clamp_range(parse_range(args_local.attn_from),  0, num_layers-1)
    attn_to       = clamp_range(parse_range(args_local.attn_to),  0, num_layers-1)

    if args.ablate_registers:
        reg_neurons_layer_11 = [9, 987, 1967, 2555, 3661, 3784]         # Register Neurons?! How, what, where? See:
        reg_neurons_layer_12 = [42, 983, 1571, 2687, 3002, 3008, 3868]  # https://github.com/zer0int/CLIP-test-time-registers
        hooks_layer_11 = [FeatureScalerHook(premodel, 11, idx, 0, 'visual') for idx in reg_neurons_layer_11]
        hooks_layer_12 = [FeatureScalerHook(premodel, 12, idx, 0, 'visual') for idx in reg_neurons_layer_12]

    if args.attn_move_init:
        premodel = attncopy(premodel, 'visual', from_=attn_from, to_=attn_to)

    if (args_local.auto_head or args_local.auto_head_multi) and len(layer_range) != 1:
        print(Fore.YELLOW + Style.BRIGHT + "[warn] --auto_head* currently expects a single target layer; using the first." + Style.RESET_ALL)
        layer_range = [layer_range[0]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ============ AUTO-HEAD, PROMPT, ETC. PREPROCESSING ============
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # --- Encode explicit text prompts (baseline path) ---
    tfeat = _encode_prompt_emb(premodel, args_local.prompt, device=device) if args_local.prompt.strip() else None

    cache_dir = os.path.join(steps_folder, "temp")
    os.makedirs(cache_dir, exist_ok=True)

    selected_heads      = head_range[:] # default: user range
    joint_target_heads  = None
    joint_head_weights  = None
    neg_heads           = []

    # --- MLP top-k channels from a real image ---
    mlp_channels = []
    if args_local.mlp_saliency_img.strip():
        L_mlp = layer_range[0]
        use_grad = bool(args_local.mlp_saliency_use_grad and args_local.prompt.strip())
        try:
            mlp_channels = _compute_mlp_topk_from_image(
                premodel, preprocess, L_mlp, args_local.mlp_saliency_img,
                args_local.prompt if use_grad else "", args_local.mlp_topk, use_grad, device=device
            )
            print(Fore.CYAN + f"[mlp-saliency] L{L_mlp} top-{len(mlp_channels)} channels: {mlp_channels}" + Fore.RESET)
        except Exception as e:
            print(Fore.YELLOW + f"[warn] MLP saliency failed: {e}. Continuing without MLP channels." + Fore.RESET)
            mlp_channels = []

    # --- AUTO-PROMPT (GA on text) → auto_tfeat (1xD) ---
    if args_local.auto_prompt:
        if not args_local.mlp_saliency_img.strip():
            raise SystemExit(Fore.RED + "[error] --auto_prompt requires --mlp_saliency_img (an image path)." + Fore.RESET)

        img_path = args_local.mlp_saliency_img
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_sha  = _sha1_of_file(img_path)[:12]
        args.auto_prompt_sha = img_sha

        os.makedirs(f"{cache_dir}/txtembeds", exist_ok=True)
        os.makedirs(f"{cache_dir}/txtopinion", exist_ok=True)
        embed_cache_path  = f"{cache_dir}/txtembeds/{img_name}_{img_sha}_emb.pt"
        opin_cache_path   = f"{cache_dir}/txtopinion/tokens_{img_name}_{img_sha}.txt"

        need_reascent = bool(getattr(args_local, "force_reprobe", False)) or not os.path.exists(embed_cache_path)

        if not need_reascent:
            print(Fore.CYAN + f"[auto_prompt] Using cached text embedding: {embed_cache_path}" + Fore.RESET)
            best_text_embeddings = torch.load(embed_cache_path, map_location=device).float()
        else:
            print(Fore.YELLOW + Style.BRIGHT + f"\n[auto_prompt] Running gradient ascent for {img_name} (no cached embed for SHA {img_sha})." + Fore.RESET)

            normalizer = Normalization([0.48145466, 0.4578275, 0.40821073],
                                       [0.26862954, 0.26130258, 0.27577711]).to(device)
            positional_shape = int(premodel.positional_embedding.shape[0])  # 77 for CLIP
            tok = clip.simple_tokenizer.SimpleTokenizer()
            augs = torch.nn.Sequential(kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8)).to(device)

            bests = {1000:'None',1001:'None',1002:'None',1003:'None',1004:'None',1005:'None'}
            prompt_ga = clip.tokenize('''''').numpy().tolist()[0]
            prompt_ga = [i for i in prompt_ga if i not in (0, 49406, 49407)]

            checkin_step = 20
            iterations   = 300
            tokinit      = 4

            lats = Pars(args.ga_batch_size, tokinit, prompt_ga, positional_shape).to(device)
            optim_ga = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])

            do_weird_thing=False
            if args.min_cos_sim:    # Minimize cosine similarity. Whatever that means... Make it *unlike* the image!
                do_weird_thing=True # Get an unknown surprise result of badly defined optimization goal. Fun?!

            _, best_text_embeddings, _, _ = generate_target_text_embeddings(
                img_path, premodel, lats, optim_ga, iterations, checkin_step,
                tokinit, prompt_ga, normalizer, augs, tok, bests, args_local, cache_dir, do_weird_thing
            )
            torch.save(best_text_embeddings, embed_cache_path)

            try:
                raw_opinion = os.path.join(cache_dir, "txtopinion", f"tokens_{img_name}.txt")
                if os.path.exists(raw_opinion):
                    with open(raw_opinion, "r", encoding="utf-8") as f_in, \
                         open(opin_cache_path, "w", encoding="utf-8") as f_out:
                        f_out.write(f_in.read())
            except Exception as e:
                print(Fore.YELLOW + f"[warn] Could not mirror opinion file to keyed path: {e}" + Fore.RESET)

        # combine GA batches (default: 10) → single normalized 1xD
        auto_tfeat = _combine_ga_batches(best_text_embeddings, mode="mean").to(device)
        args.auto_tfeat = auto_tfeat  # used later in generate_visualizations
        tfeat = auto_tfeat            # used now for probing
        args.prompt = ""              # avoid double guidance
        print(Fore.GREEN + Style.BRIGHT + f"[auto_prompt] Prepared 1xD text embedding (from B={args.ga_batch_size} views)." + Fore.RESET)
    else:
        args.auto_tfeat = None

    need_print_stats=True
    # ---- FAST HEAD SCAN (uses tfeat from --prompt or --auto_prompt, if available) ----
    if tfeat is not None and (args_local.auto_head or args_local.auto_head_multi):
        L = layer_range[0]
        cache_dir = os.path.join(steps_folder, "temp")
        os.makedirs(cache_dir, exist_ok=True)
        key = _probe_cache_key(args_local, L, clipmodel, patch_size)
        key_hash = _probe_cache_hash(key)
        cache_path = os.path.join(cache_dir, f"fast_head_scan_L{L}_{key_hash}.json")

        force_flag = bool(getattr(args_local, "force_reprobe", False))

        if force_flag:
            # No checks. Recompute and overwrite.
            need_print_stats=False
            print(Fore.CYAN + "Force re-probe requested; recomputing Fast Head Scan and overwriting cache..." + Fore.RESET)
            block = premodel.visual.transformer.resblocks[L]
            results = _probe_heads_for_prompt(
                model, premodel, block, L, tfeat, args_local, base_pos_embed, patch_size, device=device
            )
            fields = {"head","target_score","itc","spec_sparsity","spec_topk","spec_margin","glob","probe_png"}
            raw_results = [{k: v for k, v in r.items() if k in fields or k=="sim_h10"} for r in results]
            payload = {"key": key, "raw_results": raw_results}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            results_scored  = _recompute_probe_scores(raw_results, args_local)
            results_sorted  = sorted(results_scored, key=lambda r: r["score_final"], reverse=True)
            _print_probe_leaderboard(L, results_sorted)
            if args_local.exclude_head10:
                results_sorted = [r for r in results_sorted if r["head"] != 10]

            if args_local.auto_head:
                selected_heads     = [results_sorted[0]["head"]]
                joint_target_heads = None
                joint_head_weights = None
            else:
                best   = results_sorted[0]["score_final"]
                cutoff = args_local.auto_head_min_frac * best
                top    = [r["head"] for r in results_sorted if r["score_final"] >= cutoff][:args_local.auto_head_multi_n]
                selected_heads     = top
                joint_target_heads = selected_heads
                joint_head_weights = None
                if len(joint_target_heads) > 1:
                    if isinstance(args_local.head_weights, str) and args_local.head_weights.strip().lower() == "auto":
                        score_map = {r["head"]: r["score_final"] for r in results_sorted}
                        ws = [max(0.0, float(score_map.get(h, 0.0))) for h in joint_target_heads]
                        s = sum(ws)
                        joint_head_weights = [w / s for w in ws] if s > 0 else [1.0/len(ws)]*len(ws)
                        print(Fore.GREEN + Style.BRIGHT + f"[auto] head_weights (by score_final): {joint_head_weights}" + Fore.RESET)
                    else:
                        ws = _parse_csv_floats(getattr(args_local, "head_weights", ""))
                        if ws and len(ws) == len(joint_target_heads):
                            s = sum(ws)
                            joint_head_weights = [w / s for w in ws] if s > 0 else [1.0/len(ws)]*len(ws)
                            print(Fore.GREEN + Style.BRIGHT + f"[cfg] head_weights: {joint_head_weights}" + Fore.RESET)
                        elif ws:
                            print(Fore.YELLOW + Style.BRIGHT + f"[warn] --head_weights length {len(ws)} != number of selected heads {len(joint_target_heads)}; using uniform." + Style.RESET_ALL)
                            joint_head_weights = None
        else:
            if os.path.exists(cache_path):
                print(Fore.CYAN + "Using cached Fast Head Scan results." + Fore.RESET)
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                raw_results = payload.get("raw_results", [])
            else:
                print(Fore.CYAN + "Computing Fast Head Scan..." + Fore.RESET)
                block = premodel.visual.transformer.resblocks[L]
                results = _probe_heads_for_prompt(
                    model, premodel, block, L, tfeat, args_local, base_pos_embed, patch_size, device=device
                )
                fields = {"head","target_score","itc","spec_sparsity","spec_topk","spec_margin","glob","probe_png"}
                raw_results = [{k: v for k, v in r.items() if k in fields or k=="sim_h10"} for r in results]
                payload = {"key": key, "raw_results": raw_results}
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

            # Re-score under *current* weights (beta/spec/priors) without recomputing
            # TODO: This just prints twice with or new.            
            if need_print_stats:
                results_scored  = _recompute_probe_scores(raw_results, args_local)
                results_sorted  = sorted(results_scored, key=lambda r: r["score_final"], reverse=True)
                _print_probe_leaderboard(L, results_sorted)

            if args_local.exclude_head10:
                results_sorted = [r for r in results_sorted if r["head"] != 10]
            if args_local.auto_head:
                selected_heads     = [results_sorted[0]["head"]]
                joint_target_heads = None
            else:
                best   = results_sorted[0]["score_final"]
                cutoff = args_local.auto_head_min_frac * best
                top    = [r["head"] for r in results_sorted if r["score_final"] >= cutoff][:args_local.auto_head_multi_n]
                selected_heads     = top
                joint_target_heads = selected_heads

            # per-head weights (if multi)
            if joint_target_heads and len(joint_target_heads) > 1:
                if isinstance(args_local.head_weights, str) and args_local.head_weights.strip().lower() == "auto":
                    score_map = {r["head"]: r["score_final"] for r in results_sorted}
                    ws = [max(0.0, float(score_map.get(h, 0.0))) for h in joint_target_heads]
                    s = sum(ws)
                    joint_head_weights = [w / s for w in ws] if s > 0 else [1.0/len(ws)]*len(ws)
                    print(Fore.GREEN + Style.BRIGHT + f"[auto] head_weights (by score_final): {joint_head_weights}" + Fore.RESET)
                else:
                    ws = _parse_csv_floats(getattr(args_local, "head_weights", ""))
                    if ws and len(ws) == len(joint_target_heads):
                        s = sum(ws)
                        joint_head_weights = [w / s for w in ws] if s > 0 else [1.0/len(ws)]*len(ws)
                        print(Fore.GREEN + Style.BRIGHT + f"[cfg] head_weights: {joint_head_weights}" + Fore.RESET)
                    elif ws:
                        print(Fore.YELLOW + Style.BRIGHT + f"[warn] --head_weights length {len(ws)} != number of selected heads {len(joint_target_heads)}; using uniform." + Style.RESET_ALL)
                        joint_head_weights = None

        print(Fore.GREEN + Style.BRIGHT + f"[auto] Selected heads @L{L}: {selected_heads}\n" + Fore.RESET)
    else:
        joint_target_heads = None
        joint_head_weights = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ============ ACTUAL VISUALIZATION STARTS HERE ============
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    head_range = clamp_range(selected_heads, 0, num_heads-1)
    tv = 1.0
    lr = args_local.lr
    coefficient = args_local.coeff
    steps = args_local.steps
    print_every = 10
    save_every = 10
    saver = bool(args_local.save_intermediate)

    octave_sizes = _parse_csv_ints(args_local.octaves)
    assert len(octave_sizes) >= 3, "Need at least 3 octaves."
    assert all((s % 14) == 0 for s in octave_sizes), f"All octave sizes must be multiples of 14; got {octave_sizes}"

    lr_scales = _parse_csv_floats(args_local.octave_lr_scales) or [1.0] * len(octave_sizes)
    if lr_scales and len(lr_scales) != len(octave_sizes):
        raise ValueError("--octave_lr_scales length must match number of octaves.")

    weights = _parse_csv_floats(args_local.octave_step_weights)
    if weights:
        if len(weights) != len(octave_sizes):
            raise ValueError("--octave_step_weights length must match number of octaves.")
    else:
        if args_local.octave_step_policy == "equal":
            weights = [1.0] * len(octave_sizes)
        else:
            weights = [float(int(s // patch_size) ** 2) for s in octave_sizes]

    wsum = sum(weights) if sum(weights) > 0 else 1.0
    raw = [steps * (w / wsum) for w in weights]
    steps_per_octave = [max(1, int(round(x))) for x in raw]
    diff = steps - sum(steps_per_octave)
    for i in range(abs(diff)):
        j = i % len(steps_per_octave)
        steps_per_octave[j] += 1 if diff > 0 else -1
    assert sum(steps_per_octave) == steps

    tv_scales = _parse_csv_floats(args_local.octave_tv_scales) or [1.0] * len(octave_sizes)
    if tv_scales and len(tv_scales) != len(octave_sizes):
        raise ValueError("--octave_tv_scales length must match number of octaves.")

    grids = [int(s // patch_size) for s in octave_sizes]
    patches = [g * g for g in grids]
    print(Fore.CYAN + f"Octaves: {octave_sizes}" + Fore.RESET)
    print(Fore.CYAN + f"Patch grids: {grids}  (#patches: {patches})" + Fore.RESET)
    print(Fore.CYAN + f"Steps/octave: {steps_per_octave}  (policy={'manual' if _parse_csv_floats(args_local.octave_step_weights) else args_local.octave_step_policy})" + Fore.RESET)
    print(Fore.CYAN + f"TV scales: {tv_scales}" + Fore.RESET)
    print(Fore.CYAN + f"LR scales: {lr_scales}" + Fore.RESET)
    print(Fore.CYAN + f"Warmup frac: {args_local.lr_warmup_frac}, Min LR mult: {args_local.lr_min_mult}" + Fore.RESET)

    if args.attn_move_late:
        premodel = attncopy(premodel, 'visual', from_=attn_from, to_=attn_to)

    sophia_args = dict(
        sophia_k=args_local.sophia_k,
        sophia_mode=args_local.sophia_mode,
        sophia_h_subsample=args_local.sophia_h_subsample,
        sophia_gamma=args_local.sophia_gamma,
        sophia_tau=args_local.sophia_tau,
        sophia_abs_hessian=args_local.sophia_abs_hessian,
        sophia_eps=args_local.sophia_eps,
        u_lowpass=args_local.sophia_u_lowpass,
        u_lowpass_cutoff=args_local.sophia_u_lowpass_cutoff,
        num_hutch=args_local.sophia_num_hutch,
        u_dist=args_local.sophia_u_dist,
    )

    generate_visualizations(
        model, premodel, clipname, layer_range, head_range,
        image_size, tv, lr, steps, print_every, save_every, saver, coefficient,
        octave_sizes=octave_sizes, steps_per_octave=steps_per_octave, 
        tv_scales=tv_scales, lr_scales=lr_scales,
        lr_warmup_frac=args_local.lr_warmup_frac, lr_min_mult=args_local.lr_min_mult,
        grad_rms_norm=args_local.grad_rms_norm,
        base_pos_embed=base_pos_embed, patch_size=patch_size,
        sophia_args=sophia_args,
        neg_heads=neg_heads, neg_alpha=args_local.neg_alpha,
        mlp_channels=mlp_channels, mlp_alpha=args_local.mlp_alpha,
        joint_target_heads=joint_target_heads,
        joint_head_weights=joint_head_weights
    )

    if args.save_video:
        print(Fore.BLUE + Style.BRIGHT + "Attempting to create video..." + Style.RESET_ALL)
        if not args.save_intermediate:
            print(Fore.RED + Style.BRIGHT + "[save_video] ignored because --save_intermediate is not set." + Style.RESET_ALL)
        else:
            max_side = max(octave_sizes)
            candidates = []
            try:
                # recursively find any 'steps' subfolders; keep old pattern too
                for root, dirs, files in os.walk(steps_folder):
                    for d in dirs:
                        p = os.path.join(root, d)
                        if d == "steps":
                            candidates.append(p)
                # Back-compat: old top-level pattern
                for d in os.listdir(steps_folder):
                    p = os.path.join(steps_folder, d)
                    if os.path.isdir(p) and re.match(r"^steps_H\d+_L\d+$", d):
                        candidates.append(p)
            except Exception as e:
                print(Fore.RED + Style.BRIGHT + f"[save_video] could not list '{steps_folder}': {e}" + Style.RESET_ALL)
                candidates = []

            if not candidates:
                print(Fore.YELLOW + Style.BRIGHT + "[save_video] no steps folders found; nothing to render." + Style.RESET_ALL)
            else:
                for temp_folder in sorted(set(candidates), key=_natural_key):
                    ok, msg = _build_video_from_frames(temp_folder, max_side, steps_folder, framerate=args.fps_video)
                    if ok:
                        print(Fore.GREEN + Style.BRIGHT + msg + Style.RESET_ALL)
                    else:
                        print(Fore.RED + Style.BRIGHT + msg + Style.RESET_ALL)

    if args.ablate_registers:
        for h in hooks_layer_11 + hooks_layer_12:
            with contextlib.suppress(Exception):
                h.remove()
    
    print(f"Results saved to the '{steps_folder}' folder.")
    
    print("\n")
    print("┌──────────────────────────────────────┐")
    print("│  All done!  Check the output folder! │")
    print("└──────────────────────────────────────┘\n")

if __name__ == '__main__':

    main()
