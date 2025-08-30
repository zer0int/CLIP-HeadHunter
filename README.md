# CLIP HeadHunter ~ A Visual Bias Explorer 🔎🤖🎯 #

-------
Attention Head Max Visualization to find, rank, and visualize heads; map bias; see what a CLIP 'sees'.
CLIP-HeadHunter visualizes what individual or joint CLIP attention heads "want" by optimizing an image to maximally activate them.
It auto-probes and ranks heads with prompt/image signals: Human or Auto-prompt / CLIP self-prompting (gradient ascent text embeddings for cos sim with image embedding); broad/neg probes; MLP-guided acts*grad, and more.
Start from Gaussian noise, pool queries in several ways, shape attention via priors/entropy, and export steps/videos. Plus: A companion JSON for managing reruns (the code has ~100 argparse args).
Use embeddings upscale (448,448) to surface circuits and dataset/model bias by seeing in detail where heads attend and what pictures they conjure.

-------
- Use `clip_attn_explorer.py --load_json "path/to/settings.json"` to get going; see `get_started` for example configs.
- See `clip_attn_explorer.py --help` or (better) check out the code for all info; cheat sheet below!
- Auto-dumps .json files next to your visualization images for easy reproduction.
- Save frames (every 10 steps): `--save_intermediate` and `--save_video` (requires ffmpeg to be callable)
- Use --deepdream to prime on an image and DeepDream. Example in `get_started` folder.
- Example images in `image_sets_attn`. All PG13, but CLIP's interpretation will not be. See ⚠️ below.
- Uses a modified version of [OpenAI/CLIP](https://github.com/openai/CLIP) to expose attn QKV / weights.
- Can load .safetensors. Used in examples: [My model](https://huggingface.co/zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14) - direct [download](https://huggingface.co/zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14/resolve/main/ViT-L-14-KO-LITE-FULL-model-OpenAI-format.safetensors?download=true).
- Optimizer SophiaViz: Based on [github.com/Liuhong99/Sophia](https://github.com/Liuhong99/Sophia)
- Optimized for ViT-L/14 and known 'Global information' Head 10; [see details](https://github.com/zer0int/CLIP-test-time-registers).
- Have Ampere+ (NVIDIA) GPU (RTX 4090+)? Set `--fast` -> matmul precision to 'medium'.
- CUDA OOM? Remove `448` from `--octaves`. Note: Some option, like `--head_agg attn_weighted`, require more VRAM than e.g. `mean`.
- I wasn't able to make the cache loading deterministic on GPU. Use `--deterministic --force_reprobe` for determinism.
-------
### Most important arguments to pass, in a nutshell:
👇 Scroll all the way down for cheat sheet 👇

Feed an image, get a 'CLIP opinion', guide by text-image, auto-select best matching heads:
```
--mlp_saliency_img "path/to/img" --auto_prompt --auto_head_multi
# Note: --auto_prompt overrides --prompt
```
Use your own prompt, provide extra head steering (see cheat sheet for details):
```
--prompt "a cat" --probe_broad_prompts "a cat|a feline|a word" --probe_neg_prompts "a skyline|a person|a dog"
```
The most important argument. If you use defaults but get noise because you chose a different `--layer_range` (single target layer! for now): START HERE:
```
--head_agg attn_weighted   # for early layers. sharp. wide, noisy (later layers).
--head_agg center          # best for late (10+) layers.
--head_agg mean            # a median choice. can be brittle and too broad.
--head_agg topk            # if image only, superior result possible, but noisy ruin also possible.
```
## ⚠️ Content & Safety Note
This toolkit can surface and visualize harmful associations and biases learned by the model; see the [OpenAI CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md) for details. Even when starting from ordinary, SFW images or neutral prompts, visualizations may unexpectedly depict sexual or violent content (including explicit material), demeaning stereotypes, or sensitive historical references, without warning. These outputs reflect dataset/model biases and do not represent the authors’ views or endorsements. The codebase is provided as-is for research purposes only, including exploring bias and typographic/adversarial vulnerabilities. Use in controlled settings, review results before sharing, and follow applicable laws and policies.
- Recommended paper / see also: [What do we learn from inverting CLIP models?](https://arxiv.org/abs/2403.02580)
-------
### Typographic attack example, same settings, two models:
<img width="953" height="336" alt="example_Hmulti_7-11_L2_336" src="https://github.com/user-attachments/assets/4758c489-83cd-4ef3-9759-f5fcb6186a3b" />

### 👉 You can find this example / settings in the `get_started` folder

https://github.com/user-attachments/assets/fea4992c-6509-4f4f-9cea-d5f33e491ee8

### Face in coffee = emoji jumping from cup
<img width="2263" height="448" alt="example-cup-face" src="https://github.com/user-attachments/assets/0c660daf-0061-4f15-ba93-f554365bb0c4" />

### Rat with 'bat' writting on it -> rat holds a bat.
<img width="2260" height="448" alt="example-attack" src="https://github.com/user-attachments/assets/198cd341-a088-4f7f-b593-ab11cbfa195f" />

### CLIP can write, too. Say 'hi clip!' some time. ;)
<img width="1408" height="467" alt="combo" src="https://github.com/user-attachments/assets/60e51c66-8ebf-456b-9b83-ccac8f4adb9e" />

### Classic typographic attack: Granny smith with 'iPod' stuck to it.
<img width="2051" height="927" alt="example-ipod" src="https://github.com/user-attachments/assets/4451126d-9f3e-490e-b70f-4a410a5f2fdf" />

-------
# CHEAT SHEET 🤓
Most important settings only. For all, refer to --help or [see code](https://github.com/zer0int/CLIP-HeadHunter/blob/bd7c7bfdc6adf206d8d713ef733d9a5a144d044a/clip_attn_explorer.py#L57).

## --head_agg options
| --head_agg ‼️     | Definition (token space)             | When to use                                     |
| ----------------- | ------------------------------------- | ----------------------------------------------- |
| `mean`            | Average over all patch tokens         | Balanced, least opinionated                     | 
| `topk`            | Mean of top-k token scores            | Enforce localized peaks / sparse saliency       | 
| `center`          | Gaussian-weighted center mass         | Enforce central saliency (portrait-like priors) | 
| `attn_weighted`   | \sum_i \text{token\_score}_i · Aw_i   | Couple token energy to current attention        | 

## --attn_prior options
| --attn_prior    | π (target distribution)                 | Intuition                     |
| --------------- | --------------------------------------- | ----------------------------- |
| `none`          | —                                       | No KL term                    | 
| `center`        | 2-D Gaussian at image center            | Peaky center bias             | 
| `ring`          | Difference of wide and narrow Gaussians | Annular focus (avoid center)  | 
| `diag`          | Two diagonal Gaussians                  | Bias toward diagonals         | 
| `head10`        | Smoothed head-10 attention              | Borrow global-router’s layout | 

## --query_mode options
| --query_mode | Query set used for `Aw` (attention over **keys**)                        | Notes                                   |
| ------------ | ------------------------------------------------------------------------ | ----------------------------------------|
| `""` (empty) | **Defaults** to `cls` if a text embedding exists, else `mean`            | Default behavior, auto select.          |
| `cls`        | Only the CLS query (Q=1)                                                 | Typical when aligning to text; 'focus'. |
| `mean`       | Mean over all available queries                                          | Stable default when no text embedding   |
| `topk`       | Mean over the top-k **query** positions selected by current token scores | `k = --head_topk_patches` (0 ⇒ auto).  |

## auto_head settings
| Setting                            | Effect                                       | Notes                                                                           |
| ---------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------- |
| `--auto_head_multi`                | Selects top-K heads by probe score           | K = `--auto_head_multi_n`; optional score-threshold via `--auto_head_min_frac`  |
| `--head_weights auto`              | Weights heads ∝ probe `score_final`          | Normalized; uniform if scores unavailable                                       |
| Explicit `--head_weights a,b,c`    | Fixed per-head weights                       | Mismatch length ⇒ uniform                                                       |
| Competitor penalty                 | Excludes selected heads from competitors     | `comp_alpha_heads` scales penalty; `comp_k_heads>0` uses top-K competitors only |
| `--head_agg attn_weighted` (joint) | Weighted token score uses head-weighted `Aw` | Needs `attn_probs`; weights are your joint `head_weights`                       |

## text prompt / embeddings
| Flag                    | MLP saliency (grad×act)     | Auto-head probe (ranking)                                                                  | Final optimization (loss)                            |
| ----------------------- | --------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| `--prompt`              | **Yes** (enables grad×act)  | **Yes**: adds PromptAlignmentLoss, sets probe query to CLS, mixes ITC via `auto_head_beta` | **Yes**: PromptAlignmentLoss with `text_coeff`       |
| `--auto_prompt`         | N/A (prepares `auto_tfeat`) | **Yes**: same as `--prompt` but using learned `auto_tfeat` (CLIP 'opinion' about image)    | **Yes**: uses `auto_tfeat`; `args.prompt` is cleared |
| `--probe_broad_prompts` | No                          | **Yes**: contributes to `spec_margin` term → affects head ranking                          | No                                                   |
| `--probe_neg_prompts`   | No                          | **Yes**: contributes to `spec_margin` term → affects head ranking                          | No                                                   |

## misc scenarios prompt / image
| Scenario                                            | Text embedding present? | Default `--query_mode` | Auto-head available?              | MLP saliency uses grad×act?                                     |
| --------------------------------------------------- | ----------------------- | ---------------------- | --------------------------------- | --------------------------------------------------------------- |
| No prompt, no image                                 | **No**                  | `mean`                 | **No** (sticks to `--head_range`) | **No**                                                          |
| Prompt only (`--prompt`)                            | **Yes**                 | `cls`                  | **Yes**                           | If `mlp_saliency_img` **and** `mlp_saliency_use_grad` → **Yes** |
| Auto-prompt only (`--auto_prompt` + image path)     | **Yes** (from GA)       | `cls`                  | **Yes**                           | N/A for saliency unless you also set `mlp_saliency_img`         |
| Saliency image only (`--mlp_saliency_img`, no text) | **No**                  | `mean`                 | **No**                            | **No** (grad needs text)                                        |
| Saliency image + prompt                             | **Yes**                 | `cls`                  | **Yes**                           | **Yes** (grad×act)                                              |

## high level: query_mode, attn_prior, head_agg
| Arg                                                       | Purpose                             | Auto-head probe (ranking)                                               | Final optimization (loss shaping)                                              | Dependencies / Notes                                                                                           |
| --------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `--query_mode` ∈ {`""`,`cls`,`mean`,`topk`}               | Which queries to read attention     | Default: `cls` if text; else `mean`. Affects `Aw`.                      | Same; `Aw` for loss (attn\_weighted, KL, entropy).                             | `topk` needs `--head_topk_patches` (0→auto). Grid seq: drop CLS on **keys**; if `Q==Seq`, also on **queries**. |
| `--attn_prior` ∈ {`none`,`center`,`ring`,`diag`,`head10`} | KL target π over keys               | No effect.                                                              | `KL(p‖π)` if `--attn_prior_lambda>0` **and** `Aw`. `head10`: smoothed head-10. | `'center'` default inert unless λ>0. Needs `attn_probs` for `Aw`/π; otherwise skipped.                         |
| `--head_agg` ∈ {`mean`,`topk`,`center`,`attn_weighted`}   | Pool token scores → head score      | Defines `target_score`.                                                 | Defines target term in attention loss.                                         | `center`: Gaussian over grid; `attn_weighted` requires `Aw`; `topk` uses `--head_topk_patches` (0→auto).       |
| `--use_head10_prior`                                      | Penalize head-10-like heads (probe) | `score_adj = base − w₁·sim_h10 − w₂·glob`; `--exclude_head10` optional. | No effect.                                                                     | Probe-only; biases toward sparse/specific, non-router heads.                                                   |
