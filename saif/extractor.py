"""One forward pass -> all reliability signals (paper Sec. 3, App. A).
Every signal below is computed from the single forward pass that produces the answer.

Signals:
  Output family      : max-softmax confidence over MCQ letters, top-2 margin, entropy
  Logit-lens         : fraction of the last L layers whose implied letter agrees
                       with the final answer (flip-flopping => fragile prediction)
  Attention geometry : entropy of the answer position's attention over image tokens;
                       cross-modal mass (fraction of attention on image tokens)
  Hidden-state drift : standardised distance of final hidden state from the clean
                       reference manifold (mu, sigma fitted on the clean calib split)

Batch size 1, eager attention (required for output_attentions). Precision and
resolution settings are per model; see configs/.
"""
from __future__ import annotations

# ---- silence the recurring PaliGemma-processor warning ---------------------
# PaliGemmaProcessor prints an INFO/WARNING every call when text does not
# contain <image>. Our prepare_inputs already prepends <image> for the
# prefix path, but the warning is emitted by transformers before that logic
# has a chance to be examined for correctness across all model families.
# Filter it out via the transformers logger + Python's warnings module.
import warnings as _warnings
import logging as _logging
_warnings.filterwarnings(
    "ignore",
    message=".*passing both `text` and `images`.*")
_warnings.filterwarnings(
    "ignore",
    message=".*special image tokens in the text.*")
try:
    import transformers as _tf
    _tf.logging.set_verbosity_error()  # silences model-load spam too
except Exception:
    pass

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

from .data import Sample, build_mcq_prompt, LETTERS


# ----------------------------------------------------------------------------- model
class VLMWrapper:
    def __init__(self, name: str, device: str = "cuda:0", dtype: str = "float16",
                 attn_implementation: str = "eager",
                 trust_remote_code: bool = False,
                 answer_prefix: str = "",
                 max_patches: int = None,
                 max_pixels: int = None):
        self.answer_prefix = answer_prefix
        import transformers
        from transformers import AutoProcessor
        # Class name varies across transformers versions:
        #   newer: AutoModelForImageTextToText (canonical since ~4.46)
        #   older: AutoModelForVision2Seq (removed in newest majors)
        #   last resort: the concrete Qwen2.5-VL class
        ModelCls = None
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                         "Qwen2_5_VLForConditionalGeneration"):
            ModelCls = getattr(transformers, cls_name, None)
            if ModelCls is not None:
                print(f"[model] using transformers.{cls_name} "
                      f"(transformers {transformers.__version__})")
                break
        if ModelCls is None:
            raise ImportError(
                f"No compatible VLM loader class in transformers "
                f"{transformers.__version__}; install a version per "
                f"requirements.txt (>=4.49 recommended)")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=trust_remote_code)
        # Cap dynamic image tiling to bound attention-matrix memory (needed for
        # InternVL-style tiling under fp32 + eager attention on 16 GiB GPUs).
        # Leaves the model default untouched when max_patches is None, so other
        # models (Qwen, SmolVLM) are unaffected.
        if max_patches is not None:
            ip = getattr(self.processor, "image_processor", None)
            if ip is not None and hasattr(ip, "max_patches"):
                ip.max_patches = int(max_patches)
                print(f"[extractor] capped image_processor.max_patches = {max_patches}")
            else:
                print(f"[extractor] WARNING: max_patches requested ({max_patches}) "
                      f"but image_processor has no such attribute; ignoring.")
        # Store for the hard pre-resize in prepare_inputs (the reliable cap).
        self.max_pixels = int(max_pixels) if max_pixels is not None else None
        # Cap Qwen-style DYNAMIC RESOLUTION via max_pixels. Qwen2/2.5-VL expand
        # a large image into thousands of visual tokens with no default ceiling,
        # so a single high-resolution image (e.g. DisasterVQA social-media
        # photos) can blow the attention matrix. Setting max_pixels caps the
        # token count. None leaves the default untouched (other models/benchmarks
        # unaffected). 1003520 = 1280*784 is a sane ceiling that keeps most
        # images intact while bounding the worst case.
        if max_pixels is not None:
            ip = getattr(self.processor, "image_processor", None)
            capped = False
            # Qwen2-VL has NO max_pixels attribute; its pixel-AREA budget lives
            # in size.longest_edge (default ~12.8M). Set it directly. size may be
            # a dict or a SizeDict object, so handle both.
            sz = getattr(ip, "size", None) if ip is not None else None
            if sz is not None:
                try:
                    if isinstance(sz, dict):
                        sz["longest_edge"] = int(max_pixels)
                        capped = True
                    elif hasattr(sz, "longest_edge"):
                        sz.longest_edge = int(max_pixels)   # SizeDict attribute
                        capped = True
                except Exception:
                    pass
            # older/other processors: a real max_pixels attribute
            if ip is not None and hasattr(ip, "max_pixels"):
                ip.max_pixels = int(max_pixels); capped = True
            if hasattr(self.processor, "max_pixels"):
                self.processor.max_pixels = int(max_pixels); capped = True
            print(f"[extractor] processor pixel cap "
                  f"{'set via size.longest_edge = '+str(max_pixels) if capped else 'NOT settable (relying on pre-resize)'}")
        self.model = ModelCls.from_pretrained(
            name,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,  # eager => attentions available
            trust_remote_code=trust_remote_code,
        ).to(device).eval()
        self.tokenizer = self.processor.tokenizer

        # Token ids of the answer letters. We take the ids of "A".."D" both bare
        # and with a leading space and merge the logits over variants (Eq. A1).
        self.letter_ids = {}
        for L in LETTERS:
            ids = set()
            for v in (L, " " + L):
                enc = self.tokenizer.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
            assert ids, f"could not find single-token id for letter {L}"
            self.letter_ids[L] = sorted(ids)

        # Image placeholder token id (Qwen2.5-VL: <|image_pad|>); used to locate
        # the image-token span for the attention-geometry signals.
        # ---- resolve logit-lens submodules across transformers versions ----
        # Older layouts: model.model.norm ; newer refactor puts the text
        # decoder under model.model.language_model.norm. Try known paths,
        # then fall back to the LAST module named "*.norm" in the tree.
        self.lm_head = self.model.get_output_embeddings() \
            or getattr(self.model, "lm_head", None)
        self.final_norm, self.norm_path = None, None
        for path in ("model.norm", "model.language_model.norm",
                     "language_model.model.norm", "model.text_model.norm"):
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                self.final_norm, self.norm_path = obj, path
                break
            except AttributeError:
                continue
        if self.final_norm is None:
            cands = [(n, m) for n, m in self.model.named_modules()
                     if n.endswith(".norm")]
            if cands:
                self.norm_path, self.final_norm = cands[-1]
        if self.final_norm is None or self.lm_head is None:
            raise AttributeError(
                "Could not resolve final norm / lm_head for this transformers "
                "version; inspect model.named_modules() and extend the path "
                "list in extractor.py")
        print(f"[lens] resolved final norm at '{self.norm_path}', lm_head OK")

        # Image placeholder token id — different across VLM families:
        #   Qwen2.5-VL:  <|image_pad|>
        #   InternVL:    <IMG_CONTEXT>
        #   LLaVA:       <image>
        #   Idefics3:    <fake_token_around_image>  (edge case; may not need this)
        # Order: (1) model.config.image_token_id if set, (2) known token names,
        # (3) whatever appears >100 times in a probe forward's input_ids (very
        # last-resort discovery). Print the resolved value so it's obvious.
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", None)
        if self.image_token_id is None:
            for tok in ("<|image_pad|>", "<IMG_CONTEXT>", "<image>",
                        "<img>", "<image_soft_token>"):
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != self.tokenizer.unk_token_id:
                    self.image_token_id = tid
                    print(f"[model] image token resolved from tokenizer: "
                          f"'{tok}' -> id {tid}")
                    break
        if self.image_token_id is None:
            raise RuntimeError(
                "Could not resolve image_token_id. Set it explicitly in "
                "the config or extend extractor.py's known-name list.")
        else:
            print(f"[model] image_token_id = {self.image_token_id}")

    def prepare_inputs(self, sample: Sample, image):
        """Build the MCQ prompt in whichever style the model wants.

        Chat-templated (Qwen2.5-VL, LLaVA-Next, InternVL, Idefics3, ...):
            uses processor.apply_chat_template with image + text turns.
        Prefix-completion (PaliGemma, and other non-chat VLMs):
            passes the text as a raw prefix. PaliGemma prepends its own
            <image> tokens and BOS via the processor when text is a plain
            string, so we just format the prompt directly.
        """
        # HARD pixel cap: resize the PIL image BEFORE the processor sees it.
        # This is the reliable way to bound Qwen2-VL's dynamic-resolution token
        # count -- setting processor.image_processor.max_pixels is unreliable
        # across versions (the value is read from size/call-time kwargs, not the
        # attribute), so we cap the actual image instead. Only downscales images
        # that EXCEED the cap; smaller images (the majority) pass untouched, so
        # natural degradation is preserved except for the largest photos.
        mp = getattr(self, "max_pixels", None)
        if mp is not None and image is not None:
            w, h = image.size
            if w * h > mp:
                import math
                scale = math.sqrt(mp / float(w * h))
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                try:
                    from PIL import Image as _Image
                    image = image.resize((new_w, new_h), _Image.BICUBIC)
                except Exception:
                    image = image.resize((new_w, new_h))

        prompt_text = build_mcq_prompt(sample)
        answer_prefix = getattr(self, "answer_prefix", "")
        has_chat = getattr(self.processor, "chat_template", None) is not None
        if has_chat:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Append the model-specific answer prefix (e.g. " Answer:" for
                # SmolVLM) so the very next token is the answer LETTER, which
                # is the position the extractor reads signals from.
                if answer_prefix:
                    text = text + answer_prefix
                inputs = self.processor(text=[text], images=[image], return_tensors="pt")
                return {k: v.to(self.device) for k, v in inputs.items()}
            except (ValueError, TypeError):
                # Some processors declare a chat_template attr but reject the
                # image-in-content format -- fall through to prefix style.
                pass

        # Prefix-completion path: PaliGemma-style. PaliGemma expects
        # <image> tokens at the START of the text; if we don't add them,
        # the processor infers and emits a warning per call (which floods
        # the log across 42k samples). Prefixing "<image>" once is enough
        # for one image; the processor expands it to the correct number of
        # image-embedding placeholders. The trailing "\nAnswer: " nudges
        # the model to produce the answer letter as the next token.
        # Prefix-completion models: use the config's answer_prefix if set,
        # else default to "\nAnswer: " which worked for PaliGemma.
        prefix = answer_prefix or "\nAnswer: "
        inputs = self.processor(
            text=["<image>" + prompt_text + prefix],
            images=[image], return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}


# ----------------------------------------------------------------------------- record
@dataclass
class SignalRecord:
    index: int
    corruption: str
    severity: int
    # answer + label
    pred_letter: str
    gt_letter: str
    correct: int
    # output family
    conf: float            # max softmax over letters
    margin: float          # top1 - top2 prob
    entropy: float         # entropy over letter distribution
    # logit lens
    lens_agree_frac: float # fraction of last L layers agreeing with final answer
    # attention geometry
    attn_img_entropy: float  # entropy of answer-pos attention over image tokens (normalised)
    attn_img_mass: float     # cross-modal mass: fraction of attention on image tokens
    # drift (raw components; standardised distance computed later vs clean stats)
    # multiple hidden-state variants captured side-by-side (see extract_signals);
    # DriftReference is fit per-variant on the clean calibration slice.
    hiddens: dict | None = None

    def row(self) -> dict:
        d = asdict(self)
        d.pop("hiddens")
        return d


# ----------------------------------------------------------------------------- extract
@torch.no_grad()
def extract_signals(vlm: VLMWrapper, sample: Sample, image,
                    corruption: str, severity: int,
                    last_L: int = 8, attention_layers=(-1, -2)) -> SignalRecord:
    inputs = vlm.prepare_inputs(sample, image)
    out = vlm.model(**inputs, output_hidden_states=True, output_attentions=True)

    # Answer position = last prompt token; its next-token distribution is the letter.
    pos = inputs["input_ids"].shape[1] - 1
    logits = out.logits[0, pos].float()                      # [vocab]

    # ---- output family, restricted to the letters actually offered
    letters = [L for L in LETTERS if L in sample.options]
    letter_logits = torch.stack([
        torch.logsumexp(logits[vlm.letter_ids[L]], dim=0) for L in letters
    ])                                                        # merge " A"/"A" variants
    probs = F.softmax(letter_logits, dim=0)
    sorted_p, _ = probs.sort(descending=True)
    conf = float(sorted_p[0])
    margin = float(sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    pred_letter = letters[int(probs.argmax())]

    # ---- logit lens over the last L layers
    # hidden_states: tuple of (num_layers+1) tensors [1, seq, d]. Apply the final
    # norm + lm_head to each late layer's hidden state at the answer position and
    # ask which letter it implies. Submodules are resolved once at
    # load time (vlm.final_norm / vlm.lm_head) across transformers versions.
    norm = vlm.final_norm
    lm_head = vlm.lm_head
    hs = out.hidden_states
    agree = 0
    for layer in range(len(hs) - last_L, len(hs)):
        h = hs[layer][0, pos]
        layer_logits = lm_head(norm(h.unsqueeze(0))).squeeze(0).float()
        ll = torch.stack([
            torch.logsumexp(layer_logits[vlm.letter_ids[L]], dim=0) for L in letters
        ])
        if letters[int(ll.argmax())] == pred_letter:
            agree += 1
    lens_agree_frac = agree / last_L

    # ---- attention geometry (answer position -> image tokens)
    img_mask = (inputs["input_ids"][0] == vlm.image_token_id)   # [seq]
    n_img = int(img_mask.sum())
    attn_entropy, attn_mass = 1.0, 0.0   # defaults if no image tokens found
    if n_img >= 1:
        rows = []
        for li in attention_layers:
            a = out.attentions[li][0, :, pos, :].float()        # [heads, seq]
            rows.append(a.mean(dim=0))                          # head-average
        a = torch.stack(rows).mean(dim=0)                       # [seq]
        attn_mass = float(a[img_mask].sum())                    # cross-modal mass
        if n_img == 1:
            # single-token image (Qwen2.5-VL dynamic tiling): entropy over a
            # 1-element distribution is 0; call it "perfectly concentrated"
            # rather than dividing by log(1)==0 and producing NaN.
            attn_entropy = 0.0
        else:
            p_img = a[img_mask]
            p_img = p_img / p_img.sum().clamp_min(1e-12)
            h_ent = float(-(p_img * p_img.clamp_min(1e-12).log()).sum())
            attn_entropy = h_ent / np.log(n_img)                # in [0,1]: 1 = smeared

    # ---- hidden-state variants for drift (three, so the analysis picks the
    # placement empirically rather than by prior). All are pulled from the
    # same forward pass, so the extra cost is a couple of pooled tensors.
    #   answer_pos  : last layer, answer-position token (inversely associated
    #                 with correctness; paper Sec. 4.4)
    #   img_mean    : last layer, mean over image-token positions (the
    #                 empirically motivated fallback: images-tokens are where
    #                 visual corruption lives; the answer-pos hidden state is
    #                 dominated by question semantics in a decoder VLM)
    #   img_mean_mid: same pooling at the middle layer (a mid-layer variant
    #                 in case the final layer is already too linguistic)
    mid = len(hs) // 2
    hiddens = {"answer_pos": hs[-1][0, pos].float().cpu().numpy()}
    if n_img >= 1:
        hiddens["img_mean"] = hs[-1][0, img_mask].float().mean(dim=0).cpu().numpy()
        hiddens["img_mean_mid"] = hs[mid][0, img_mask].float().mean(dim=0).cpu().numpy()
    else:
        # no image tokens found (should never happen after preflight); fall
        # back to answer-pos so the array shape is preserved for downstream.
        hiddens["img_mean"] = hiddens["answer_pos"]
        hiddens["img_mean_mid"] = hiddens["answer_pos"]

    del out
    return SignalRecord(
        index=sample.index, corruption=corruption, severity=severity,
        pred_letter=pred_letter, gt_letter=sample.answer,
        correct=int(pred_letter == sample.answer),
        conf=conf, margin=margin, entropy=entropy,
        lens_agree_frac=lens_agree_frac,
        attn_img_entropy=attn_entropy, attn_img_mass=attn_mass,
        hiddens=hiddens,
    )


# DriftReference lives in saif.drift (torch-free); re-exported for convenience
from .drift import DriftReference  # noqa: E402,F401
