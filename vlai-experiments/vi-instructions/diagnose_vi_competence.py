#!/usr/bin/env python3
"""Tách 'nén tốt hơn' khỏi 'hiểu tốt hơn' cho backbone tiếng Việt.

Bối cảnh: ladder BPC d0=4.4098 -> d100=3.3183 trông như dose-response đẹp, nhưng
`stock` (backbone tiếng Anh, chưa từng pretrain tiếng Việt) đạt 3.34 trên cùng
corpus. Nghĩa là 98% khoảng cách d0->d100 chỉ là hồi phục sau khi mở rộng vocab
49.280 -> 57.344, và chỉ 0.0217 b/char (2%) là năng lực vượt trên `stock`.

Hai lệnh con trả lời hai câu hỏi khác nhau:

  bpc        Lợi thế của d100 có tập trung vào 8.064 token tiếng Việt mới không?
             Nếu có -> vẫn có câu chuyện thật, chỉ là câu chuyện khác.
             Nếu không -> pretrain không thêm được gì; phải đổi cách viết bài.

  downstream BPC bị chi phối bởi hiệu quả tokenizer. Lệnh này đo accuracy trên
             một task tiếng Việt hạ nguồn, tách "nén tốt" khỏi "hiểu tốt".

Cách so sánh giữa hai vocab khác nhau: quy bits của mỗi token về từng KÝ TỰ nó
phủ (chia đều), rồi mới cắt theo tập ký tự cần đo. Đây là cùng giả định đã ngầm
nằm trong việc dùng bits-per-character để so vocab 49.280 với 57.344.

Ví dụ:
  python diagnose_vi_competence.py bpc \
      --stock outputs/backbones/... hoặc HF id của backbone stock \
      --d100  outputs/backbones/vi_dose_100 \
      --corpus vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt

  python diagnose_vi_competence.py downstream \
      --stock ... --d100 ... --data vietnamese_sentiment.jsonl \
      --labels "tiêu cực,trung tính,tích cực" \
      --prompt "Câu sau có sắc thái gì? {text}\nTrả lời:"
"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np


# ----------------------------------------------------------------- core (pure)
def char_bits_from_tokens(n_chars: int, offsets, bits) -> np.ndarray:
    """Rải bits của mỗi token đều lên các ký tự nó phủ."""
    cb = np.zeros(n_chars, dtype=np.float64)
    for (a, b), bt in zip(offsets, bits):
        if b > a:
            cb[a:b] += bt / (b - a)
    return cb


def char_covered_mask(n_chars: int, offsets) -> np.ndarray:
    m = np.zeros(n_chars, dtype=bool)
    for a, b in offsets:
        if b > a:
            m[a:b] = True
    return m


def new_token_char_mask(n_chars: int, offsets, ids, first_new_id: int) -> np.ndarray:
    """CẢNH BÁO: vùng này do tokenizer của d100 định nghĩa => THIÊN LỆCH CHỌN MẪU.

    Nó chọn đúng những ký tự mà d100 có token chuyên dụng, tức đúng nơi d100 nén
    tốt nhất, và đẩy phần byte-fallback của d100 sang vùng còn lại. Hai model có
    năng lực HOÀN TOÀN NHƯ NHAU vẫn sinh ra dấu hiệu '+ ở vùng mới, - ở vùng cũ,
    tổng bằng 0' dưới phép phân vùng này. Dùng để mô tả, KHÔNG dùng để kết luận
    'backbone có học được tiếng Việt hay không' — dùng script_char_masks().
    """
    m = np.zeros(n_chars, dtype=bool)
    for (a, b), i in zip(offsets, ids):
        if i >= first_new_id and b > a:
            m[a:b] = True
    return m


def strip_diacritics(w: str) -> str:
    import unicodedata
    BASE = {"ă":"a","â":"a","đ":"d","ê":"e","ô":"o","ơ":"o","ư":"u"}
    out = []
    for ch in unicodedata.normalize("NFD", w):
        if unicodedata.combining(ch):
            continue
        out.append(BASE.get(ch.lower(), ch.lower()) if ch.lower() in BASE else ch)
    return "".join(out)


def build_minimal_pairs(lines, max_items=400, n_distract=4, seed=0):
    """Sinh bài toán lựa chọn cưỡng bức: che một từ có dấu, đưa các biến thể khác dấu.

    Tại sao cần: mọi phép so bits/ký-tự giữa stock (byte-fallback trên tiếng Việt) và
    d100 (có token cấp từ) đều bị confound — stock tốn nhiều bits hơn trên ký tự
    nhiều byte kể cả khi hai model gán CÙNG xác suất cho cùng chuỗi. Bài toán lựa
    chọn cho kết quả 0/1: mỗi model chấm mọi phương án bằng tokenizer của CHÍNH NÓ,
    ta chỉ hỏi nó có chọn đúng không. Không mẫu số, không đơn vị => bất biến tokenizer.
    """
    import random, re, unicodedata
    rng = random.Random(seed)
    # nhóm các từ cùng dạng-không-dấu xuất hiện trong corpus
    forms = {}
    for ln in lines:
        for w in re.findall(r"[^\W\d_]+", ln, flags=re.UNICODE):
            forms.setdefault(strip_diacritics(w), set()).add(w)
    items = []
    for ln in lines:
        for mm in re.finditer(r"[^\W\d_]+", ln, flags=re.UNICODE):
            w = mm.group()
            has_vi = any(c in "ăâđêôơưĂÂĐÊÔƠƯ" or
                         any(unicodedata.combining(x) for x in unicodedata.normalize("NFD", c))
                         for c in w)
            if not has_vi or len(w) < 2:
                continue
            alts = [f for f in forms.get(strip_diacritics(w), ()) if f != w]
            bare = strip_diacritics(w)
            pool = [a for a in alts if a != bare] + ([bare] if bare != w else [])
            if len(pool) < 2:
                continue
            rng.shuffle(pool)
            items.append({"prefix": ln[:mm.start()], "suffix": ln[mm.end():],
                          "correct": w, "options": [w] + pool[:n_distract]})
    rng.shuffle(items)
    return items[:max_items]


def script_char_masks(text: str) -> dict:
    """Phân vùng ký tự KHÔNG phụ thuộc tokenizer nào — cùng một mask cho cả hai model.

    'vi_marked'  : chữ cái có dấu tiếng Việt (ă â đ ê ô ơ ư + mọi dấu thanh)
    'latin_plain': chữ cái Latin không dấu
    'glue'       : khoảng trắng, dấu câu, chữ số, còn lại
    """
    import unicodedata
    VI_BASE = set("ăâđêôơưĂÂĐÊÔƠƯ")
    vi = np.zeros(len(text), bool); la = np.zeros(len(text), bool)
    for i, ch in enumerate(text):
        if not ch.isalpha():
            continue
        d = unicodedata.normalize("NFD", ch)
        if ch in VI_BASE or d[0] in VI_BASE or any(unicodedata.combining(c) for c in d):
            vi[i] = True
        else:
            la[i] = True
    return {"vi_marked": vi, "latin_plain": la, "glue": ~(vi | la)}


# ------------------------------------------------------------------ model I/O
def load(path, device):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if not tok.is_fast:
        sys.exit(f"{path}: cần fast tokenizer để lấy offset_mapping")

    def _load(cls):
        try:
            return cls.from_pretrained(path, dtype=torch.float32)
        except TypeError:      # transformers < 4.56 dùng tên cũ
            return cls.from_pretrained(path, torch_dtype=torch.float32)

    # SmolVLM2 KHONG nap duoc bang AutoModelForCausalLM tren may nay
    # ("Unrecognized configuration class ... SmolVLMConfig"). AutoModelForImageTextToText
    # nap duoc, va forward voi input_ids-only van tra logits tren vocab van ban.
    try:
        model = _load(AutoModelForCausalLM)
    except (ValueError, KeyError, OSError):
        model = _load(AutoModelForImageTextToText)
    return tok, model.to(device).eval()


def line_token_bits(tok, model, line, device):
    """-> (offsets, ids, bits) cho MỌI token của dòng.

    Chèn BOS làm ngữ cảnh cho token đầu. Nếu bỏ token đầu thay vì chèn BOS, mẫu số
    co lại khác nhau giữa hai tokenizer (d100 ~13 token/dòng, stock ~34) và có thể
    ĐẢO DẤU kết luận. Với BOS, coverage = 1.0000 cho cả hai model.
    """
    import torch
    enc = tok(line, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if not ids:
        return [], [], []
    bos = tok.bos_token_id
    if bos is None:
        bos = tok.eos_token_id if tok.eos_token_id is not None else ids[0]
    with torch.no_grad():
        t = torch.tensor([[bos] + list(ids)], device=device)
        logits = model(t).logits[0].float()
        logp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = t[0, 1:]
        nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)   # nats, cho MỌI token thật
    bits = (nll / math.log(2)).cpu().numpy()
    return offs, ids, bits


def accumulate(tok, model, lines, device, first_new_id=None, label=""):
    """-> dict với char_bits, covered mask, new-token mask, ghép toàn corpus."""
    CB, COV, NEW, NCH = [], [], [], 0
    for n, line in enumerate(lines):
        offs, ids, bits = line_token_bits(tok, model, line, device)
        L = len(line)
        CB.append(char_bits_from_tokens(L, offs, bits))
        COV.append(char_covered_mask(L, offs))
        NEW.append(new_token_char_mask(L, offs, ids, first_new_id)
                   if first_new_id is not None else np.zeros(L, bool))
        NCH += L
        if n % 500 == 0:
            print(f"  [{label}] {n}/{len(lines)} dòng", flush=True)
    return dict(char_bits=np.concatenate(CB), covered=np.concatenate(COV),
                new=np.concatenate(NEW), n_chars=NCH)


# ----------------------------------------------------------------- subcommands
def cmd_bpc(a):
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lines = [l.rstrip("\n") for l in open(a.corpus, encoding="utf-8") if l.strip()]
    print(f"corpus: {len(lines)} dòng, {sum(map(len, lines))} ký tự\ndevice: {dev}\n")

    tk_s, md_s = load(a.stock, dev)
    tk_d, md_d = load(a.d100, dev)
    first_new = a.first_new_id if a.first_new_id else len(tk_s)
    n_new_ids = sum(1 for i in tk_d.get_vocab().values() if i >= first_new)
    print(f"biên token mới: id >= {first_new}  ({n_new_ids} token mới trong tokenizer d100)\n")

    D = accumulate(tk_d, md_d, lines, dev, first_new_id=first_new, label="d100")
    S = accumulate(tk_s, md_s, lines, dev, first_new_id=None,      label="stock")

    # Tập ký tự do d100 phủ bằng token MỚI, giao với vùng cả hai model đều tính được
    both = D["covered"] & S["covered"]
    new  = D["new"] & both
    old  = (~D["new"]) & both
    print(f"\ncoverage: d100 {D['covered'].mean():.4f}  stock {S['covered'].mean():.4f}  "
          f"giao {both.mean():.4f}")
    print(f"ký tự do token MỚI phủ: {new.sum()} / {both.sum()} = {100*new.sum()/both.sum():.2f}% corpus\n")

    def table(title, parts, note=""):
        rows = []
        for name, m in parts:
            if m.sum() == 0:
                continue
            bs, bd = S["char_bits"][m].sum()/m.sum(), D["char_bits"][m].sum()/m.sum()
            rows.append((name, int(m.sum()), 100*m.sum()/both.sum(), bs, bd, bs - bd))
        w = max(len(r[0]) for r in rows)
        print(f"\n{title}")
        if note:
            print(f"  {note}")
        print(f"{'vùng':<{w}} {'n_char':>9} {'%':>6} {'stock':>8} {'d100':>8} {'lợi thế d100':>13}")
        for name, n, pc, bs, bd, d in rows:
            print(f"{name:<{w}} {n:>9} {pc:>5.1f}% {bs:>8.4f} {bd:>8.4f} {d:>+13.4f}")
        return rows

    # --- BẢNG 1: phân vùng theo CHỮ VIẾT — cùng một mask cho cả hai model.
    # Đây là bảng dùng để KẾT LUẬN.
    full = "".join(lines)
    sm = script_char_masks(full)
    rows_script = table(
        "[A] Phân vùng theo chữ viết (độc lập tokenizer) — DÙNG ĐỂ KẾT LUẬN",
        [("toàn corpus", both),
         ("chữ cái có dấu VI", sm["vi_marked"] & both),
         ("chữ Latin không dấu", sm["latin_plain"] & both),
         ("khoảng trắng / dấu câu / số", sm["glue"] & both)])

    # --- BẢNG 2: phân vùng theo tokenizer d100 — CHỈ để mô tả, có thiên lệch chọn mẫu.
    rows_tok = table(
        "[B] Phân vùng theo tokenizer d100 — CHỈ MÔ TẢ, KHÔNG KẾT LUẬN",
        [("ký tự token MỚI", new), ("ký tự token cũ", old)],
        note=("thiên lệch: vùng do chính d100 định nghĩa, chọn đúng nơi d100 có token "
              "chuyên dụng.\n  Hai model năng lực BẰNG NHAU vẫn cho '+ ở vùng mới, "
              "− ở vùng cũ, tổng ~0' dưới phép này."))
    rows = rows_script

    # Quy tắc quyết định đăng ký TRƯỚC khi thấy số (tránh tự thuyết phục sau khi thấy kết quả).
    # Lợi thế toàn corpus 0.0217 b/char; để lợi thế CỤC BỘ trên vùng token mới đạt
    # 0.15 b/char (~4.5% tương đối, mức đủ nói là "có học được gì"), token mới phải
    # phủ <= 14.5% corpus.
    # Quy tắc cũ (ngưỡng phủ 14.5%) đã BỊ RÚT: nó giả định lợi thế không âm ở mọi vùng,
    # nên lợi thế tổng chặn trên lợi thế cục bộ — giả định đó bị vi phạm. Thay bằng
    # quy tắc đọc trực tiếp trên bảng [A], nơi vùng không do tokenizer nào định nghĩa.
    vi_row = next((r for r in rows_script if r[0] == "chữ cái có dấu VI"), None)
    if vi_row:
        adv = vi_row[5]
        print(f"\n[quy tắc quyết định — trên bảng A] lợi thế d100 ở vùng chữ có dấu VI = {adv:+.4f} b/char")
        print("  ->", "CÓ năng lực thêm: pretrain dạy được tiếng Việt vượt mức tokenizer"
              if adv >= 0.15 else
              ("tín hiệu mờ (0.06–0.15): cần lệnh `downstream` mới kết luận được"
               if adv >= 0.06 else
               "KHÔNG có năng lực thêm đo được: bỏ câu 'backbone thật sự học được tiếng Việt'"))
        print("  Điều kiện đi kèm: nếu vùng 'khoảng trắng/dấu câu' âm mạnh (< −0.3) thì")
        print("  backbone bị thoái hoá năng lực chung — nêu như một phát hiện riêng, và nó")
        print("  là giả thuyết giải thích kết quả null của Arm B.")

    print("\nĐọc kết quả:")
    print("  * Bảng [A] và [B] trả lời hai câu hỏi KHÁC NHAU. [B] đo 'd100 nén tốt hơn ở đâu'")
    print("    (đã biết trước: ở nơi nó có token). [A] đo 'd100 hiểu tiếng Việt tốt hơn không'.")
    print("  * BPC toàn corpus có thể ~0 trong khi [A] khác 0: hai hiệu ứng ngược dấu triệt tiêu.")
    print("  * BPC đo NÉN, không đo HIỂU. Kể cả [A] dương, vẫn cần lệnh `downstream`.")

    if a.out:
        def pack(rs):
            return [{"region": r[0], "n_chars": r[1], "pct_corpus": r[2],
                     "bpc_stock": r[3], "bpc_d100": r[4], "d100_advantage": r[5]} for r in rs]
        json.dump({"scoring": "BOS-prefixed; every real token scored",
                   "first_new_id": first_new, "n_new_token_ids": n_new_ids,
                   "coverage": {"d100": float(D["covered"].mean()),
                                "stock": float(S["covered"].mean()),
                                "intersect": float(both.mean())},
                   "pct_chars_new_tokens": float(100*new.sum()/both.sum()),
                   "regions_script_A": pack(rows_script),
                   "regions_tokenizer_B_descriptive_only": pack(rows_tok)},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n-> {a.out}")


def cmd_downstream(a):
    """Zero-shot accuracy bằng cách chấm log-likelihood của từng nhãn."""
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels = [s.strip() for s in a.labels.split(",")]
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    print(f"{len(rows)} mẫu, {len(labels)} nhãn: {labels}\ndevice: {dev}\n")

    def score(tok, model, prompt, cont):
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(cont,   add_special_tokens=False)["input_ids"]
        t = torch.tensor([p + c], device=dev)
        with torch.no_grad():
            lp = torch.log_softmax(model(t).logits[0, :-1].float(), dim=-1)
        tgt = t[0, 1:]
        tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)[len(p)-1:]
        return tok_lp.sum().item() / max(len(c), 1)   # chuẩn hoá theo độ dài nhãn

    out = {}
    for tag, path in (("stock", a.stock), ("d100", a.d100)):
        tok, model = load(path, dev)
        ok = 0
        for n, r in enumerate(rows):
            prompt = a.prompt.format(text=r[a.text_key])
            pred = max(range(len(labels)),
                       key=lambda i: score(tok, model, prompt, " " + labels[i]))
            ok += (labels[pred] == str(r[a.label_key]).strip())
            if n % 200 == 0:
                print(f"  [{tag}] {n}/{len(rows)}", flush=True)
        out[tag] = ok / len(rows)
        print(f"  [{tag}] accuracy = {out[tag]:.4f}\n")
        del model
        torch.cuda.empty_cache() if dev.startswith("cuda") else None

    base = 1.0 / len(labels)
    print(f"chance = {base:.4f}   stock = {out['stock']:.4f}   d100 = {out['d100']:.4f}   "
          f"chênh = {out['d100']-out['stock']:+.4f}")
    print("\nĐọc kết quả:")
    print("  * d100 >> stock: pretrain có thêm năng lực HIỂU thật, BPC chỉ không bắt được.")
    print("    Đây là chỉ số nên đưa vào bài thay cho BPC.")
    print("  * d100 ~= stock, cả hai >> chance: hai backbone tương đương về tiếng Việt.")
    print("    Lập luận 'backbone giỏi tiếng Việt vẫn sụp' mất tiền đề -> đổi cách viết.")
    print("  * Cả hai ~= chance: task/prompt quá khó cho model 500M, chỉ số vô hiệu.")
    print("    Thử prompt khác hoặc task dễ hơn trước khi kết luận gì.")
    if a.out:
        json.dump({"chance": base, **out}, open(a.out, "w"), indent=2)
        print(f"\n-> {a.out}")


def cmd_diacritics(a):
    """Lựa chọn cưỡng bức trên cặp tối thiểu về dấu — BẤT BIẾN với tokenizer."""
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lines = [l.rstrip("\n") for l in open(a.corpus, encoding="utf-8") if l.strip()]
    items = build_minimal_pairs(lines, max_items=a.max_items,
                                n_distract=a.n_distract, seed=a.seed)
    if not items:
        print("Không sinh được cặp tối thiểu nào: corpus thiếu từ cùng dạng-không-dấu.")
        print("Dùng --corpus lớn hơn, hoặc cung cấp bộ cặp thủ công.")
        return
    n_opt = np.mean([len(it["options"]) for it in items])
    print(f"{len(items)} bài, trung bình {n_opt:.1f} phương án/bài  (chance ~ {1/n_opt:.3f})")
    print(f"device: {dev}\n")

    def seq_logprob(tok, model, text):
        enc = tok(text, add_special_tokens=False)["input_ids"]
        bos = tok.bos_token_id or tok.eos_token_id
        if not enc:
            return -1e9
        with torch.no_grad():
            t = torch.tensor([[bos] + list(enc)], device=dev)
            lp = torch.log_softmax(model(t).logits[0, :-1].float(), -1)
            return lp.gather(1, t[0, 1:].unsqueeze(1)).sum().item()

    out, per_item = {}, {}
    for tag, path in (("stock", a.stock), ("d100", a.d100)):
        tok, model = load(path, dev)
        ok, hits = 0, []
        for n, it in enumerate(items):
            scores = [seq_logprob(tok, model, it["prefix"] + o + it["suffix"])
                      for o in it["options"]]
            got = it["options"][int(np.argmax(scores))] == it["correct"]
            ok += got; hits.append(bool(got))
            if n % 100 == 0:
                print(f"  [{tag}] {n}/{len(items)}", flush=True)
        out[tag] = ok / len(items); per_item[tag] = hits
        print(f"  [{tag}] accuracy = {out[tag]:.4f}\n")
        del model
        torch.cuda.empty_cache() if dev.startswith("cuda") else None

    # McNemar ghép cặp: cùng bộ bài, chỉ khác model
    hs, hd = np.array(per_item["stock"]), np.array(per_item["d100"])
    b, c = int((hs & ~hd).sum()), int((~hs & hd).sum())
    from math import comb
    n = b + c
    p = (sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n * 2) if n else 1.0
    p = min(1.0, p)
    print(f"chance ~ {1/n_opt:.4f}   stock = {out['stock']:.4f}   d100 = {out['d100']:.4f}   "
          f"chênh = {out['d100']-out['stock']:+.4f}")
    print(f"McNemar ghép cặp: chỉ-stock-đúng={b}, chỉ-d100-đúng={c}, p={p:.4g}")

    print("\nVì sao chỉ số này đáng tin hơn BPC theo vùng:")
    print("  stock dùng byte-fallback cho tiếng Việt nên LUÔN tốn nhiều bits hơn trên ký tự")
    print("  nhiều byte, kể cả khi hai model gán cùng xác suất. Mọi phép so bits/ký-tự đều")
    print("  dính confound đó. Ở đây mỗi model chấm bằng tokenizer của chính nó, đầu ra 0/1.")
    print("\nĐọc kết quả:")
    print("  * d100 >> stock, p nhỏ: stage-1 THẬT SỰ dạy được chính tả tiếng Việt.")
    print("  * d100 ~= stock: lợi thế BPC trước đó là tạo tác tokenizer, không phải năng lực.")
    print("  * cả hai ~= chance: bài quá khó / corpus thiếu cặp tốt -> chỉ số vô hiệu.")
    if a.out:
        json.dump({"n_items": len(items), "mean_options": float(n_opt),
                   "chance": float(1/n_opt), **out,
                   "mcnemar": {"only_stock": b, "only_d100": c, "p": float(p)}},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n-> {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bpc", help="phân bổ BPC theo token mới / token cũ")
    b.add_argument("--stock", required=True, help="backbone tiếng Anh gốc (vocab 49.280)")
    b.add_argument("--d100", required=True, help="backbone pretrain đầy đủ (vocab 57.344)")
    b.add_argument("--corpus", required=True, help="vi_bpc_corpus.txt")
    b.add_argument("--first-new-id", type=int, default=None,
                   help="id nhỏ nhất của token mới; mặc định = len(tokenizer stock)")
    b.add_argument("--device", default=None)
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_bpc)

    k = sub.add_parser("diacritics",
                       help="lựa chọn cưỡng bức trên cặp tối thiểu về dấu (bất biến tokenizer)")
    k.add_argument("--stock", required=True)
    k.add_argument("--d100", required=True)
    k.add_argument("--corpus", required=True, help="vi_bpc_corpus.txt")
    k.add_argument("--max-items", type=int, default=400)
    k.add_argument("--n-distract", type=int, default=4)
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--device", default=None)
    k.add_argument("--out", default=None)
    k.set_defaults(func=cmd_diacritics)

    d = sub.add_parser("downstream", help="accuracy zero-shot trên task tiếng Việt")
    d.add_argument("--stock", required=True)
    d.add_argument("--d100", required=True)
    d.add_argument("--data", required=True, help="jsonl")
    d.add_argument("--labels", required=True, help="danh sách nhãn, phân tách bằng dấu phẩy")
    d.add_argument("--prompt", required=True, help="template chứa {text}")
    d.add_argument("--text-key", default="text")
    d.add_argument("--label-key", default="label")
    d.add_argument("--device", default=None)
    d.add_argument("--out", default=None)
    d.set_defaults(func=cmd_downstream)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
