# Vách năng lực ngôn ngữ của backbone — thiết kế thí nghiệm ladder

Ngày: 2026-08-03 · Trạng thái: đã duyệt thiết kế, chờ implementation plan

## 1. Câu hỏi

Vách năng lực ngôn ngữ của backbone VLM nằm ở đâu, và nó là hàm của **quy mô mô hình**
hay của **liều lượng phơi nhiễm ngôn ngữ lúc pretrain**?

Phát biểu cần kiểm chứng:

> Với VLA quy mô nhỏ, năng lực đa ngôn ngữ của backbone không suy giảm dần theo quy mô —
> nó có một ngưỡng vách. Dưới ngưỡng đó, chỉ dẫn ngoài ngôn ngữ huấn luyện không phải
> "hiểu kém" mà là hoàn toàn ngoài phân phối.

## 2. Định vị so với văn liệu (đã kiểm chứng 2026-08-03)

Ba bài trong `de_xuat_thuc_nghiem_cho_paper.md` đều có thật, đã đọc abstract/HTML:

- **`arXiv:2606.11906`** — "When Does Language Matter?". 10 ngôn ngữ **có tiếng Việt**,
  dịch bằng Google Translate + back-translation (98–99% nhất quán). Model: OpenVLA-OFT,
  π₀.₅. Số tiếng Việt zero-shot: **OpenVLA-OFT 59.6%**, **π₀.₅ 57.3%**; can thiệp
  step-wise lúc suy luận nâng lên 68.0% / 81.9%. Cơ chế can thiệp: phân tích offline tỉ
  lệ gradient text/image chọn 50% step nhạy ngôn ngữ → lúc chạy retrieve K=5 biểu diễn
  tham chiếu tiếng Anh → alignment có trọng số α (1.0 / 0.8).
- **`arXiv:2606.15714`** — "Beyond English". 5 ngôn ngữ (Trung, Pháp, Nga, Ả Rập, Anh),
  **không có tiếng Việt**. Backbone: PaliGemma/Prismatic (EN-centric) vs Qwen-VL (đa ngôn
  ngữ). Đề xuất MPCA. **Có nhánh can thiệp phía train**: M-FT (finetune VLM đa ngôn ngữ
  trước, rồi train action head) và M-CT (co-train COCO-VQA đa ngôn ngữ + LIBERO); M-CT
  tốt hơn M-FT, cả hai đều tụt trên tiếng Anh so với baseline EN-only.
- **`arXiv:2602.17659`** — LIBERO-CF, benchmark phản thực, chỉ tiếng Anh.

Ba bài chồng lấn thêm mà đề xuất gốc bỏ sót: **`2603.00592` LangGap** (nhiễu loạn ngữ
nghĩa 4 chiều trên π₀.₅ + data augmentation, 0%→90% ở single-task), **`2603.28301`
LIBERO-Para** (robustness với diễn đạt lại), **LIBERO-Plus** (CVPR 2026). Cộng LIBERO-CF
là **bốn** benchmark phản thực đã tồn tại — E1 trong đề xuất gốc không còn là đóng góp,
chỉ là bước phòng thủ bắt buộc.

**Hai chỗ đề xuất gốc nói sai, đã sửa ở đây:**

1. "Cách khắc phục của cả ba đều là hậu kỳ" — sai. `2606.15714` có M-FT/M-CT, đúng là can
   thiệp phía train. Tuyên bố phải hẹp hơn: chưa ai thử backbone **đơn ngữ đích** (họ dùng
   finetune **đa ngôn ngữ**), và chưa ai làm ở **quy mô <1B**.
2. E4 đề xuất thay backbone sang Qwen2-VL-2B / PaliGemma-3B. Không cần: SmolVLA dẫn xuất
   `lm_expert_config` từ text-config của VLM (`hidden_size × expert_width_multiplier`,
   `num_hidden_layers = num_vlm_layers`, xem `smolvlm_with_expert.py:100-118`) và
   `num_vlm_layers=16` cắt VLM xuống 16 layer đầu — nên đổi kích thước **trong họ**
   SmolVLM2 chỉ là đổi `vlm_model_name`. Ladder quy mô rẻ hơn nhiều so với phẫu thuật
   kiến trúc xuyên họ.

**Ô trống còn lại** (giao của bốn điều kiện): backbone **đơn ngữ đích** × quy mô **<1B** ×
ngôn ngữ **ít tài nguyên** × **liều lượng phơi nhiễm đo được**. Thí nghiệm này chiếm đúng ô
đó. `2606.15714` kết luận "khoảng cách do nguồn ngôn ngữ của base VLM" nhưng đo bằng cách so
các model **khác kiến trúc** — bị nhiễu; ở đây mọi biến khác được giữ cố định, chỉ đổi liều.

Đối chiếu định lượng làm nên đóng góp: **59.6% (7B, đa ngôn ngữ) vs 0.0% (500M, EN-centric)**
trên cùng loại setup EN-train/VI-eval. Đây là hai chế độ thất bại khác hẳn nhau, không phải
cùng hiện tượng ở hai mức độ.

## 3. Trạng thái hiện có

Từ `outputs/report_smolvla_en_vs_vi.md` (giao thức: full-FT 50K, batch 16, seed 1000,
train trên `VLAIResearchLab/lerobot_libero_vi`, eval 4 suite × 10 task × 10 episode):

| Đã có | Backbone | Train | Eval | Overall |
|---|---|---|---|---|
| Arm A `smolvla-ft-50k` | `SmolVLM2-500M-Video-Instruct` | VI | VI | 45.0 |
| Arm B `smolvla-vi-ft-50k` | `thuanan/SmolVLM2-500M-vi-stage1` | VI | VI | 53.2 |
| Arm C vocab-only | `outputs/backbones/smolvlm2_vi_vocab_only/` | VI | VI | 3 seed, p=0.027 |
| Baseline ngoài | `HuggingFaceVLA/smolvla_libero` | EN | EN / VI | 66.5 / **0.0** |

Chưa có run nào **tự train trên dữ liệu LIBERO tiếng Anh** (`outputs/eval_en_*` là eval EN
của checkpoint train VI). Toàn bộ nhánh EN-train trong spec này là mới, và nó lấp luôn ô
"backbone EN × dữ liệu EN" mà E3 của đề xuất gốc đòi hỏi — hai thí nghiệm gộp làm một.

Backbone VI stage-1 (`thuanan/SmolVLM2-500M-vi-stage1`): ViOCRVQA ~19.7k, OpenViVQA ~9.1k,
UIT-ViIC ~13.5k, mẫu VQA tiếng Anh chống quên, ViWiki ~25k; 2 epoch, batch hiệu dụng 48
(4 × 3 GPU × 4 grad-accum), LR 1e-4 cosine, bf16, ảnh 1536px, LoRA r=32 α=64 trên attention
+ MLP projection, cộng embedding tiếng Việt mới train. Code và dữ liệu stage-1 **vẫn còn**
(người dùng xác nhận), nhưng **chưa nằm trong repo này**.

## 4. Thiết kế

### 4.1 Điều kiện chính

Mọi điểm ladder: full-FT SmolVLA **50K steps** trên LIBERO **tiếng Anh**
(`TRAIN_EXPERT_ONLY=false`, batch 16), rồi eval **cả EN lẫn VI** (400 episode mỗi bên).
Chênh EN−VI của từng điểm là "khoảng cách đa ngôn ngữ", so sánh trực tiếp được với 59.6%.

Vách chỉ tồn tại trong điều kiện zero-shot này. Trong điều kiện VI-train/VI-eval không có
gì ở 0% (Arm A đã 45.0), nên ở đó chỉ đo được con dốc thoải 45→53, không phải vách.

### 4.2 Phase 1 — ladder liều lượng tiếng Việt (giữ cố định 500M)

| Điểm | Backbone | Vocab | Liều VI | Trạng thái |
|---|---|---|---|---|
| `stock` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 49,280 | không có gì cả | có sẵn |
| `d0` | `outputs/backbones/smolvlm2_vi_vocab_only/` | 57,344 | 0% — có token, chưa train | **đã build** |
| `d10` | stage-1 trên 10% mixture | 57,344 | 10% | cần build |
| `d25` | stage-1 trên 25% mixture | 57,344 | 25% | cần build |
| `d50` | stage-1 trên 50% mixture | 57,344 | 50% | cần build |
| `d100` | `thuanan/SmolVLM2-500M-vi-stage1` | 57,344 | 100% | có sẵn |

`stock` là điểm bắt buộc mà đề xuất gốc thiếu: nó tách "không có token tiếng Việt" khỏi
"có token nhưng chưa train tiếng Việt" (`d0`).

**Trục liều = lấy mẫu con mixture theo tỉ lệ**, giữ nguyên thành phần (mọi nguồn con, kể cả
mẫu EN chống quên, đều cắt cùng tỉ lệ), giữ nguyên 2 epoch và mọi siêu tham số khác. Vậy
liều ≈ số token tiếng Việt đã thấy — đúng ràng buộc thực tế của ngôn ngữ ít tài nguyên, và
cho ra khuyến nghị dùng được: *cần bao nhiêu dữ liệu để pretrain backbone*.

Các tập mẫu con phải **lồng nhau**: 10% ⊂ 25% ⊂ 50% ⊂ 100%. Nếu không lồng nhau, chênh lệch
giữa hai mốc lẫn cả nhiễu do đổi mẫu, không còn là dose-response.

**Phase 1a:** 6 điểm × EN-train × 1 seed (1000) = 6 run ≈ 84 GPU-h. Chỉ dùng để đọc **hình
dạng** đường cong.
**Phase 1b:** thêm seed 2000, 3000 cho cả 6 điểm = 12 run ≈ 168 GPU-h. Mọi phát biểu định
lượng dựa vào đây.

### 4.3 Phase 2 — ladder quy mô (giữ cố định EN-centric)

`HuggingFaceTB/SmolVLM2-256M-Video-Instruct` · `…-500M-Video-Instruct` (= `stock`, dùng
chung, không train lại) · `HuggingFaceTB/SmolVLM2-2.2B-Instruct`. Cùng giao thức EN-train,
eval EN + VI. 2 run mới ≈ 30 GPU-h (chưa tính phụ trội của 2.2B).

Nếu VI đứng yên ≈0% suốt 256M→2.2B trong khi ladder liều lượng nhấc nó lên, thì vách là
chuyện **phơi nhiễm ngôn ngữ, không phải quy mô**.

**Giới hạn phải ghi rõ trong bài:** SmolLM2 là EN-centric nhưng không thuần EN (có web text
đa ngôn ngữ), và 2.2B vừa nhiều tham số vừa có thể đã thấy nhiều tiếng Việt hơn. Phase 2 là
**kiểm chứng giới hạn trên**, không phải tách biến hoàn hảo.

### 4.4 Nhánh phụ VI-train

Nối phần mới với kết quả cũ. Đã xong 3/6 điểm: `stock`→Arm A (45.0), `d100`→Arm B (53.2),
`d0`→Arm C (3 seed). Chỉ cần thêm `d10`/`d25`/`d50` VI-train, 1 seed ≈ 42 GPU-h, là có ma
trận 6 điểm × 2 ngôn ngữ dữ liệu đầy đủ.

### 4.5 Ngân sách

| Hạng mục | GPU-h |
|---|---|
| Phase 1a (6 run, 1 seed) | ~84 |
| Nhánh VI-train (3 run) | ~42 |
| Phase 2 (2 run) | ~30 |
| Phase 1b (12 run, +2 seed) | ~168 |
| **Tổng** | **~324** |

Stage-1 pretrain cho `d10`/`d25`/`d50` rẻ (mixture ~67k mẫu, 2 epoch) — vài GPU-h mỗi mốc,
gộp chung vào phần dư. Đường cong đọc được ngay sau Phase 1a. Không hard-code số GPU: driver
xếp hàng đợi và lấp slot rảnh.

## 5. Thành phần

### 5.1 `vlai-experiments/vi-instructions/pretrain_vi_backbone.py` (mới)

Đưa code stage-1 vào repo, tham số hoá theo liều lượng.

- Đầu vào: `--data-fraction {0.10,0.25,0.50,1.00}`, `--subsample-seed`, `--output-dir`.
- Lấy mẫu con theo **từng nguồn con**, cùng tỉ lệ, seed cố định, **lồng nhau** (sắp xếp tất
  định rồi cắt tiền tố — cách đơn giản nhất bảo đảm tính lồng).
- Giữ nguyên mọi siêu tham số stage-1 ở mục 3.
- `resize_token_embeddings(57344, mean_resizing=True)`, merge LoRA, `save_pretrained` **kèm
  tokenizer/processor**.
- Ghi `dose_manifest.json`: tỉ lệ, số mẫu mỗi nguồn, số token VI, seed, git SHA.

### 5.2 `vlai-experiments/vi-instructions/validate_backbone.py` (mới, nhẹ)

Cổng chặn cứng, chạy trước mỗi lần train VLA:

- `embed_tokens.weight.shape[0] == lm_head.weight.shape[0] == 57344` (49,280 cho `stock`).
- Tokenizer round-trip một chỉ dẫn LIBERO tiếng Việt ra đúng id vocab mở rộng.
- **Perplexity tiếng Việt trên corpus ngoài.** Đây là trục hoành thứ hai: đổi đường cong từ
  "% dữ liệu → thành công điều khiển" thành "năng lực ngôn ngữ → thành công điều khiển",
  mạnh hơn hẳn về mặt lập luận.

  Tập đo **không được cắt ra từ mixture stage-1**: mốc `d100` là checkpoint công khai
  `thuanan/SmolVLM2-500M-vi-stage1` đã train trên toàn bộ mixture, nên bất kỳ phần nào cắt
  ra từ đó cũng đã bị `d100` nhìn thấy và perplexity của nó sẽ bị thổi phồng. Thay vào đó
  dùng một **corpus tiếng Việt ngoài** mà không mốc liều nào từng thấy (ứng viên: một split
  test của bộ VQA/caption tiếng Việt không nằm trong mixture, hoặc văn bản tin tức tiếng
  Việt sau thời điểm crawl của ViWiki — chốt cụ thể ở bước plan).

  Cùng một tập dùng cho mọi backbone, kể cả `stock` (vocab 49,280) — perplexity của `stock`
  sẽ cao vì phân mảnh token, và đó chính là tín hiệu cần đo. Perplexity so sánh được giữa
  hai vocab khác nhau **chỉ khi chuẩn hoá theo byte/ký tự, không theo token**; báo cáo
  bits-per-character để tránh đúng cái bẫy này.

### 5.3 `vlai-experiments/vi-instructions/ladder_driver.sh` (mới)

Sao khuôn `outputs/ft_multiseed_pipeline/driver.sh`: bảng job phẳng, hàng đợi lấp slot GPU
rảnh, sống sót qua mất SSH (`setsid nohup`). Mỗi job: `validate_backbone.py` (chặn) → train
(`run_vi.sh` với `DATASET_REPO=HuggingFaceVLA/libero` — đây là dataset EN gốc mà
`VLAIResearchLab/lerobot_libero_vi` được fork ra, xác nhận ở
`docs/superpowers/plans/2026-07-02-smolvla-vietnamese-instructions.md:710`;
`TRAIN_EXPERT_ONLY=false`, `STEPS=50000`, `BATCH_SIZE=16`) → eval EN (`run_eval_en.sh`) →
eval VI (`run_eval_vi.sh`, dùng `--env.task_language_overrides_path` sẵn có).

### 5.4 `vlai-experiments/vi-instructions/ladder_report.py` (mới)

Gom `eval_info.json`, xuất `outputs/report_backbone_ladder.md`: bảng ladder, đường cong
(liều → pc_success EN/VI; perplexity → pc_success VI), paired bootstrap trên 40 task. Map
các run cũ (Arm A/B/C) vào ô `*_vi` qua bảng alias, **không đổi tên thư mục cũ**.

### 5.5 Luồng và đặt tên

```
pretrain_vi_backbone.py --data-fraction X
  → outputs/backbones/vi_dose_{10,25,50}/
  → validate_backbone.py            (chặn nếu fail)
  → ladder_driver.sh
      → outputs/train_ladder_<point>_<lang>[_seed<N>]/
      → outputs/eval_ladder_<point>_<lang>[_seed<N>]_{en,vi}/
  → ladder_report.py → outputs/report_backbone_ladder.md
```

`<point> ∈ {stock, d0, d10, d25, d50, d100, s256m, s2200m}` · `<lang> ∈ {en, vi}` (ngôn ngữ
**dữ liệu train**) · hậu tố seed bỏ trống nghĩa là seed 1000.

## 6. Xử lý lỗi

**E1 — `tokenizer_name` cũ (nghiêm trọng nhất).** Đã cắn hai lần trong dự án này:
`policy_preprocessor.json` giữ tokenizer EN 49,280 trong khi embedding table là 57,344 —
sai âm thầm, không crash, và ra 0% trông y hệt một phát hiện khoa học. Spec này thêm 3
backbone mới, mỗi cái là một cơ hội tái phạm. `validate_backbone.py` là cổng chặn cứng
trong driver; sau khi train xong, assert `policy_preprocessor.json → tokenizer_name` khớp
`config.json → vlm_model_name`.

**E2 — race condition symlink `checkpoints/last`.** Đã làm hỏng lần đọc ft-10K trước đây.
Driver chỉ eval đường dẫn step tường minh `checkpoints/050000/pretrained_model`.

**E3 — `save_pretrained` không lưu processor pipeline.** Assert sự tồn tại 4 file
(`policy_preprocessor.json` / `.safetensors`, `policy_postprocessor.json` / `.safetensors`)
trước mỗi eval.

**E4 — OOM ở SmolVLM2-2.2B.** Full-FT batch 16 gần như chắc chắn không vừa A5000 24GB. Hạ
`BATCH_SIZE` và bù bằng grad-accum để **giữ nguyên batch hiệu dụng 16**; nếu không,
"2.2B kém hơn" không tách được khỏi "2.2B train với batch khác". Nếu vẫn không vừa, Phase 2
chuyển sang GPU lớn hơn — đây là lý do Phase 2 tách riêng.

**E5 — job chết giữa chừng.** Log riêng mỗi job, phát hiện `Traceback|CUDA out of memory`,
đánh dấu hỏng và **tiếp tục hàng đợi**. `ladder_report.py` báo rõ ô nào thiếu thay vì lặng
lẽ bỏ qua.

## 7. Kiểm chứng

**Thống kê.** Phase 1a (1 seed) chỉ để đọc hình dạng đường cong, không dùng để tuyên bố.
Phát biểu định lượng (ví dụ "d25 hơn d10 bao nhiêu điểm") chờ Phase 1b 3 seed + paired
bootstrap trên 40 task. Riêng phát biểu **"có vách"** chỉ cần độ lớn hiệu ứng thô (0% vs
hàng chục %), nên chịu được 1 seed — spec ghi rõ phát biểu nào cần bao nhiêu seed.

**Unit test** (`vlai-experiments/vi-instructions/tests/`) cho phần lấy mẫu con của
`pretrain_vi_backbone.py`: tính lồng nhau (10% ⊂ 25% ⊂ 50% ⊂ 100%), tính tất định theo seed,
và tỉ lệ thành phần mixture được giữ nguyên qua mọi mốc.

**Sanity trước khi tốn GPU.** `validate_backbone.py` pass trên cả 6 backbone Phase 1 trước
khi launch job nào.

## 8. Điều kiện dừng sớm

Nếu Phase 1a cho thấy cả `stock`, `d0`, `d10`, `d25`, `d50`, `d100` **đều** ≈0% ở eval VI:
không có vách trong khoảng liều này. Dừng Phase 1b, chuyển ngân sách sang Phase 2 (vách có
thể nằm ở quy mô), và báo cáo kết quả âm tính một cách trung thực.

## 9. Ngoài phạm vi

- **E5 đường cong hiệu quả dữ liệu robot** (bao nhiêu demo VI bù được backbone) — thí nghiệm
  giá trị cao, brainstorm riêng, spec riêng.
- **E6 tái lập can thiệp lúc suy luận** (`2606.11906` step-wise, `2606.15714` MPCA) trên
  SmolVLA — spec riêng. Rủi ro cao nhất trong toàn bộ đề xuất: dễ bị bắt lỗi "implement sai",
  và kết quả gần như chắc chắn là ≈0% (một negative result dễ đoán trước).
- **E1 battery phản thực** — nay là bước phòng thủ, phải dùng benchmark đã có (LIBERO-CF /
  LangGap / LIBERO-Para) chứ không tự chế. Spec riêng.
- **E2 phân tích per-task** trên dữ liệu đã có (0 GPU) — việc nhỏ, không cần spec.
- Đóng khung lại phần mở đầu bài báo theo mục 2 — việc viết, không phải việc code.
- Thay backbone **xuyên họ** (Qwen2-VL, PaliGemma) — bị Phase 2 thay thế.

## 10. Liên kết

`de_xuat_thuc_nghiem_cho_paper.md` · `outputs/report_smolvla_en_vs_vi.md` ·
`outputs/eda_libero_suites.md` ·
`docs/superpowers/specs/2026-07-28-vi-vocab-only-backbone-ablation-design.md` ·
`docs/superpowers/runbooks/2026-07-14-smolvla-vs-smolvla-vi-backbone.md` ·
[[project-vi-ab-pipeline-running]] · [[smolvla-hybrid-backbone-gotchas]]
