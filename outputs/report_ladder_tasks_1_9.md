# Báo cáo thực thi Task 1–9 — Backbone Language-Cliff Ladder

Khoảng thời gian: 2026-08-03 → 2026-08-05
Spec: `docs/superpowers/specs/2026-08-03-backbone-language-cliff-ladder-design.md`
Plan: `docs/superpowers/plans/2026-08-03-backbone-language-cliff-ladder.md`

**Tóm tắt một dòng:** hạ tầng ladder đã hoàn tất và chạy được; kết quả bác bỏ giả
thuyết trung tâm cũ của dự án và thay bằng một phát biểu mạnh hơn.

---

## 1. Kết quả khoa học

### 1.1 Phase 1a — điều kiện chính (EN-train → eval EN + VI zero-shot)

| Nấc | Backbone biết tiếng Việt? | bits/char | EN | **VI** |
|---|---|---|---|---|
| `stock` | không, vocab 49.280 | 3.34 | 55.8 | **0.0** |
| `d0` | có vocab 57.344, chưa train | 4.41 | 53.2 | **0.0** |
| `d100` | **pretrain đầy đủ** | **3.32** | 54.0 | **0.0** |

1.200 episode tiếng Việt, **không một lần thành công**, kể cả backbone đã pretrain đầy
đủ tiếng Việt.

> **Đọc cột bits/char cho đúng (sửa 2026-08-05).** Cả sáu con số BPC trong tài liệu này
> đều đo trên **cùng một corpus tiếng Việt** (`vi_bpc_corpus.txt`, 400 dòng đầu) —
> không có phép đo nào trên văn bản tiếng Anh. Do đó `stock` = 3.34 và `d100` = 3.32 là
> **ngang nhau** (chênh 0.018 bit/ký tự, vô nghĩa), chứ không phải "`d100` hiểu tiếng
> Việt tốt hơn".
>
> Nguyên nhân: tiếng Việt dùng chữ Latin, nên BPE tiếng Anh băm nó thành nhiều mảnh mà
> model đều thạo; `d100` dùng ít token dài hơn và cũng thạo. Quy về mỗi *ký tự* thì hoà.
>
> **Hệ quả:** BPC như đang đo **không tách được** `stock` khỏi `d100`.
>
> **Cập nhật 2026-08-06 — đã đo bằng chỉ số bất biến tokenizer, và kết quả đảo dấu
> hoàn toàn theo hướng ngược lại dự đoán ban đầu.** Xem mục 1.3′. `d100` không "ngang
> `stock`" mà **kém `stock`** ở năng lực tiếng Việt thật (0.52 vs 0.62 trên bài lựa
> chọn cưỡng bức chính tả, p = 7.6×10⁻⁴). `d0` gần như đoán mò (0.27, ≈ ngẫu nhiên
> 0.23). Ba số 0.0 trong bảng trên vì thế không còn là nghịch lý "backbone giỏi tiếng
> Việt mà vẫn sụp" — chúng chỉ đơn giản là ba backbone không giỏi tiếng Việt (theo
> mức độ khác nhau) đều sụp khi eval tiếng Việt sau khi chỉ train tiếng Anh.

Nhánh EN phẳng (55.8 / 53.2 / 54.0, sd tổng thể ~6 điểm đo từ multiseed) — backbone
không ảnh hưởng tiếng Anh, đúng như thiết kế mong đợi.

**Đối chứng loại trừ lỗi eval:** cùng checkpoint, cùng harness, chỉ khác ngôn ngữ chỉ
dẫn → 54.0 vs 0.0. Bộ eval hoạt động bình thường.

### 1.2 Phát biểu thay thế (thu hẹp lại — xem đính chính mục 1.3′)

Giả thuyết ban đầu — *dưới một ngưỡng năng lực backbone, chỉ dẫn ngoài ngôn ngữ train
là ngoài phân phối* — dự đoán `d100` phải cứu được. Nó không.

> **Năng lực ngôn ngữ của backbone không chuyển hoá thành neo ngôn ngữ–hành động.**
> Tháp VLM biết tiếng Việt, nhưng action expert chỉ từng được điều kiện hoá trên
> embedding của chỉ dẫn tiếng Anh. Đổi ngôn ngữ lúc test làm vỡ hoàn toàn.

> **Đính chính 2026-08-06:** vế "bất kể backbone hiểu ngôn ngữ mới đến đâu" của câu
> trên **chưa được chứng minh** và nên bỏ. Mục 1.3′ cho thấy `d100` — thứ được gọi là
> "pretrain đầy đủ" — thật ra kém `stock` ở năng lực tiếng Việt đo bằng chỉ số bất biến
> tokenizer. Dự án chưa có một backbone nào *thật sự* giỏi tiếng Việt hơn `stock` để
> kiểm chứng vế đó. Phần còn đứng vững: train expert đúng ngôn ngữ là điều kiện cần,
> và với hai backbone đã thử (đều không giỏi tiếng Việt hơn baseline EN), thiếu điều
> kiện đó cho kết quả 0/400 cả hai lần.

Được chống lưng bởi cả hai phía dữ liệu:

- **Train đúng ngôn ngữ thì chạy được** — Arm A/B/C ở VI-train: 51.1 / 52.2 / 39.2
- **Không train đúng ngôn ngữ thì bằng 0**, bất kể backbone giỏi ngôn ngữ đó cỡ nào

Khớp với thí nghiệm hybrid tháng 7 (ghép tháp VI vào expert đã train EN → 0.0%), nay
được xác nhận ở dạng mạnh nhất: **pretrain đầy đủ cũng không đủ**.

### 1.3 Ladder BPC — dose-response 5 điểm (SỤP ĐỔ, xem đính chính 2026-08-06)

| Mốc | bits/char | Đóng khoảng cách |
|---|---|---|
| `d0` | 4.4098 | — |
| `d10` | 4.1304 | 26% |
| `d25` | 3.7549 | 60% |
| `d50` | 3.4261 | 90% |
| `d100` | 3.3183 | 100% |

Giảm đơn điệu, lõm mạnh: **50% dữ liệu lấy được 90% lợi ích**.

Đo trên corpus held-out `vi_bpc_corpus.txt` (4.590 dòng, dùng 400 dòng đầu), chồng lấn
với dữ liệu train stage-1: **0,0000%**.

> **ĐÍNH CHÍNH 2026-08-06 — mục này SAI, giữ lại để lộ quá trình sửa.** Kết luận ban
> đầu ("đường cong chứng minh 8.064 hàng embedding đã được train có hiệu quả") dựa
> trên BPC toàn corpus, chỉ số bị chi phối bởi **hiệu quả nén của tokenizer**, không
> đo được **năng lực ngôn ngữ**. Sau ba lần sửa (`vlai-experiments/vi-instructions/
> diagnose_vi_competence.py`), phép đo bất biến tokenizer (mục 1.3′ dưới đây) cho kết
> luận **ngược lại hoàn toàn**: mọi mốc liều đều kém `stock` ở chính tả tiếng Việt.

#### 1.3′ Phép đo đúng — lựa chọn cưỡng bức bất biến tokenizer

BPC không so được `stock` (vocab 49.280) với các mốc liều (vocab 57.344) một cách công
bằng: `stock` byte-fallback trên ký tự tiếng Việt nhiều byte nên **cơ học** tốn nhiều
bits hơn mỗi ký tự, kể cả khi nó dự đoán chuỗi tốt ngang hoặc hơn model kia. Ba lần thử
sửa BPC (rải bits theo ký tự, chèn BOS, phân vùng theo chữ viết thay vì theo token) đều
không thoát được confound này — lần cuối còn cho ra dấu **sai** (`d100` "hơn" `stock`
+1,54 b/char ở ký tự có dấu VI, xem lịch sử sửa trong `diagnose_vi_competence.py`).

Phép đo thoát được confound: **bài toán lựa chọn cưỡng bức**. Che một từ có dấu tiếng
Việt trong câu, đưa các biến thể sai dấu làm phương án nhiễu, mỗi model chấm toàn bộ
phương án bằng tokenizer của **chính nó**, chỉ hỏi nó có chọn đúng không. Đầu ra 0/1,
không mẫu số, không đơn vị — bất biến tokenizer.

| Mốc | Liều VI | `stock` | Backbone liều | Chênh | McNemar p |
|---|---|---|---|---|---|
| `d0` | 0% | 0.6075 | **0.2725** (≈ ngẫu nhiên 0.231) | −0.335 | 2.8×10⁻²³ |
| `d10` | 10% | 0.6150 | **0.3300** | −0.285 | 6.0×10⁻¹⁸ |
| `d25` | 25% | 0.6225 | **0.4400** | −0.183 | 7.4×10⁻¹⁰ |
| `d50` | 50% | 0.6250 | **0.5125** | −0.113 | 1.3×10⁻⁴ |
| `d100` | 100% | 0.6200 | **0.5200** | −0.100 | 7.6×10⁻⁴ |

**`stock` — backbone chưa từng thấy một dòng tiếng Việt nào — thắng ở mọi mốc liều,
có ý nghĩa thống kê mạnh ở cả năm.** Đây là dose-response đơn điệu và rõ ràng, nhưng đi
từ *tan rã gần như hoàn toàn* (`d0` xấp xỉ đoán mò) lên *hồi phục một phần* (`d100`) —
không bao giờ vượt qua điểm xuất phát.

Bài kiểm tra còn **ưu ái** các mốc liều theo hai chiều (dạng đúng dấu thường ít token
hơn dưới tokenizer 57.344, kéo tổng log-prob lên; dạng đúng dấu nhiều token hơn dưới
tokenizer của `stock`, kéo xuống) — khoảng cách thật nhiều khả năng còn lớn hơn 10–34
điểm phần trăm đo được.

**Đọc lại đường cong dose-response dưới ánh sáng này:** nó không phải "học thêm được
bao nhiêu", mà là "vá lại được bao nhiêu phần đã bị chính quá trình mở rộng vocab +
LoRA phá hỏng". `d0` (0.27, ≈ may rủi) là đáy tổn thương; liều tăng dần vá dần
(0.33 → 0.44 → 0.51 → 0.52); nhưng mixture 100% cũng chỉ đủ vá **một phần**.

**Đây là phát hiện đáng công bố độc lập với toàn bộ ladder LIBERO**: continued-
pretraining bằng LoRA + mở rộng vocab ở quy mô 500M có thể **làm hỏng** năng lực ngôn
ngữ mục tiêu thay vì thêm vào, đo được bằng chỉ số bất biến tokenizer, dose-response
sạch, mọi mốc có ý nghĩa thống kê mạnh. Cảnh báo thực tiễn cho cộng đồng ngôn ngữ ít
tài nguyên: công thức "mở rộng vocab + LoRA" ở quy mô nhỏ cần kiểm chứng bằng chỉ số
bất biến tokenizer trước khi tin, BPC/perplexity thô có thể đánh lừa theo đúng chiều
ngược lại.

**Hệ quả cho mục 1.1 và 1.2, cần đọc lại:** ba số 0.0 trong bảng 1.1 giờ **không còn
bất ngờ**. `d100` không phải "backbone giỏi tiếng Việt vẫn sụp" — nó là backbone **kém
tiếng Việt hơn `stock`**. Câu chuyện "năng lực ngôn ngữ không neo được vào hành động"
(mục 1.2) mất đi ví dụ minh hoạ mạnh nhất của nó. Điều còn đứng vững, thu hẹp lại: hai
backbone khác nhau đáng kể ở năng lực tiếng Việt, cả hai đều cho 0/400 khi eval tiếng
Việt zero-shot sau khi chỉ train tiếng Anh — tức việc train-đúng-ngôn-ngữ là điều kiện
cần, và không backbone nào trong hai cái này tự nó đủ để bỏ qua điều kiện đó.

### 1.4 Multiseed full-FT (chạy song song, hoàn tất 2026-08-04)

| Arm | s1000 | s2000 | s3000 | mean | sd |
|---|---|---|---|---|---|
| A — backbone EN | 45.0 | 51.5 | 56.8 | **51.1** | 5.9 |
| B — backbone VI | 53.2 | 48.5 | 55.0 | **52.2** | 3.4 |
| C — vocab-only | 43.0 | 34.2 | 40.2 | **39.2** | 4.5 |

Paired bootstrap 40 task (trung bình 3 seed, 10.000 lần lấy mẫu):

| So sánh | Chênh | CI 95% | Ý nghĩa |
|---|---|---|---|
| B − A | +1.17 | [−3.00, +5.42] | **không** |
| C − A | −11.92 | [−16.50, −7.33] | **có** |
| C − B | −13.08 | [−18.67, −7.58] | **có** |

Kết quả "+8.2, backbone VI thắng" — từng là kết luận chính của dự án — **không tái lập
được**. Hai trong ba seed cho dấu âm. Nguồn gốc truy được: `libero_object` của Arm A
dao động 33.0 → 48.0 → 60.0 chỉ do đổi seed, và seed 1000 rơi đúng đáy.

Kết quả duy nhất sống sót là Arm C, với ý nghĩa mạnh hơn cách phát biểu cũ: không phải
*"vá vocab là không đủ"* mà **mở rộng vocab không kèm pretrain ngôn ngữ thì làm hỏng
model** — kém 12 điểm so với không làm gì.

**Ngưỡng nhiễu của thiết lập:** sd 4,5–5,9 điểm ở mức overall, tới 13,5 ở mức suite.
Mọi so sánh dưới ~12 điểm dựa trên 1 seed đều không đáng tin. Con số này nên vào phần
method của bài.

---

## 2. Thực thi từng task

| Task | Nội dung | Kết quả |
|---|---|---|
| 1 | `dose_mixture.py` — nhân tỉ lệ `sampling_strategy` theo liều | 17 test pass; tính lồng nhau miễn phí nhờ `random.seed(42)` của smollm-vi |
| 2 | Env override trong `smollm-vi/train_gpus.sh` | `MIXTURE_TEMPLATE`, `WARMUP_RATIO`, `RUN_NAME`, `DDP_TIMEOUT`; default giữ nguyên hành vi cũ |
| 3 | `merge_stage1_adapter.py` — adapter → backbone độc lập | 11 test pass; merge lại khớp **bit-perfect** với `thuanan/SmolVLM2-500M-vi-stage1` (diff 0.0 mọi tensor) |
| 4 | `build_bpc_corpus.py` — corpus tiếng Việt held-out | 4.590 dòng / 240.803 ký tự, chồng lấn train **0,0000%** |
| 5 | `validate_backbone.py` — cổng chặn + BPC | 14 test pass; BPC bits-per-**character** để so được giữa vocab 49.280 và 57.344 |
| 6 | `d0` anchor (liều 0, khởi tạo sub-token) | BPC 4.4098, nằm đúng giữa Arm C (7.97) và `d100` (3.32) |
| 7 | `d10` / `d25` / `d50` backbones | 3 lượt stage-1, 142/358/718 bước; BPC giảm đơn điệu |
| 8 | `ladder_driver.sh` — hàng đợi GPU | cổng validate chặn trước khi tiêu GPU; idempotent; `DRY_RUN`; một nấc một card |
| 9 | `ladder_report.py` — tổng hợp + bootstrap | 21 test pass; tái lập chính xác 45.0 / 53.2 từ dữ liệu thật |

**Ngoài plan** (phát sinh do sự cố hoặc câu hỏi mới):

- `multiseed_eval_driver.sh` — cứu mẻ multiseed sau khi driver gốc hỏng
- `probe_n_action_steps.sh` — điều tra khoảng cách 66.5 vs 87.3 so với paper SmolVLA
- Cập nhật `outputs/report_smolvla_en_vs_vi.md` với kết quả 3 seed, đánh dấu 4 chỗ đã
  bị bác bỏ thay vì xoá
- Task 14 — amend spec với 4 phát hiện

**Quy mô:** 1.394 dòng code mới, **101 test pass**, 9 commit ở `lerobot` + 2 ở
`smollm-vi`.

---

## 3. Tám lỗi phát hiện, mỗi lỗi đủ sức làm hỏng kết luận

### 3.1 Trong hạ tầng có sẵn

**`checkpoints/last` xuất hiện ở lần save đầu tiên, không phải lần cuối.**
`ft_multiseed_pipeline/driver.sh` kiểm tra hoàn thành bằng symlink đó, nên tuyên bố cả
6 job "DONE" ở khoảng bước 10.000. Hậu quả: 6 job chen 3 GPU (chậm ~1,9 lần), và một
lượt eval chạy trên checkpoint 20k gắn nhãn là kết quả cuối. Cùng lỗi đã ghi trong
report mục 3 về lượt đọc ft-10K. → Viết `multiseed_eval_driver.sh` với **hai** điều
kiện độc lập: thư mục bước tường minh `050000` tồn tại **và** không còn tiến trình
`lerobot-train` nào của job đó.

**Batch hiệu dụng thật của `d100` là 128, không phải 48.**
Cả model card lẫn chú thích trong `train_gpus.sh` đều ghi "48 (4 × 3 GPU × 4)". Đọc
`training_args.bin`: `per_device_batch=8`, `grad_accum=8`, `world_size=2` ⇒ **128**,
và chỉ 2 GPU dù thư mục tên `_3gpu_v2`. Kiểm chứng số học: 91.920 mẫu × 2 epoch / 128 =
1436 bước, đúng `global_step`. Phát hiện khi lượt `d10` đầu ra 382 bước thay vì ~143.
→ Huỷ lượt đó, pin `NUM_GPUS=2 / PER_DEVICE_BATCH=8 / GRAD_ACCUM=8`.

**`ddp_timeout` mặc định 1800s giết `d50`.**
Log: `ALLREDUCE ... ran for 1800012 milliseconds before timing out`, chết ở bước
109/718. Máy chạy `NCCL_SHM_DISABLE=1` (bug shared-memory trên topology multi-NUMA),
nên collective đi qua PCIe/socket và khựng được vài phút khi có job khác tranh bus.
→ Nâng lên 7200s; lượt chạy lại hoàn tất 718/718 với **0** sự kiện watchdog, dù chia
card với một eval.

**Split dev không hề held-out theo văn bản.**
Cam kết rời nhau của ViOCRVQA/OpenViVQA là về *image id*. Câu hỏi VQA rất công thức,
nên **21,08%** dòng dev trùng nguyên văn train — riêng ViOCRVQA 24.536 dòng. → Lọc
sạch kèm assertion, và thêm nguồn ngoài miền (`Vietnamese_sentiment`) để chỉ số đo
năng lực tiếng Việt chứ không phải thuộc lòng cách hỏi VQA.

### 3.2 Trong thiết kế của spec/plan

**`d0` cần khởi tạo sub-token, không dùng lại Arm C.**
Arm C dùng `resize_token_embeddings(mean_resizing=True)` của HF (rút mẫu chuẩn đa
biến); stage-1 dùng trung bình sub-token (tất định). Dùng lại Arm C sẽ trộn *liều
lượng* với *cách khởi tạo* ở đúng đoạn đầu đường cong. Khoảng cách không nhỏ: BPC
**7.97 vs 4.41**. → Task 6 dựng anchor riêng, qua đúng một pipeline với `d10/d25/d50`
(cờ `--no-adapter`).

**`warmup_steps 100` cố định phá liều thấp.**
Run đầy đủ 1436 bước ⇒ 100 bước warmup = 7%. Ở liều 10% chỉ còn ~142 bước ⇒ warmup
chiếm ~70%. → `warmup_ratio 0.07` (= 100/1436), `warmup_steps 0` vì HF cho nó quyền ưu
tiên.

### 3.3 Trong driver do tôi viết

**`sleep 180` sau khi phóng job là quá ngắn.** SmolVLA mất vài phút index dataset và
nạp backbone trước khi chạm GPU, nên vòng lặp kế tiếp thấy card vẫn trống. → Thay bằng
vòng chờ tới khi VRAM của card thực sự tăng (+2 GB, cap 20 phút, thoát sớm nếu job
chết).

**Ngưỡng VRAM là mô hình sai để lập lịch.** Kể cả sau khi job chiếm card, SmolVLA chỉ
dùng 6,3 GB nên card 24 GB vẫn còn 18 GB — vượt ngưỡng 9 GB, hàng đợi xếp tiếp nấc thứ
hai lên đó. Cả ba nấc dồn một card, **hai lần**. → `GPU_TAKEN` ghi nhận card driver đã
đặt job, trả lại khi nấc đó eval xong hoặc thất bại.

Chi phí hai lần phóng hỏng: ~4 GPU-giờ, không mất checkpoint nào.

Kèm theo: `DRY_RUN` ban đầu **không** chiếm card, nên nó báo cả ba nấc trên một GPU và
che đúng cái lỗi nó sinh ra để bắt. Đã sửa.

---

## 4. Câu hỏi mở

**Khoảng cách 66.5 vs 87.3 với paper SmolVLA (`arXiv:2506.01844`).**
Paper báo 87.3 trung bình cho SmolVLA 0.45B (90 spatial / 96 object / 92 goal / 71
long). Dự án đo `HuggingFaceVLA/smolvla_libero` được **66.5** dưới cùng giao thức.

Ba khác biệt cấu hình đã xác minh:

| | Paper | Dự án | Tỉ lệ |
|---|---|---|---|
| Pretrain action expert | 22,9K episode / 10,6M frame | **không có** | — |
| Bước × batch | 100k × 64 = 6,4M mẫu | 50k × 16 = 0,8M mẫu | **8× ít hơn** |

`run_vi.sh` không truyền `--policy.pretrained_path`, nên action expert khởi tạo ngẫu
nhiên. Điều này giải thích phần lớn khoảng cách tới model ta tự train (55.8), nhưng
**không** giải thích vì sao checkpoint đã finetune sẵn của họ chỉ đạt 66.5.

Ứng viên chính: `config.json` của checkpoint đó đặt **`n_action_steps: 1`** trong khi
mặc định LeRobot cho SmolVLA là 50 (= `chunk_size`). Ở mức 1, model dự đoán chunk 50
hành động rồi chỉ thực thi hành động đầu — gấp 50 lần forward và là chế độ điều khiển
khác hẳn với thực thi theo chunk mà nó được huấn luyện.

`probe_n_action_steps.sh` đang quét {1, 10, 25, 50} trên `libero_spatial` (100 episode,
chỉ dẫn tiếng Anh) để đối chiếu với mức 90 của paper.

**Tại sao quan trọng:** bằng chứng mạnh nhất cho tuyên bố của bài là dòng baseline
"EN 66.5 → VI 0.0". Nếu `n_action_steps` đúng đưa EN lên ~90 thì dòng đó được đo trong
chế độ suy giảm và nhánh VI phải đo lại — 0.0 vẫn sẽ là 0.0, nhưng từ một model thực
sự hoạt động.

---

## 5. Trạng thái và việc còn lại

| Task | Trạng thái |
|---|---|
| 1–9, 14 | ✅ hoàn tất |
| 10 — Phase 1a | ✅ 3 nấc (`stock`/`d0`/`d100`); 3 nấc còn lại **bỏ theo quyết định 2026-08-05** |
| 11 — Phase 2 (256M/2.2B) | chưa làm, ~30h, rủi ro OOM ở 2.2B |
| 12 — nhánh VI-train | khuyến nghị **bỏ** — hai đầu mút không phân biệt được thống kê |
| 13 — thêm seed | chưa làm, ~26h bản rút gọn |
| Probe `n_action_steps` | đang chạy |

**Lý do bỏ 3 nấc còn lại của Task 10:** `d10`/`d25`/`d50` nằm giữa `d0` = 0.0 và
`d100` = 0.0. Chúng chắc chắn cũng 0.0. Chạy tiếp là tiêu ~19 giờ để xác nhận một mặt
phẳng bằng không. Ladder BPC 5 điểm đã có vẫn dùng được, nhưng chỉ với cách đọc mới ở
mục 1.3′ — không phải "dose-response của việc học thêm" mà là "dose-response của việc
vá lại tổn thương".

**✅ Đã làm — mắt xích hở của bản báo cáo trước đã bịt xong (2026-08-06).** Chỉ số
ngữ nghĩa bất biến tokenizer (`diacritics`, mục 1.3′) thay cho đề xuất chạy
`run_evaluation.py` — rẻ hơn (không cần dữ liệu VQA ngoài, chạy trực tiếp trên corpus
đã có), và trả lời đúng câu hỏi: `d100` **không** giỏi tiếng Việt hơn `stock`. Phát
biểu "backbone giỏi tiếng Việt hơn hẳn mà vẫn ra 0.0" ở bản báo cáo trước — điều tôi
định lấy làm bằng chứng chính — **bị bác bỏ theo đúng nghĩa đen**, không phải "chưa
chứng minh được".

**Việc đáng làm nhất tiếp theo, không cần GPU:** viết lại phần định vị bài theo hai
trụ, không phải một:

1. *Train expert đúng ngôn ngữ là điều kiện cần* (mục 1.2, thu hẹp) — vẫn đứng.
2. *Continued-pretraining bằng LoRA + mở rộng vocab có thể làm hỏng năng lực ngôn ngữ
   mục tiêu ở quy mô nhỏ* (mục 1.3′) — phát hiện mới, độc lập với ladder LIBERO, tự nó
   đã là một đóng góp đáng công bố cho cộng đồng ngôn ngữ ít tài nguyên.

**Đối chứng bài mới còn cần:** eval `d100` với chỉ dẫn **tiếng Anh diễn đạt lại**.
Trước 2026-08-06 mục đích của nó là "chứng minh 0.0 là hiệu ứng ngôn ngữ". Giờ mục
đích khác đi: `stock` mới là backbone tiếng Việt "tốt nhất" trong ba, nên đối chứng
paraphrase nên chạy trên **`stock`**, không phải `d100` — để xác nhận baseline mạnh
nhất cũng vỡ vì đổi ngôn ngữ, chứ không phải vì đổi câu chữ nói chung. ~1,5h trên một
card.

---

## 6. Nguồn dữ liệu thô

- Chẩn đoán năng lực tiếng Việt (mục 1.3′):
  `vlai-experiments/vi-instructions/diagnose_vi_competence.py` (xem lịch sử git cho
  ba lần sửa BPC bị confound), kết quả `outputs/backbones/diacritics_forced_choice_d{0,10,25,50}.json`
  và `..._100.json`, phân vùng BPC mô tả `outputs/backbones/bpc_attribution.json`
- Phase 1a: `outputs/eval_ladder_{stock,d0,d100}_en_{en,vi}/eval_info.json`,
  checkpoint `outputs/train_ladder_*_en/`, log `outputs/ladder_pipeline/`
- Backbone liều: `outputs/backbones/vi_dose_{0,10,25,50}/` (kèm `validation.json`,
  `build_metadata.json`), mixture + manifest `outputs/backbones/mixtures/dose_*/`
- Stage-1 checkpoint: `~/Repository/smollm-vi/checkpoints/vietnamese_stage1_dose_*/`
- Multiseed: `outputs/eval_vi_ft_{smolvla,vi}_seed{2000,3000}/`,
  `outputs/eval_vi_vocab_only_ft_seed{2000,3000}/`,
  `outputs/ft_multiseed_pipeline/comparison_seed2000_3000.md`
- Corpus BPC: `vlai-experiments/vi-instructions/data/vi_bpc_corpus.txt`
- Báo cáo ladder: `outputs/report_backbone_ladder.md`
- Bối cảnh trước đó: `outputs/report_smolvla_en_vs_vi.md`
