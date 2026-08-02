# Thực nghiệm tiếp theo để có đóng góp cho paper

Tài liệu này khác với `phan_tich_va_huong_di_tiep.md`: bản trước hỏi *"kết quả hiện tại có đứng vững
không"*, bản này hỏi *"chạy gì để có đóng góp mới đáng công bố"*. Đã đối chiếu với văn liệu VLA đa
ngôn ngữ 2026 trước khi đề xuất, vì định vị novelty quyết định thí nghiệm nào đáng chạy.

---

## 0. Tin quan trọng trước tiên: đã có người làm VLA đa ngôn ngữ trên LIBERO

Ba bài 2026 liên quan trực tiếp, cần đọc kỹ trước khi viết:

**`arXiv:2606.11906` — "When Does Language Matter? Multilingual Instructions Reveal Step-wise
Language Sensitivity in VLA Models"**
Dịch LIBERO ra 10 ngôn ngữ — **bao gồm tiếng Việt**. Chạy trên OpenVLA-OFT và π₀.₅. Đề xuất can
thiệp lúc suy luận (inference-time), không train lại.

**`arXiv:2606.15714` — "Beyond English: Uncovering the Multilingual Gap in VLA Models"**
5 ngôn ngữ (Anh, Trung, Pháp, Nga, Ả Rập — *không có tiếng Việt*). Đề xuất MPCA, căn chỉnh biểu diễn
đa ngôn ngữ bằng PCA. Có nhánh can thiệp phía train (M-FT / M-CT: finetune VLM đa ngôn ngữ rồi mới
train action head).

**`arXiv:2602.17659` — "When Vision Overrides Language" (LIBERO-CF)**
Benchmark phản thực cho LIBERO: gán chỉ dẫn thay thế dưới bố cục cảnh hợp lý về mặt thị giác. Đúng
chẩn đoán tôi đề xuất ở mục 3.1 của tài liệu trước — **đã tồn tại, chỉ tiếng Anh**.

### Điều này đổi gì

**Tin xấu:** "VLA sụp đổ khi đổi sang chỉ dẫn không phải tiếng Anh" **không còn là đóng góp mới**.
Đã được công bố, với nhiều model hơn và nhiều ngôn ngữ hơn. Nếu bài viết đóng khung quanh phát hiện
này, nó sẽ bị đánh giá là trùng lặp.

**Tin tốt — và đây mới là điểm cần bám vào:** không bài nào trong ba bài trên **thay backbone**.
`2606.15714` nói rõ rằng dù backbone VLM có khả năng đa ngôn ngữ, pipeline train VLA hiện nay vẫn
gần như hoàn toàn đơn ngữ, khiến việc căn chỉnh ngôn ngữ-hành động thiên lệch về tiếng Anh. Cách
khắc phục của cả ba đều là **hậu kỳ**: căn chỉnh biểu diễn, can thiệp lúc suy luận, hoặc finetune đa
ngôn ngữ. **Không ai hỏi: nếu ngay từ đầu dùng backbone đơn ngữ của chính ngôn ngữ đó thì sao?**

Đó chính xác là điều Arm B của bạn làm. Và Arm C (backbone tiếng Anh + vocab tiếng Việt, không
pretrain) là đối chứng tách bạch mà **không bài nào có** — nó là thứ phân biệt "cần đúng vocab" với
"cần hiểu ngôn ngữ thật", và bạn đã có dữ liệu 3-seed cho nó.

### Phát hiện thứ hai, đáng chú ý hơn về mặt khoa học

`2606.11906` báo cáo mức tụt **30–50%** dưới chỉ dẫn không phải tiếng Anh. Bạn đo được **sụp đổ về
đúng 0.0%** — trên 400 episode, cả 4 suite, ở 4 checkpoint độc lập. Đây không phải cùng một hiện
tượng ở mức độ khác nhau; đây là **hai chế độ thất bại khác hẳn nhau về bản chất**.

Khác biệt gần như chắc chắn nằm ở chỗ: model của họ (OpenVLA-OFT, π₀.₅) dùng backbone **đa ngôn ngữ
quy mô lớn** đã thấy tiếng Việt lúc pretrain — nên tụt nhưng vẫn còn phần nào hoạt động. SmolVLM2-500M
của bạn thực chất **đơn ngữ tiếng Anh** — nên sụp đổ hoàn toàn. Nếu đúng, đây là một phát biểu có thể
kiểm chứng và chưa ai công bố:

> **Khả năng đa ngôn ngữ của backbone không suy giảm dần theo quy mô mô hình — nó có một ngưỡng vách.
> Dưới ngưỡng đó, chỉ dẫn ngoài ngôn ngữ huấn luyện không phải "hiểu kém" mà là hoàn toàn ngoài phân
> phối.**

Đây là hướng đóng khung mạnh nhất mà dữ liệu hiện tại của bạn ủng hộ, và mục E4 dưới đây là thí
nghiệm để chứng minh nó.

---

## 1. Định vị: bài báo nên nói điều gì

Đừng viết *"VLA thất bại với tiếng Việt"* (đã có người viết rồi). Hãy viết:

> **Với VLA quy mô nhỏ, năng lực ngôn ngữ phải nằm sẵn trong backbone — không thể vá bằng vocab,
> không thể vá lúc suy luận, và không thể vá bằng dữ liệu finetune hẹp miền.**

Ba trụ đỡ cho tuyên bố này, xếp theo mức độ sẵn sàng:

1. **Vá vocab là không đủ** — Arm C, đã có 3 seed, p = 0.027. *Sẵn sàng.*
2. **Vá lúc suy luận là không đủ ở quy mô nhỏ** — cần E6, so trực tiếp với phương pháp của
   `2606.11906`. *Chưa có.*
3. **Có sẵn năng lực trong backbone là đủ** — Arm B, nhưng cần E1/E2/E3 để phòng thủ. *Một phần.*

Điều làm bài của bạn không trùng: hai bài kia đo *khoảng cách*, bạn đo *cách vá khoảng cách từ phía
train* — và cho thấy chỉ có một cách vá hiệu quả. Cộng thêm tiếng Việt là ngôn ngữ ít tài nguyên mà
`2606.15714` bỏ trống hoàn toàn, còn `2606.11906` chỉ đưa vào như 1 trong 10 ngôn ngữ eval, không
phân tích riêng.

---

## 2. Danh sách thí nghiệm, xếp theo thứ tự nên chạy

Ba mục đầu (E1–E3) là **bắt buộc để bảo vệ kết quả đã có** — đã nêu ở tài liệu trước, nhắc lại ngắn
gọn. Ba mục sau (E4–E6) là **đóng góp mới cho paper**.

### E1 — Battery phản thực (~4h GPU) · bắt buộc

Chạy `smolvla-vi-ft-50k` với chỉ dẫn: gốc / rỗng / xáo trộn / diễn đạt lại. Chi tiết ở
`phan_tich_va_huong_di_tiep.md` mục 3.1. **Điểm mới:** giờ nên theo giao thức LIBERO-CF
(`2602.17659`) thay vì tự chế, để so sánh được với văn liệu. Reviewer sẽ hỏi tại sao không dùng
benchmark phản thực đã có.

Bổ sung một điều kiện mà LIBERO-CF không có: **chỉ dẫn tiếng Việt xáo trộn** (chỉ dẫn của task khác,
đúng ngôn ngữ). Tách bạch "sai ngôn ngữ" khỏi "sai nội dung" — chưa ai làm.

### E2 — Phân tích per-task trên dữ liệu đã có (~0h GPU) · bắt buộc

Chênh 28 điểm ở `libero_object` đến từ mấy task? Bỏ suite đó ra thì gap còn +1.7 điểm. Paired
bootstrap trên 40 task. Chi tiết ở tài liệu trước, mục 3.2.

### E3 — Ma trận 2×2 backbone × ngôn ngữ dữ liệu (~34h GPU) · bắt buộc

Lấp ô "backbone EN × dữ liệu EN". Không có ô này, "backbone VI tốt hơn" không tách được khỏi
"backbone khớp ngôn ngữ dữ liệu thì tốt hơn". Chi tiết ở tài liệu trước, mục 3.3.

### E4 — Vách sụp đổ theo quy mô backbone (~40h GPU) · ĐÓNG GÓP MỚI

**Câu hỏi:** tại sao bạn thấy 0% còn `2606.11906` thấy tụt 30–50%? Giả thuyết: năng lực đa ngôn ngữ
của backbone có ngưỡng vách, không suy giảm tuyến tính theo quy mô.

**Thiết kế:** cùng recipe SmolVLA, thay backbone theo thang năng lực tiếng Việt tăng dần:

| Backbone | Quy mô | Tiếng Việt lúc pretrain |
|---|---|---|
| SmolVLM2-500M (EN) | 500M | không (Arm A hiện có) |
| SmolVLM2-500M-vi (Arm B hiện có) | 500M | có, đơn ngữ |
| Qwen2-VL-2B | 2B | có, đa ngôn ngữ |
| Qwen2.5-VL-3B hoặc PaliGemma-3B | 3B | có, đa ngôn ngữ |

Eval bằng tiếng Việt, tất cả zero-shot cross-lingual (train tiếng Anh, test tiếng Việt) **và** train
tiếng Việt. Hai điểm đầu đã có sẵn, chỉ cần chạy thêm 2 backbone.

**Vì sao đáng công bố:** nếu 500M cho 0% còn 2B cho 40%, bạn đã định vị được vách — và giải thích
được vì sao kết quả của bạn khác kết quả họ công bố. Đây là biến chỗ *"số của tôi khác số của họ"*
(điểm yếu) thành *"tôi giải thích được vì sao chúng khác nhau"* (đóng góp). Cũng đưa ra khuyến nghị
dùng được: dưới ngưỡng nào thì buộc phải có backbone chuyên ngôn ngữ.

### E5 — Đường cong hiệu quả dữ liệu (~90h GPU) · ĐÓNG GÓP MỚI, giá trị cao nhất

**Câu hỏi thực tiễn mà chưa bài nào trả lời:** cần bao nhiêu dữ liệu robot tiếng Việt để bù cho việc
không có backbone tiếng Việt?

**Thiết kế:** train Arm A (backbone EN) và Arm B (backbone VI) trên 5%, 10%, 25%, 50%, 100% dataset
LIBERO tiếng Việt. Vẽ hai đường cong success-vs-data.

Ba kịch bản đọc kết quả:
- **Hai đường hội tụ ở 100%** → lợi thế backbone chỉ là hiệu quả dữ liệu, tuyên bố cần dịu đi, nhưng
  vẫn có ích: *"backbone VI tiết kiệm được N giờ demo"* — đây là con số kỹ sư thực sự cần.
- **Hai đường song song** → lợi thế backbone là hằng số, không mua được bằng dữ liệu. Tuyên bố mạnh
  nhất.
- **Hai đường phân kỳ** → backbone VI mở rộng tốt hơn theo dữ liệu. Mạnh hơn nữa.

Mọi kịch bản đều cho ra kết quả đáng công bố — đó là dấu hiệu của một thí nghiệm được thiết kế tốt.
Với ngôn ngữ ít tài nguyên, "cần bao nhiêu dữ liệu" chính là câu hỏi trung tâm, vì thu thập demo
tiếng Việt đắt hơn nhiều so với tải một backbone.

Nếu ngân sách hẹp, cắt còn 3 mốc (10%, 50%, 100%) × 2 arm × 1 seed ≈ 45h.

### E6 — So với can thiệp lúc suy luận (~20h GPU) · ĐÓNG GÓP MỚI

**Câu hỏi:** phương pháp inference-time của `2606.11906` và MPCA của `2606.15714` có cứu được
SmolVLA 500M không, hay ở quy mô này bắt buộc phải đổi backbone?

**Thiết kế:** áp can thiệp lúc suy luận của họ lên checkpoint Arm A của bạn, eval bằng tiếng Việt.
So ba mức: Arm A thô (0%) → Arm A + can thiệp (?) → Arm B (53.2%).

**Vì sao đáng công bố:** đây là thí nghiệm biến bài của bạn từ "một điểm dữ liệu nữa" thành "phản
biện có căn cứ với văn liệu hiện có". Nếu can thiệp lúc suy luận đưa 0% lên 5%, còn đổi backbone đưa
lên 53.2%, bạn có bằng chứng trực tiếp rằng **ở quy mô nhỏ, vá hậu kỳ là không đủ** — chính là trụ
đỡ số 2 của tuyên bố ở mục 1. Đây cũng là mục reviewer nhiều khả năng đòi hỏi nhất nếu bài trích dẫn
hai công trình kia.

---

## 3. Hai lộ trình theo ngân sách

**Tối thiểu để có bài chắc chắn (~60h GPU):** E2 → E1 → E3 → E6.
Cho ra một bài phân tích chặt chẽ, phòng thủ tốt, có phản biện trực tiếp với văn liệu. Thiếu phần
"tại sao số của tôi khác số của họ".

**Đủ để có bài mạnh (~190h GPU):** thêm E4 (40h) và E5 (90h) vào lộ trình trên.
Nếu dùng bản E5 rút gọn 3 mốc (~45h) thì tổng còn ~145h.
E4 giải thích khác biệt 0% vs 30–50%; E5 cho khuyến nghị thực tiễn dùng được. Đây là điểm mà bài
chuyển từ "workshop paper" sang "conference paper".

Nếu chỉ chạy được **một** thí nghiệm mới: chọn **E5**. Nó cho ra kết quả đáng công bố ở mọi kịch bản,
trả lời câu hỏi mà cộng đồng ngôn ngữ ít tài nguyên thực sự quan tâm, và không phụ thuộc vào việc
tái lập được phương pháp của người khác (rủi ro chính của E6).

## 4. Việc cần làm ngay, không tốn GPU

1. **Đọc kỹ cả ba bài** — đặc biệt phần setup của `2606.11906` (họ dùng backbone nào, dịch thế nào,
   tại sao ra 30–50% chứ không phải 0%).
2. **Kiểm tra trùng lặp chỉ dẫn tiếng Việt**: bản dịch LIBERO tiếng Việt của họ so với
   `VLAIResearchLab/lerobot_libero_vi` của bạn. Nếu khác nhau nhiều, đó là biến gây nhiễu cần nêu;
   nếu gần giống, có thể so sánh trực tiếp số liệu — rất có lợi.
3. **Đóng khung lại phần mở đầu** theo mục 1 trước khi chạy thêm thí nghiệm — biết bài định nói gì
   sẽ quyết định thí nghiệm nào đáng chạy, không phải ngược lại.
