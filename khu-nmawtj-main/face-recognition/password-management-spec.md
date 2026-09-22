# Spec: Quản lý mật khẩu (Password Management)

> Trạng thái: DRAFT — kết quả phỏng vấn người dùng (5 vòng, 2026-09-03). Chưa có code thay đổi.
> Mục tiêu của spec: làm rõ **"thêm xử lý cho việc quản lý mật khẩu"** — xử lý khi **quên mật khẩu**, chính sách mật khẩu **mạnh hơn**, và cải thiện **trải nghiệm xác thực**.

---

## 1. Bối cảnh / hiện trạng (đã đọc code)

| Thành phần | Vị trí | Hành vi hiện tại |
|---|---|---|
| Xác thực | `app/services/auth.py` | Argon2 hash lưu trong `config.json` (trường `password_hash`); `MIN_PASSWORD_LENGTH = 4`; sai **5 lần → khóa cứng 60 giây** (`MAX_ATTEMPTS=5`, `LOCKOUT_SECONDS=60`); `verify()` tự đếm lần sai, reset khi đúng. |
| Màn hình khóa | `app/ui/lock_screen.py` | 2 chế độ: chưa có mật khẩu → **đặt mật khẩu mới** (2 lần); đã có → **mở khóa**. Hiện đếm ngược khóa 60s bằng QTimer. Hint tĩnh: "Nhập sai 5 lần sẽ khóa 60 giây". |
| Xác thực thao tác nhạy cảm | `app/ui/password_dialog.py` | `PasswordDialog.require(auth, operation, note)` — dùng khi xóa/sửa người, xóa lịch sử, đổi mật khẩu, bật cloud. Nhập sai cũng chịu khóa 60s. |
| Đổi mật khẩu | `app/ui/settings_view.py` (`ChangePasswordDialog`, `_on_change_password`) | Yêu cầu mật khẩu **cũ** qua `PasswordDialog.require`, rồi nhập mật khẩu mới 2 lần; chỉ kiểm tra độ dài ≥ 4 và khớp nhau. |
| Cấu hình | `app/config.py` (`Config`) | `password_hash` nằm trong `config.json` (tách khỏi DB). Không có khái niệm quên mật khẩu, không có câu hỏi bảo mật, không có thời hạn. |
| Màn hình chính | `app/ui/main_window.py` | Luôn mở app bằng LockScreen; nút `🔒 Khoá` / `Ctrl+L` để khóa thủ công. |

**Lỗ hổng chính người dùng muốn xử lý:** mất mật khẩu = mất quyền truy cập app vĩnh viễn (chỉ chỉnh tay `config.json`); không có chính sách độ mạnh; nhập sai không biết còn mấy lần; khóa 60s có thể quá khắc nghiệt/không đủ chống dò.

---

## 2. Mục tiêu & ngoài phạm vi (theo phỏng vấn)

### Trong phạm vi
1. **Xử lý khi QUÊN mật khẩu** bằng **2 câu hỏi bảo mật tự đặt**.
2. **Chính sách mật khẩu mạnh hơn** cho mật khẩu mới.
3. **Cải thiện trải nghiệm xác thực**: đếm lần thử còn lại, khóa mềm, hiện/ẩn mật khẩu.
4. Mật khẩu duy nhất cho cả app (KHÔNG làm nhiều tài khoản/người dùng).
5. Mật khẩu áp dụng ở máy đơn; có tính đến mối liên hệ dữ liệu cloud (xem mục 8 — **chưa đồng bộ auth** ở bản này).

### Ngoài phạm vi (đã chốt KHÔNG làm)
- Nhiều hồ sơ/mật khẩu theo người dùng.
- Chặn trùng mật khẩu cũ / lưu lịch sử mật khẩu.
- Hết hạn mật khẩu định kỳ (vd 90 ngày).
- Recovery key / chuỗi khôi phục dự phòng.
- Backdoor kiểu "reset cứng xóa mật khẩu" (không có nút xóa mật khẩu từ UI).
- Khôi phục qua email/điện thoại (app desktop offline).

---

## 3. Yêu cầu chức năng (FR)

### FR-1 — Chính sách mật khẩu MỚI
- Mật khẩu **mới** (đặt lần đầu / đổi / đặt lại khi quên) phải thỏa:
  - tối thiểu **8 ký tự**;
  - có **chữ thường** + **chữ hoa** + **chữ số** (không bắt buộc ký tự đặc biệt);
  - KHÔNG dùng danh sách đen mật khẩu phổ biến (đã chốt — chỉ áp quy tắc trên).
- **Mật khẩu cũ yếu (ngắn, đã đặt trước bản này) VẪN dùng được bình thường** — không bắt đổi, không chặn mở khóa. Chính sách mới chỉ áp dụng cho mật khẩu được tạo/đổi sau khi nâng cấp.
- Validation dùng chung 1 hàm (ví dụ `validate_password_strength(pw) -> list[str]` lỗi) — cả LockScreen, ChangePasswordDialog, luồng khôi phục đều gọi.
- Hiện thanh **độ mạnh** (Yếu/Vừa/Mạnh) khi nhập mật khẩu mới — chỉ hiển thị, không chặn thêm ngoài FR-1.

### FR-2 — Khóa MỀM thay khóa cứng 60 giây
- Bỏ khóa cứng "5 lần sai → khóa 60s".
- Thay bằng: sau **5 lần sai liên tiếp**, mỗi lần thử tiếp theo phải chờ **10 giây** (cooldown giữa 2 lần nhập) — vẫn cho nhập, chỉ làm chậm kẻ dò.
- Bộ đếm lần sai **reset khi xác thực đúng**.
- **Dùng CHUNG 1 bộ đếm** cho mọi đường xác thực: mật khẩu ở màn hình khóa, `PasswordDialog` (thao tác nhạy cảm), và trả lời câu hỏi bảo mật (đã chốt) — không được dùng đường khác để né cooldown.
- `AuthService` thay `_locked_until`/`is_locked` bằng trạng thái cooldown mới (giữ tên API càng tương thích càng tốt nhưng thay ngữ nghĩa; cập nhật mọi nơi gọi `is_locked`/`lockout_remaining`).

### FR-3 — Hiển thị số lần thử còn lại
- Khi nhập sai (chưa chạm ngưỡng): hiện **"Sai mật khẩu — còn N lần thử"** ở màn hình khóa và `PasswordDialog`.
- Khi đang trong cooldown: hiện đếm ngược **"Thử lại sau X giây"** (như lock screen hiện có, giữ QTimer đếm ngược).

### FR-4 — Hiện/ẩn mật khẩu (eye toggle)
- Thêm nút 👁 hiện/ẩn cho **tất cả** ô mật khẩu:
  - LockScreen (ô đặt mới + ô mở khóa);
  - `PasswordDialog` (xác thực thao tác nhạy cảm);
  - `ChangePasswordDialog` (ô mới + ô nhập lại);
  - luồng khôi phục (ô mật khẩu mới).
- Mặc định ẩn (EchoMode.Password), click giữ/đổi chế độ hiển thị.

### FR-5 — Câu hỏi bảo mật (2 câu)
- Khi **đặt mật khẩu lần đầu** (LockScreen setup): bắt buộc tạo **2 câu hỏi + 2 câu trả lời**.
- Mỗi câu trả lời được **hash argon2** riêng (không lưu plaintext) — tương tự cách hash mật khẩu.
- So khớp khi khôi phục (đã chốt): chuẩn hóa = **trim khoảng trắng thừa + lowercase**, **KHÔNG bỏ dấu tiếng Việt** ("Hà Nội" ≠ "ha noi") rồi verify hash.
- 2 câu hỏi phải khác nhau; câu trả lời tối thiểu 3 ký tự sau chuẩn hóa.
- Khi **đổi mật khẩu**: giữ nguyên câu hỏi cũ; có nút phụ **"Đổi câu hỏi bảo mật"** (yêu cầu xác thực mật khẩu cũ trước).

### FR-6 — Khôi phục khi quên mật khẩu (CHỈ ở màn hình khóa)
- Vị trí: duy nhất **màn hình khóa** (đã chốt — KHÔNG đặt ở PasswordDialog thao tác nhạy cảm, KHÔNG ở trang Cài đặt).
- Nâng cấp người dùng cũ (đã chốt): nếu `config.json` có `password_hash` nhưng THIẾU câu hỏi bảo mật → lần mở app ĐẦU TIÊN sau nâng cấp, LockScreen thêm bước **bắt buộc khai 2 câu hỏi** trước khi vào màn hình chính (vẫn dùng mật khẩu cũ để mở khóa). Phòng thủ: nếu dữ liệu câu hỏi vẫn thiếu (config chỉnh tay), ẩn link "Quên mật khẩu?".
- Luồng: link **"Quên mật khẩu?"** → màn nhập trả lời **cả 2 câu hỏi** (câu hỏi hiển thị sẵn) → trả lời đúng **cả 2** → cho đặt mật khẩu mới (theo FR-1) và tùy chọn cập nhật lại câu hỏi bảo mật.
- Trả lời sai cũng tính vào bộ đếm lần sai / chịu cooldown FR-2 (không bypass).
- Nếu quên cả câu trả lời → không có đường khôi phục (chấp nhận; ghi rõ trong hint).

### FR-7 — Tự khóa khi không dùng (idle)
- Thêm tùy chọn trong Cài đặt → BẢO MẬT: **Tắt / 1 phút / 5 phút / 15 phút** (mặc định **Tắt** — giữ hành vi hiện tại).
- Khi bật: không thấy tương tác (bàn phím/chuột trong cửa sổ chính) quá thời gian quy định → tự chuyển sang LockScreen. Bộ đếm reset khi có tương tác; **vẫn đếm cả khi camera đang chạy** (đã chốt — chỉ phím/chuột mới reset); không đếm thời gian khi app đang ở LockScreen.

### FR-8 — Lưu trữ (giữ trong `config.json`)
- Không tách file `auth.json` (đã chốt giữ nguyên cấu trúc).
- `Config` thêm trường mới:
  - `security_question_1: str`, `security_question_2: str` (câu hỏi, lưu thường — là văn bản hiển thị);
  - `security_answer_hash_1: str | None`, `security_answer_hash_2: str | None` (hash argon2 của câu trả lời);
  - `idle_lock_minutes: int = 0` (0 = tắt tự khóa);
  - (giữ `password_hash`).
- Config cũ thiếu field mới → `Config.load` phải tự điền mặc định an toàn (không crash); người dùng có mật khẩu cũ nhưng chưa có câu hỏi bảo mật: xử lý "nâng cấp" — xem mục 10.

---

## 4. Luồng chi tiết

### 4.1 Tạo mật khẩu lần đầu (LockScreen setup)
1. Nhập mật khẩu mới + nhập lại (2 ô) — validate FR-1 + khớp nhau.
2. Tạo **2 cặp câu hỏi/trả lời** (validate FR-5).
3. Lưu: `password_hash`, `security_question_1/2`, `security_answer_hash_1/2` → `config.save()`.
4. Mở khóa thành công, vào màn hình chính.

### 4.2 Mở khóa bình thường
1. Nhập mật khẩu.
2. Sai → hiện "còn N lần thử" (FR-3); đủ 5 sai liên tiếp → cooldown 10s giữa lần thử (FR-2) + đếm ngược.
3. Đúng → reset bộ đếm, vào app.
4. Quên mật khẩu → link "Quên mật khẩu?" (FR-6).

### 4.3 Đổi mật khẩu (Settings → BẢO MẬT)
1. `PasswordDialog.require` mật khẩu **cũ** (giữ hiện trạng).
2. `ChangePasswordDialog`: nhập mới 2 lần, validate FR-1; có eye toggle.
3. Nút "Đổi câu hỏi bảo mật" (tùy chọn, cần mật khẩu cũ).
4. Lưu hash mới; trạng thái "Mật khẩu: đã đặt" cập nhật.

### 4.4 Khôi phục quên mật khẩu (LockScreen)
1. Bấm "Quên mật khẩu?" → hỏi trả lời 2 câu (đã hiển thị câu hỏi).
2. Trả lời đúng cả 2 → form đặt mật khẩu mới (FR-1) → lưu `password_hash` mới (câu hỏi giữ nguyên hoặc cập nhật theo lựa chọn).
3. Sai → lỗi + áp dụng đếm lần sai/cooldown như mở khóa.

### 4.5 Xác thực thao tác nhạy cảm (PasswordDialog)
- Không đổi vị trí/ngữ cảnh; chỉ áp dụng: eye toggle, "còn N lần thử", cooldown 10s thay vì khóa 60s.

---

## 5. Thay đổi dự kiến theo file (chưa code)

- `app/services/auth.py`:
  - hằng số mới: `MIN_PASSWORD_LENGTH = 8` (thay 4), yêu cầu chữ thường/hoa/số; bỏ `MAX_ATTEMPTS`/`LOCKOUT_SECONDS` cứng hoặc thay bằng `SOFT_LOCK_AFTER = 5`, `SOFT_LOCK_WAIT_SECONDS = 10`;
  - hàm `validate_password_strength(pw) -> list[str]` (danh sách lỗi tiếng Việt);
  - `set_password(pw, q1=None, a1=None, q2=None, a2=None)` — hash câu trả lời, ném lỗi nếu vi phạm chính sách;
  - `verify_answers(a1, a2) -> bool` (chuẩn hóa + verify 2 hash);
  - trạng thái cooldown: `is_locked`/`lockout_remaining` giữ tên nhưng theo ngữ nghĩa cooldown 10s (cập nhật UI dùng).
- `app/ui/lock_screen.py`: form 2 câu hỏi bảo mật lúc setup; link "Quên mật khẩu?"; màn khôi phục; eye toggle; hiển thị "còn N lần"; đếm ngược 10s.
- `app/ui/password_dialog.py`: eye toggle, hiện lần thử còn lại, thông báo cooldown.
- `app/ui/settings_view.py`: `ChangePasswordDialog` validate mới + eye; mục "Đổi câu hỏi bảo mật"; combo `idle_lock_minutes`.
- `app/ui/main_window.py`: QTimer tự khóa theo idle (đọc `config.idle_lock_minutes`); reset khi tương tác.
- `app/config.py`: thêm field mới (FR-8); giữ tương thích config cũ.
- Copy/hint tiếng Việt cập nhật: bỏ "sẽ khóa 60 giây", thêm nội dung mới.

---

## 6. Bản sao nội dung (copy) dự kiến (tiếng Việt)
- Sai mật khẩu: `Sai mật khẩu — còn {n} lần thử`
- Cooldown: `Quá nhiều lần thử sai — thử lại sau {x}s`
- Link quên: `Quên mật khẩu?`
- Tiêu đề khôi phục: `Khôi phục mật khẩu` / `Trả lời 2 câu hỏi bảo mật để đặt lại mật khẩu`
- Chính sách (hiển thị khi nhập mới): `Tối thiểu 8 ký tự, gồm chữ thường, chữ hoa và chữ số`
- Cảnh báo: `Không đặt lại được mật khẩu nếu quên câu trả lời — hãy lưu câu hỏi/trả lời ở nơi an toàn`

---

## 7. Ràng buộc & nguyên tắc
- Không lưu mật khẩu hay câu trả lời bảo mật dạng plaintext (chỉ hash argon2).
- Không thêm backdoor / reset cứng từ UI.
- Chính sách mới không phá vỡ người dùng hiện hữu (mật khẩu cũ yếu vẫn dùng được).
- Desktop Windows, chạy từ mã nguồn lẫn `.exe` đóng gói (cấu hình ở `APP_DIR/config.json`).

---

## 8. Mối liên hệ cloud (mở — triển khai sau)
- Người dùng ban đầu chọn "đồng bộ dùng chung nhiều máy" nhưng **chưa chốt cơ chế** → spec ghi nhận là việc làm SAU:
  - Hiện tại: **KHÔNG đồng bộ** `password_hash` / câu hỏi bảo mật lên Cloudflare D1 (chỉ đồng bộ dữ liệu người/lịch sử).
  - Khi làm sau: cần quyết định (a) máy mới phải xác thực mật khẩu máy đầu để "nhận quyền", (b) lưu hash ở D1 như bản sao lưu, hay (c) giữ local tuyệt đối.
  - Rủi ro đã biết: đưa hash + câu hỏi bảo mật lên cloud làm tăng bề mặt lộ; câu hỏi bảo mật thường kém entropy hơn mật khẩu → cần cân nhắc riêng.

---

## 9. Các quyết định đã chốt (thay cho câu hỏi mở)
| # | Vấn đề | Quyết định (2026-09-03) | Ghi chú áp dụng |
|---|---|---|---|
| 1 | Dấu tiếng Việt khi so khớp câu trả lời | **KHÔNG bỏ dấu** — chỉ trim + lowercase | "Hà Nội" ≠ "ha noi"; đúng chuỗi người dùng gõ lúc đặt |
| 2 | Người dùng cũ chưa có câu hỏi bảo mật | **Bắt buộc khai NGAY** ở lần mở app đầu sau nâng cấp (thêm bước trong LockScreen) | Vẫn mở khóa bằng mật khẩu cũ; xong bước khai câu hỏi mới vào app |
| 3 | Blacklist mật khẩu phổ biến | **Không dùng** | Chỉ áp quy tắc FR-1 (8 ký tự + thường/hoa/số) |
| 4 | Idle lock khi camera đang chạy | **Vẫn đếm** | Bất kỳ phím/chuột nào cũng reset bộ đếm |
| 5 | Bộ đếm sai/cooldown giữa các đường xác thực | **Dùng chung 1 bộ đếm** | Mật khẩu (khóa + dialog) và trả lời bảo mật tính chung; không né được qua đường khác |

---

## 10. Định nghĩa "hoàn thành" (acceptance criteria)
- [ ] Tạo mật khẩu mới phải tuân FR-1; mật khẩu cũ ngắn vẫn mở khóa được.
- [ ] Sau 5 lần sai → các lần thử sau phải chờ 10s; không còn khóa 60s; đúng mật khẩu thì reset đếm.
- [ ] Màn hình khóa hiển thị "còn N lần thử" và đếm ngược cooldown.
- [ ] Eye toggle hoạt động ở mọi ô mật khẩu (FR-4).
- [ ] Đặt lần đầu bắt buộc 2 câu hỏi bảo mật; trả lời lưu dạng hash.
- [ ] "Quên mật khẩu?" chỉ ở màn hình khóa; trả lời đúng cả 2 câu → đặt lại được mật khẩu (đúng FR-1); sai vẫn chịu cooldown (cùng bộ đếm).
- [ ] So khớp câu trả lời: trim + lowercase, KHÔNG bỏ dấu tiếng Việt.
- [ ] Nâng cấp: user cũ (có mật khẩu, thiếu câu hỏi) bị yêu cầu khai 2 câu hỏi ở lần mở app đầu sau nâng cấp trước khi vào app.
- [ ] Cài đặt có mục tự khóa (Tắt/1/5/15 phút, mặc định Tắt) và hoạt động đúng — vẫn đếm khi camera chạy, chỉ phím/chuột reset.
- [ ] `config.json` cũ không crash khi thiếu field mới; dữ liệu người/lịch sử không đổi.
- [ ] Không có bất kỳ thay đổi nào về luồng đồng bộ D1 hiện tại (auth không lên cloud ở bản này).
