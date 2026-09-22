# 📖 HƯỚNG DẪN MỞ & CHẠY ỨNG DỤNG ĐIỂM DANH NHẬN DIỆN KHUÔN MẶT

Ứng dụng desktop (PySide6) điểm danh nhân viên bằng nhận diện khuôn mặt, lưu dữ liệu trên **SQL Server**.

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 10 / 11 (64-bit) |
| Python | **3.10 hoặc 3.11** (khuyên dùng — insightface cần bản có wheel sẵn) |
| Database | SQL Server Express + **ODBC Driver 18 for SQL Server** |
| Camera | Webcam hoạt động bình thường |
| GPU | Không bắt buộc (dùng DirectML, chạy được cả máy không có GPU rời) |
| Internet | Cần ở lần chạy đầu (tự tải model buffalo_l ~350MB) |

Tải về:
- Python: https://www.python.org/downloads/ (khi cài, tích ✅ **Add Python to PATH**)
- SQL Server Express: https://www.microsoft.com/vi-vn/sql-server/sql-server-downloads
- ODBC Driver 18: https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server

---

## 2. Cài đặt lần đầu (chỉ làm 1 lần)

Mở **PowerShell** hoặc **CMD**, di chuyển vào thư mục app:

```powershell
cd face-recognition
```

### 2.1. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2.2. Kích hoạt môi trường ảo

```powershell
.venv\Scripts\activate
```

### 2.3. Cài thư viện

```powershell
pip install -r requirements.txt
```

> ⏳ Bước này mất 5–15 phút tùy mạng (gói `insightface`, `onnxruntime-directml` khá nặng).

### 2.4. Chuẩn bị database SQL Server

1. Mở **SQL Server Management Studio (SSMS)**, đăng nhập vào `LAPTOP-OBLS1HOE\SQLEXPRESS` (hoặc tên server của bạn — xem lệnh `hostname` trong CMD).
2. Tạo database tên **`FaceRecognitionDB`** (nếu chưa có):

```sql
CREATE DATABASE FaceRecognitionDB;
```

3. Kiểm tra kết nối từ Python (chạy ở thư mục gốc project):

```powershell
cd ..
python test_sqlserver.py
```

> ✅ Kết nối thành công khi in ra `KẾT NỐI SQL SERVER THÀNH CÔNG`.
> ⚠️ Nếu tên máy bạn khác `LAPTOP-OBLS1HOE`, sửa biến `SERVER` trong `test_sqlserver.py` và cấu hình kết nối của app cho khớp.

---

## 3. Chạy ứng dụng

### Cách 1 — Nhấp đúp file `run.bat` (dễ nhất) ⭐

Vào thư mục `face-recognition`, **nhấp đúp** vào `run.bat`. File này tự:
- Kiểm tra và cài `pyodbc` nếu thiếu
- Khởi động app

### Cách 2 — Chạy bằng lệnh

```powershell
cd face-recognition
.venv\Scripts\activate
python -m app.main
```

### Lần chạy đầu tiên

- App **tự tải model nhận diện `buffalo_l`** (khoảng 350MB) vào thư mục `models/` — cần internet, chỉ xảy ra một lần.
- File `config.json` tự được tạo để lưu cài đặt (ngưỡng nhận diện, hash mật khẩu, cài đặt camera...).
- Đặt **mật khẩu quản lý** khi được hỏi — mật khẩu được băm bằng argon2, không lưu dạng thô.

---

## 4. Sử dụng nhanh

1. **Đăng nhập** bằng mật khẩu đã đặt (có câu hỏi bảo mật khôi phục nếu quên).
2. **Đăng ký khuôn mặt**: vào mục Nhân viên → Thêm nhân viên → chụp/nạp ảnh khuôn mặt.
3. **Điểm danh**: vào tab Camera → nhìn vào webcam → hệ thống tự nhận diện và ghi nhận giờ vào/ra.
4. **Chấm công & lương**: xem lịch sử điểm danh, tính lương ở tab tương ứng.
5. **Cài đặt**: đổi camera, ngưỡng nhận diện, dark/light theme, tự khóa khi không dùng.

---

## 5. Xử lý sự cố thường gặp

| Lỗi | Cách khắc phục |
|---|---|
| `Cannot find .venv\Scripts\python.exe` | Chưa tạo venv — làm lại mục **2.1** |
| `ODBC Driver 18 not found` | Cài ODBC Driver 17/18 (link ở mục 1), khởi động lại máy |
| Không kết nối được SQL Server | Kiểm tra dịch vụ **SQL Server (SQLEXPRESS)** đang chạy trong `services.msc`; sửa `SERVER` trong code theo tên máy bạn |
| Qt plugin / màn hình trắng | Đã có sẵn fix trong `run.bat` và `app/main.py`; nếu chạy tay thì set `QT_PLUGIN_PATH` trước khi chạy |
| Model tải chậm / lỗi mạng | Tải lại bằng cách xóa thư mục `models/` rồi chạy lại app |
| Camera đen / không mở được | Đổi `camera_index` (0, 1, 2...) trong file `config.json` hoặc trong Cài đặt của app |
| `pip install insightface` báo lỗi build | Dùng Python **3.10 hoặc 3.11**; cài **Microsoft C++ Build Tools** nếu vẫn lỗi |
| Quên mật khẩu | Trả lời 2 câu hỏi bảo mật đã thiết lập khi đăng ký |

---

## 6. Cấu trúc thư mục

```
khu-nmawtj-main/
├── .gitignore                  # Loại trừ venv, data, model... khỏi git
├── test_sqlserver.py           # Script kiểm tra kết nối SQL Server
└── face-recognition/
    ├── run.bat                 # ⭐ Nhấp đúp để chạy app
    ├── requirements.txt        # Danh sách thư viện Python
    ├── config.json             # Cấu hình app (tự tạo, KHÔNG commit)
    ├── models/                 # Model buffalo_l (tự tải, KHÔNG commit)
    ├── data/                   # Dữ liệu runtime (KHÔNG commit)
    ├── scripts/                # Script test & seed dữ liệu demo
    └── app/
        ├── main.py             # Entry point
        ├── config.py           # Đọc/ghi config.json
        ├── core/               # Detector, embedder (nhận diện khuôn mặt)
        ├── services/           # Business logic (attendance, payroll...)
        ├── infrastructure/     # Kết nối DB (SQL Server, cloud sync)
        └── ui/                 # Giao diện PySide6
```

---

## 7. Script hữu ích (thư mục `face-recognition/scripts/`)

| Script | Chức năng |
|---|---|
| `seed_demo.py` | Tạo dữ liệu demo cho nhân viên |
| `seed_payroll_demo.py` | Tạo dữ liệu demo bảng lương |
| `migrate_sqlite_to_sqlserver.py` | Chuyển dữ liệu SQLite cũ sang SQL Server |
| `calibrate_threshold.py` | Chỉnh ngưỡng nhận diện phù hợp với webcam |
| `test_*.py` | Kiểm tra từng thành phần (auth, detector, payroll...) |

Chạy ví dụ:

```powershell
.venv\Scripts\activate
python scripts\seed_demo.py
```
