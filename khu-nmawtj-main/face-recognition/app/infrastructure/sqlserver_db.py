"""Kết nối SQL Server — cấu hình cố định của dự án.

Server:      LAPTOP-OBLS1HOE\\SQLEXPRESS
Database:    FaceRecognitionDB
Auth:        Windows Authentication (Trusted_Connection)
Driver:      ODBC Driver 18 for SQL Server
Thư viện:    pyodbc

App KHÔNG tạo/xóa bảng — các bảng (persons, face_samples,
recognition_events, sync_outbox, settings, attendance) đã được tạo sẵn
trong SSMS; db.py chỉ kiểm tra sự tồn tại khi mở kết nối.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyodbc


SERVER = r"LAPTOP-OBLS1HOE\SQLEXPRESS"
DATABASE = "FaceRecognitionDB"


def get_connection():
    connection_string = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        "Trusted_Connection=yes;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )

    return pyodbc.connect(connection_string)


def utc_now_iso() -> str:
    """Thời điểm hiện tại UTC dạng ISO 'YYYY-MM-DDTHH:MM:SS.ffffffZ'.

    Thay cho ``strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`` của SQLite (DEFAULT
    của bảng cũ). App luôn truyền giá trị này vào câu INSERT khi cột
    created_at/captured_at/detected_at/created_at(outbox) không có sẵn.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"