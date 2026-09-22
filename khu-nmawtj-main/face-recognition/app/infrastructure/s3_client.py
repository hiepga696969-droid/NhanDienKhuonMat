"""S3Client — upload/tải ảnh lên object storage S3-compatible (Bước 16).

App dùng **Supabase Storage** làm nơi lưu ảnh (lý do: free tier KHÔNG yêu
cầu thẻ tín dụng — phù hợp người dùng không có thẻ quốc tế; Cloudflare R2
bắt buộc thêm thẻ để kích hoạt dù dùng free tier).

Supabase Storage hỗ trợ **S3 protocol** (docs chính thức:
https://supabase.com/docs/guides/storage/s3/compatibility) → dùng AWS SDK
``boto3`` với endpoint trỏ tới Supabase. Đây cũng chính là lý do tách
thành lớp ``S3Client`` TỔNG QUÁT: endpoint/region/bucket/credential được
truyền vào từ config — muốn đổi provider (Supabase → R2 → MinIO...) chỉ
cần đổi thông tin trong Cài đặt, KHÔNG đổi code.

Cấu hình (tạo trong dashboard Supabase → free, không cần thẻ):
- **Project**: supabase.com → New project (free plan) → chờ tạo xong.
- **Endpoint S3**: trang Storage → Settings → \"S3 Access Keys\" hiển thị
  sẵn endpoint dạng  ``https://<project_ref>.storage.supabase.co/storage/v1/s3``
  (hoặc ``https://<project_ref>.supabase.co/storage/v1/s3``) — dán thẳng.
- **Region**: cùng trang S3 Access Keys (ví dụ ``ap-southeast-1``).
- **Access Key ID / Secret Access Key**: nút \"Create new access key\" —
  copy ngay (secret chỉ hiện 1 lần). Cần bật \"S3 protocol\" trong Settings.
- **Bucket**: Storage → New bucket (đặt tên, public/private tùy chọn).

QUAN TRỌNG — path-style addressing: endpoint Supabase có dạng
``.../storage/v1/s3`` (không phải hostname theo bucket) → boto3 phải dùng
``addressing_style='path'`` (tương đương ``forcePathStyle=true`` trong docs
JS của Supabase), nếu không URL sẽ sai dạng ``<bucket>.<endpoint>/...``.
``forcePathStyle`` cũng an toàn với mọi provider S3-compatible khác.

Quy ước key object (giống bản R2 trước — xóa được dễ dàng):
- ``thumbnails/{person_id}.jpg``  — ảnh đại diện người
- ``snapshots/{event_id}.jpg``    — ảnh chụp sự kiện nhận diện

D1 chỉ lưu key (cột ``thumbnail_r2_key`` / ``snapshot_r2_key`` — giữ tên
cũ cho tương thích), ảnh thật nằm trên Supabase Storage.

Mọi lỗi (sai token, mất mạng, bucket không tồn tại) được gói thành
``S3Error`` — SyncService bắt và báo trạng thái, không làm sập app.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Region mặc định nếu người dùng bỏ trống (Supabase hay dùng vùng này)
DEFAULT_REGION = "ap-southeast-1"


class S3Error(Exception):
    """Lỗi khi gọi object storage: sai thông tin, mất mạng, bucket/object lỗi."""


class S3Client:
    """Khách upload/tải/xóa object qua boto3 (S3 API) — provider-agnostic.

    Cách dùng:
        client = S3Client(endpoint_url, region, bucket, access_key_id, secret)
        client.upload("thumbnails/abc.jpg", image_bytes, "image/jpeg")
        data = client.download("thumbnails/abc.jpg")
        client.delete("thumbnails/abc.jpg")
        client.test_connection()   # ném S3Error nếu thông tin sai
    """

    def __init__(
        self,
        endpoint_url: str,
        region: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
    ) -> None:
        import boto3  # import trễ — boto3 nặng, chỉ cần khi thực sự sync ảnh
        from botocore.config import Config

        self._bucket = bucket
        self._s3 = boto3.client(
            service_name="s3",
            endpoint_url=endpoint_url.rstrip("/"),
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region or DEFAULT_REGION,
            config=Config(s3={"addressing_style": "path"}),  # bắt buộc cho Supabase
        )

    # ------------------------------------------------------------------
    # Thao tác object
    # ------------------------------------------------------------------
    def upload(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        """Upload một object (ảnh) lên storage. Lỗi → S3Error."""
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            logger.info("Đã upload %s lên %s", key, self._bucket)
        except Exception as exc:  # noqa: BLE001 — boto3 ném nhiều loại lỗi
            raise S3Error(f"Upload {key} thất bại: {_short_error(exc)}") from exc

    def download(self, key: str) -> bytes:
        """Tải object về dạng bytes. Lỗi (hoặc không tồn tại) → S3Error."""
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()
        except Exception as exc:  # noqa: BLE001
            raise S3Error(f"Tải {key} thất bại: {_short_error(exc)}") from exc

    def delete(self, key: str) -> None:
        """Xóa object. Lỗi → S3Error.

        Chú ý: S3 delete_object trả về thành công NGAY CẢ khi key không
        tồn tại (idempotent) — an toàn khi xóa nhiều lần / xóa ảnh mồ côi.
        """
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
            logger.info("Đã xóa %s", key)
        except Exception as exc:  # noqa: BLE001
            raise S3Error(f"Xóa {key} thất bại: {_short_error(exc)}") from exc

    # ------------------------------------------------------------------
    # Kiểm tra kết nối
    # ------------------------------------------------------------------
    def test_connection(self) -> None:
        """Kiểm tra thông tin kết nối: bucket tồn tại + token đúng quyền.

        Dùng ``head_bucket`` (nhẹ, không cần list object). Sai token /
        bucket không tồn tại → S3Error.
        """
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except Exception as exc:  # noqa: BLE001
            raise S3Error(f"Storage từ chối kết nối: {_short_error(exc)}") from exc


def _short_error(exc: Exception) -> str:
    """Rút gọn thông điệp lỗi boto3 (thường rất dài)."""
    message = str(exc).strip()
    # boto3 ClientError: "An error occurred (403) when calling ... : message"
    if message.startswith("An error occurred"):
        try:
            return message.split(": ", 1)[1].split("(")[0].strip() or message
        except (IndexError, ValueError):
            return message
    return message or exc.__class__.__name__
