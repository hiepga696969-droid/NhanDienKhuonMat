import pyodbc

SERVER = r"LAPTOP-OBLS1HOE\SQLEXPRESS"
DATABASE = "FaceRecognitionDB"

connection_string = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

try:
    conn = pyodbc.connect(connection_string)

    print("✅ KẾT NỐI SQL SERVER THÀNH CÔNG")
    print("Database:", DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
    """)

    print("\nDanh sách bảng:")

    for row in cursor.fetchall():
        print("-", row[0])

    conn.close()

except Exception as e:
    print("❌ KẾT NỐI THẤT BẠI")
    print(e)