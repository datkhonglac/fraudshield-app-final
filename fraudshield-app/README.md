# Fraud Shield v3.1 – Web app Streamlit

Bàn làm việc chống gian lận thanh toán số · Festival Chuyển đổi số 2026 · Bảng B · bản vòng Chung kết.
Mọi đặc trưng chỉ dùng thông tin có trước hoặc tại thời điểm giao dịch; Isolation Forest và bảng phân vị
tham chiếu chỉ học trên giai đoạn quá khứ (tháng 1–4).

Kết quả trên Data Pack BTC (kiểm định 5-fold, ngưỡng lồng – số ngoài mẫu):

| Chỉ số | Kết quả |
|---|---|
| Recall can thiệp (chặn hoặc xác thực thêm) | 91,4% (117/128) · 1,44% khách thật phải xác thực thêm |
| Giá trị tiền gian lận bị chặn | 93,6% (98,1% nếu tính cả xác thực thêm) |
| Recall theo vụ tấn công | 85,7% (96/112 vụ) |
| Recall chặn ngay (HIGH/CRITICAL) | 82,8% (106/128) · 10 báo nhầm / 1.872 · F1 0,869 · PR-AUC 0,934 |
| Kiểm tra theo thời gian (học T1–4, thử T5–6) | F1 0,854 · can thiệp 86,4% |
| Kiểu gian lận chưa từng học (ẩn từng kiểu) | chặn 76/128 · can thiệp 96/128 (75%) |
| Chấm điểm thời gian thực | trùng 100% với chấm theo lô trên 8.000 giao dịch · ~0,4 ms/giao dịch |

## Ứng dụng có gì (6 màn hình)

- **Hàng đợi của tôi** (trang mở đầu): 4 chip lọc nhanh (Khẩn cấp chưa xử lý · Thiết bị dùng chung R7 · Xác thực thêm
  tại merchant HIGH · Nghi vấn mẫu mới), danh sách hồ sơ có cột SLA (CRITICAL 5 phút, HIGH 2 giờ), xử lý hàng loạt
  các hồ sơ cùng loại rủi ro (có hộp xác nhận), ô quyết định, nút tác vụ **mô phỏng** (tạm khóa thiết bị, yêu cầu
  sinh trắc học, gửi thông báo), sơ đồ liên kết đa khách hàng, nhật ký quyết định và tác vụ.
- **Luồng thời gian thực**: mô hình học trên tháng 1–4, phát lần lượt giao dịch tháng 5–6 (1–50 giao dịch/giây),
  chấm từng giao dịch bằng bảng trạng thái trong bộ nhớ, hàng đợi trực tiếp có đồng hồ SLA, đối chiếu đáp án BTC,
  nút **Kiểm tra trùng khớp lô – luồng** trên toàn bộ 8.000 giao dịch.
- **Tổng quan**: số giao dịch tạm dừng / xác thực thêm, bộ 4 chỉ số Recall, phân bố mức rủi ro, xu hướng, tiến độ.
- **Giao dịch**: lọc, tìm kiếm, **bộ lọc nâng cao** (theo từng luật, khoảng tiền, bội số tiền, khung giờ, IP, số chiều
  lệch thói quen), chi tiết điểm từng lớp và hành động tự động.
- **Kiểm định**: tổng quan, 3 tầng hành động, kiểu gian lận mới (ẩn từng kiểu), thời gian thực, hiệu quả từng luật,
  phân tích lỗi, so sánh cấu hình, độ nhạy ngưỡng, công bằng.
- **Dữ liệu & Quản trị**: tải Data Pack, kiểm tra dữ liệu, chạy mô hình, tải kết quả, **học lại có kiểm soát** (so sánh
  trước – sau trên giai đoạn kiểm tra độc lập, áp dụng, khôi phục bản cũ), xoá kết luận đã ghi.

## Cấu trúc

```
streamlit_app.py        giao diện (tệp chính khi deploy)
fraud_core.py           lõi mô hình: hồ sơ hành vi + Rule + Logistic Regression + Isolation Forest, 3 tầng hành động,
                        thí nghiệm ẩn từng kiểu, học lại có kiểm soát
fs_realtime.py          chấm điểm thời gian thực: bảng trạng thái trong bộ nhớ, Isolation Forest dạng mảng numpy,
                        kiểm tra trùng khớp lô – luồng
model/pretrained.json   mô hình học sẵn, dùng khi Data Pack tải lên không có nhãn
data/                   Data Pack BTC (9 tệp CSV) cho nút "Chạy với Data Pack BTC"
.streamlit/config.toml  màu sắc, font, giới hạn tải lên 100 MB  ← BẮT BUỘC phải có trên GitHub
requirements.txt        thư viện cần cài
```

## Đưa lên mạng (Streamlit Community Cloud, miễn phí)

**Bước 1 – Tạo repo GitHub**
1. Vào github.com → **New repository** → tên `fraudshield-app` → chọn **Private** → **Create repository**.
2. Bấm **uploading an existing file**, kéo **toàn bộ nội dung** thư mục này vào (kể cả thư mục `.streamlit`).
   - macOS: Finder ẩn thư mục bắt đầu bằng dấu chấm – bấm `Cmd + Shift + .` để hiện rồi kéo cả `.streamlit`.
3. Bấm **Commit changes**. Kiểm tra trên GitHub thấy đủ `.streamlit/config.toml`, `data/`, `model/`.

**Bước 2 – Deploy**
1. Vào https://share.streamlit.io → **Continue with GitHub**, cho phép Streamlit truy cập repo (kể cả repo private).
2. **Create app** → **Yup, I have an app**.
3. Repository: `<tài-khoản>/fraudshield-app` · Branch: `main` · Main file path: `streamlit_app.py`.
4. App URL: đặt tên dễ nhớ, ví dụ `fraudshield-tendoi` → link sẽ là `https://fraudshield-tendoi.streamlit.app`.
5. **Advanced settings** → Python **3.12** → Save → **Deploy**. Lần đầu mất khoảng 2–5 phút.

**Bước 3 – Mở link cho mọi người**
Repo private thì app mặc định cũng private. Trong app bấm **Share** (góc phải trên) → **Make this app public**
(hoặc Settings → Sharing → *This app is public and searchable*). Mở link bằng tab ẩn danh để kiểm tra.

## Lưu ý

- App không có người dùng 12 giờ sẽ "ngủ"; người mở bấm **Yes, get this app back up!** và chờ khoảng 1 phút.
  Nên tự mở link trước khi gửi giáo viên / giám khảo.
- Sửa code trên GitHub → app tự cập nhật sau ít phút.
- Điền tên đội: sửa dòng `TEAM_NAME = ""` đầu tệp `streamlit_app.py`.
- Nhật ký quyết định chỉ lưu trong phiên đang mở – tải `audit_log.csv` ở tab **Tải kết quả** trước khi đóng.
- Nếu BTC không cho phép công khai Data Pack, xoá thư mục `data/`: app vẫn chạy, người dùng tự tải tệp .zip lên.

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
