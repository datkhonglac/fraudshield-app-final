"""
FRAUD SHIELD – Digital Payment Risk Console (bản Streamlit)
Festival Chuyển đổi số 2026 · Bảng B · Digital Payment Fraud Shield

Chạy trên máy:   streamlit run streamlit_app.py
Triển khai:      Streamlit Community Cloud – chọn tệp chính là streamlit_app.py
Lõi mô hình:     fraud_core.py (giống hệt pipeline trong hồ sơ dự thi)

Phiên bản v3.1 – 6 màn hình: Hàng đợi của tôi · Luồng thời gian thực · Tổng quan · Giao dịch · Kiểm định ·
Dữ liệu & Quản trị. Chấm điểm thời gian thực: fs_realtime.py (trùng 100% với chấm theo lô).
"""
import hashlib
import html
import io
import json
import re
import sqlite3
import tempfile
import threading
import time
import zipfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

import fraud_core as F
import fs_realtime as RT

# ----------------------------------------------------------------------------- cấu hình chung
TEAM_NAME = ""  # điền tên đội, ví dụ "Đội ABC" – để trống thì không hiển thị
BASE = Path(__file__).parent
SAMPLE_DIR = BASE / "data"
VN_TZ = timezone(timedelta(hours=7))
LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
LEVEL_COLOR = {"CRITICAL": "#E03131", "HIGH": "#F76707", "MEDIUM": "#F5B700", "LOW": "#3B7DDD"}
NAVY, BLUE, BLUE_SOFT, INK, MUTED, GRID = "#0B2552", "#1E56C8", "#BFD0F7", "#0F172A", "#5B6577", "#EDF1F7"
TEAL, GREEN, PURPLE = "#12A4B8", "#2F9E44", "#7048E8"
DECISIONS = ["Xác nhận gian lận", "Báo động giả", "Cần thêm thông tin"]
STATUSES = ["Chưa xử lý", "Đang xác minh", "Xác nhận gian lận", "Báo động giả"]
STATUS_COLOR = {"Chưa xử lý": "#F03E3E", "Đang xác minh": "#1C7ED6", "Xác nhận gian lận": "#862E9C",
                "Báo động giả": "#2F9E44", "Theo dõi": "#F59F00", "Bình thường": "#12B886",
                "Xử lý một phần": "#1C7ED6", "Đã xử lý": "#2F9E44", "Xác thực thêm": "#F59F00",
                "Nghi vấn mẫu mới": "#7048E8"}
DONE = ("Xác nhận gian lận", "Báo động giả")
LEVEL_INFO = {"CRITICAL": ("Khẩn cấp", "Tạm dừng giao dịch, gọi xác minh ngay · SLA 5 phút"),
              "HIGH": ("Cao", "Tạm dừng, chuyên viên kiểm tra · SLA 2 giờ"),
              "MEDIUM": ("Xác thực thêm", "Không chặn, tự động yêu cầu OTP/sinh trắc học"),
              "LOW": ("Bình thường", "Cho qua")}
ACTION_SHORT = {"CRITICAL": "Tạm dừng", "HIGH": "Tạm dừng", "MEDIUM": "Xác thực thêm", "LOW": "Cho qua"}
SLA_S = {"CRITICAL": 300, "HIGH": 7200}
SLA_TXT = {"CRITICAL": "5 phút", "HIGH": "2 giờ"}
SIM_ACTIONS = [("lock", "Tạm khóa thiết bị", ":material/phonelink_lock:"),
               ("bio", "Yêu cầu xác thực sinh trắc học", ":material/fingerprint:"),
               ("sms", "Gửi thông báo cho khách", ":material/sms:")]
DEC2STATUS = {"Xác nhận gian lận": "Xác nhận gian lận", "Xác nhận rủi ro": "Xác nhận gian lận",
              "Báo động giả": "Báo động giả", "Cần thêm thông tin": "Đang xác minh"}
DB_PATH = Path(tempfile.gettempdir()) / "fraudshield_decisions.sqlite"
LOG_COLS = ["transaction_id", "case_id", "decision", "note", "user", "time", "model", "threshold"]
# Loại rủi ro chính của một giao dịch: luật đầu tiên bật theo thứ tự này
RISK_TYPES = [("flag_R5", "Thử sai mật khẩu/OTP"), ("flag_R6", "Giao dịch dồn dập"), ("flag_R2", "Thiết bị lạ"),
              ("flag_R7", "Thiết bị dùng chung"), ("flag_R3", "IP nước ngoài, nơi lạ"),
              ("flag_R1_strong", "Số tiền bất thường"), ("flag_R1_mild", "Số tiền bất thường"),
              ("flag_R4", "Merchant rủi ro cao")]
OTHER_TYPE = "Lệch thói quen"
TYPE_ORDER = list(dict.fromkeys([t for _, t in RISK_TYPES] + [OTHER_TYPE]))
FRAUD_VN = {"FAILED_THEN_SUCCESS": "Thử sai rồi thành công", "VELOCITY_BURST": "Giao dịch dồn dập",
            "AMOUNT_SPIKE": "Số tiền tăng vọt", "NEW_DEVICE": "Thiết bị mới",
            "LOCATION_ANOMALY": "Nơi lạ, IP nước ngoài", "HIGH_RISK_MERCHANT": "Merchant rủi ro cao"}
NAV = [("cases", "Hàng đợi của tôi", ":material/inbox:", "Hàng đợi hồ sơ cần xử lý"),
       ("live", "Luồng thời gian thực", ":material/bolt:", "Giám sát giao dịch theo thời gian thực"),
       ("overview", "Tổng quan", ":material/space_dashboard:", "Tổng quan kết quả phát hiện rủi ro"),
       ("tx", "Giao dịch", ":material/payments:", "Tra cứu giao dịch"),
       ("eval", "Kiểm định", ":material/verified_user:", "Kết quả kiểm định mô hình"),
       ("data", "Dữ liệu & Quản trị", ":material/database:", "Dữ liệu, chạy mô hình và học lại có kiểm soát")]

st.set_page_config(page_title="Fraud Shield · Digital Payment Risk Console", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

SHIELD = ('<svg viewBox="0 0 48 48" width="{w}" height="{w}" aria-hidden="true"><defs><linearGradient id="g{w}" x1="0" y1="0" '
          'x2="1" y2="1"><stop offset="0" stop-color="#5B9BFF"/><stop offset="1" stop-color="#1E56C8"/></linearGradient></defs>'
          '<path d="M24 3 6 10v12c0 11.5 7.7 20.6 18 23 10.3-2.4 18-11.5 18-23V10L24 3z" fill="url(#g{w})" '
          'stroke="#BFD6FF" stroke-width="1.5"/><path d="m15.5 24.5 6 6 11-12" fill="none" stroke="#fff" stroke-width="4" '
          'stroke-linecap="round" stroke-linejoin="round"/></svg>')
USER_SVG = ('<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><circle cx="12" cy="8" r="4" fill="#1E56C8"/>'
            '<path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7" fill="#1E56C8"/></svg>')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root{--navy:#0B2552;--blue:#1E56C8;--blue-soft:#E8EEFC;--ink:#0F172A;--muted:#5B6577;--line:#E3E8F0;--bg:#F3F6FB;
--card:#FFFFFF;--crit:#E03131;--high:#F76707;--med:#F5B700;--low:#3B7DDD;--ok:#2F9E44}
html,body,.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp h1,.stApp h2,
.stApp h3,.stApp h4,.stMarkdown,.fs,.fs div,.fs span,.fs td,.fs th{font-family:'Be Vietnam Pro',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp{background:var(--bg)}
header[data-testid="stHeader"]{background:transparent}
footer{visibility:hidden}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1480px;padding-top:1.1rem;padding-bottom:2.5rem}
/* ---------- thanh điều hướng bên trái ---------- */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0B2552 0%,#0A2048 60%,#081A3B 100%);border-right:none}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{padding-bottom:0}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] span{color:#C9D6EA!important;fill:#C9D6EA!important}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:6px}
section[data-testid="stSidebar"] .stButton button{background:transparent!important;border:none!important;color:#C9D6EA!important;
justify-content:flex-start!important;padding:10px 14px!important;border-radius:10px!important;min-height:46px;box-shadow:none!important}
section[data-testid="stSidebar"] .stButton button>div{justify-content:flex-start!important}
section[data-testid="stSidebar"] .stButton button p{font-size:14.5px;font-weight:500;text-align:left;color:inherit!important}
section[data-testid="stSidebar"] .stButton button span{color:inherit!important}
section[data-testid="stSidebar"] .stButton button:hover{background:rgba(255,255,255,.08)!important;color:#fff!important}
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"]{background:#1E56C8!important;color:#fff!important;
box-shadow:inset 4px 0 0 #8DB8FF!important}
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] p{font-weight:700}
.fs-brand{display:flex;align-items:center;gap:12px;padding:0 4px 18px;margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,.1)}
.fs-brand .nm{font-size:21px;font-weight:800;color:#fff;letter-spacing:-.01em;line-height:1.1}
.fs-brand .sb{font-size:11.5px;color:#9FB3D1;margin-top:3px}
.fs-sidefoot{margin-top:26px;padding:14px 6px 0;border-top:1px solid rgba(255,255,255,.1);font-size:11.5px;line-height:1.55;color:#8EA3C4}
.fs-sidefoot b{color:#DCE6F5;font-weight:600}
/* ---------- thanh trên cùng ---------- */
.fs-top{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;background:#fff;border:1px solid var(--line);
border-radius:14px;padding:12px 18px;margin-bottom:16px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.fs-top .ttl{font-size:22px;font-weight:800;color:var(--navy);letter-spacing:-.01em;line-height:1.2}
.fs-top .src{font-size:12.5px;color:var(--muted);margin-top:2px}
.fs-top .right{display:flex;align-items:center;gap:14px}
.fs-run{font-size:11.5px;color:var(--muted);text-align:right;line-height:1.4}.fs-run b{display:block;color:#1F2937;font-weight:600;font-size:12.5px}
.fs-ok{display:inline-flex;gap:6px;align-items:center;padding:6px 12px;border-radius:999px;background:#E9F7EE;color:#2B8A3E;
font-weight:600;font-size:12.5px;border:1px solid #C3E6CF;white-space:nowrap}
.fs-ok i{width:16px;height:16px;border-radius:50%;background:#2F9E44;color:#fff;font-style:normal;font-size:10px;display:inline-flex;
align-items:center;justify-content:center}
.fs-av{width:38px;height:38px;border-radius:50%;background:#E8EEFC;display:flex;align-items:center;justify-content:center;border:1px solid #D3DEF7}
/* ---------- thẻ ---------- */
[class*="st-key-card"]{box-sizing:border-box;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px 12px;
box-shadow:0 1px 2px rgba(15,23,42,.04)}
.fs-h{font-size:15.5px;font-weight:700;color:var(--navy);margin:0 0 2px}
.fs-sub{font-size:12.5px;color:var(--muted);margin-bottom:10px;line-height:1.5}
.fs-sec{font-size:19px;font-weight:800;color:var(--navy);margin:6px 0 2px}
/* ---------- KPI ---------- */
.fs-kgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:0 0 16px}
.fs-k{position:relative;background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 18px 14px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.fs-k .v{font-size:32px;font-weight:800;color:var(--c);letter-spacing:-.02em;line-height:1.1;font-variant-numeric:tabular-nums}
.fs-k .l{font-size:13px;color:#475467;display:flex;align-items:center;gap:7px;margin-top:6px;font-weight:500}
.fs-k .l i{width:8px;height:8px;border-radius:50%;background:var(--c);flex:none}
.fs-k .s{font-size:12px;color:var(--muted);margin-top:6px;line-height:1.45}
.fs-k .tag{position:absolute;right:14px;top:14px;background:var(--cbg);color:var(--c);font-size:11.5px;font-weight:700;
padding:2px 9px;border-radius:999px}
.fs-k::before{content:"";position:absolute;left:0;top:14px;bottom:14px;width:4px;border-radius:0 4px 4px 0;background:var(--c)}
/* ---------- huy hiệu ---------- */
.fs-badge{display:inline-block;padding:3px 10px;border-radius:7px;font-size:11.5px;font-weight:700;letter-spacing:.02em;white-space:nowrap}
.b-CRITICAL{background:#FFE3E3;color:#C92A2A}.b-HIGH{background:#FFE8CC;color:#D9480F}
.b-MEDIUM{background:#FFF3BF;color:#A06A00}.b-LOW{background:#E3F0FF;color:#1C64C4}
.b-ĐẠT{background:#E6F4EA;color:#2B8A3E}.b-CẢNH{background:#FFF3BF;color:#A06A00}
.b-LỖI{background:#FFE3E3;color:#C92A2A}.b-THIẾU{background:#EEF1F5;color:#5B6577}
.fs-st{display:inline-flex;align-items:center;gap:7px;white-space:nowrap;font-size:12.5px;color:#1F2937}
.fs-st i{width:8px;height:8px;border-radius:50%;flex:none}
/* ---------- bảng HTML ---------- */
.fs-tw{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
.fs-table{width:100%;border-collapse:collapse;font-size:13px;background:#fff}
.fs-table th{background:#F5F7FB;color:#475467;font-weight:600;text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
.fs-table td{padding:10px 12px;border-bottom:1px solid #EEF1F5;color:#1F2937;vertical-align:middle}
.fs-table tr:last-child td{border-bottom:none}
.fs-table tbody tr:hover td{background:#F8FAFE}
.fs-table .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.fs-table .mono{font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:600;color:#0B2552}
.fs-table tr.hl td{font-weight:700;background:#F3F7FF}
.fs-table tr.bad td{background:#FFF5F5}
.fs-table .best{color:var(--blue);font-weight:700}
.fs-table .new{color:var(--high);font-weight:700}
.fs-table .yes{color:#2F9E44;font-weight:800}.fs-table .no{color:#E03131;font-weight:800}
/* ---------- chi tiết ---------- */
.fs-case{display:flex;flex-wrap:wrap;gap:22px;align-items:flex-start}
.fs-case .item .k{font-size:12px;color:var(--muted);font-weight:600}
.fs-case .item .v{font-size:16px;font-weight:700;color:var(--ink);margin-top:3px;font-variant-numeric:tabular-nums}
.fs-reason{background:#F6F8FC;border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:10px;
padding:11px 14px;font-size:13.5px;line-height:1.55;color:#1F2937;margin-top:14px}
.fs-flag{display:inline-block;padding:5px 11px;border-radius:9px;font-size:12.5px;margin:0 6px 6px 0;border:1px solid var(--line);background:#fff}
.fs-flag.on{background:#FFF4ED;border-color:#FCC9A8;color:#9A3412;font-weight:600}
.fs-flag.off{color:#98A2B3}
.fs-kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:0 22px}
.fs-kv div{display:flex;justify-content:space-between;gap:12px;border-bottom:1px dashed #E6EAF1;padding:7px 0;font-size:13px}
.fs-kv span{color:var(--muted)}
.fs-kv b{font-weight:600;color:#111827;text-align:right}
.fs-dl{display:grid;grid-template-columns:auto 1fr;gap:9px 16px;font-size:13.5px;margin:4px 0 6px}
.fs-dl span{color:var(--muted)}.fs-dl b{font-weight:600;color:#111827}
.fs-phead{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.fs-cause{background:#FFF5F5;border:1px solid #FFD6D6;border-radius:12px;padding:12px 14px;margin:12px 0 4px}
.fs-cause .t{color:#C92A2A;font-weight:700;font-size:14px;margin-bottom:6px}
.fs-cause ul{margin:0;padding-left:18px}.fs-cause li{font-size:13px;line-height:1.55;color:#1F2937;margin:2px 0}
.fs-warnbox{display:flex;gap:12px;background:#FFF5F5;border:1px solid #FFD6D6;border-radius:12px;padding:12px 14px;margin-top:10px}
.fs-warnbox .ic{width:26px;height:26px;border-radius:50%;background:#E03131;color:#fff;font-weight:800;display:flex;align-items:center;
justify-content:center;flex:none;font-size:14px}
.fs-warnbox .t{color:#C92A2A;font-weight:700;font-size:14px}
.fs-warnbox .d{font-size:13px;color:#1F2937;line-height:1.55;margin-top:2px}
/* ---------- tiến độ, chú thích ---------- */
.fs-prog{display:flex;height:14px;border-radius:999px;overflow:hidden;background:#EEF1F5;margin:6px 0 10px}
.fs-prog i{display:block;height:100%}
.fs-legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:#334155}
.fs-legend span{display:inline-flex;align-items:center;gap:6px}
.fs-legend b{font-weight:700;font-variant-numeric:tabular-nums}
.fs-dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.fs-mini{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 0;border-bottom:1px solid #EEF1F5;font-size:13px}
.fs-mini:last-child{border-bottom:none}
.fs-mini .id{font-weight:700;color:var(--navy);min-width:82px}
.fs-mini .why{flex:1;color:#475467;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* ---------- bước dữ liệu ---------- */
.fs-steps{display:flex;align-items:flex-start;justify-content:space-between;background:#fff;border:1px solid var(--line);border-radius:14px;
padding:18px 26px 14px;margin-bottom:16px}
.fs-stp{display:flex;flex-direction:column;align-items:center;gap:8px;flex:none;width:130px;text-align:center;font-size:13px;color:#475467}
.fs-stp .c{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;
border:2px solid #CBD5E1;color:#94A3B8;background:#fff}
.fs-stp.done .c{background:#2F9E44;border-color:#2F9E44;color:#fff}
.fs-stp.cur .c{background:#1E56C8;border-color:#1E56C8;color:#fff;box-shadow:0 0 0 5px rgba(30,86,200,.15)}
.fs-stp.err .c{background:#E03131;border-color:#E03131;color:#fff}
.fs-stp.cur,.fs-stp.done{color:#0B2552;font-weight:600}
.fs-sline{flex:1;height:2px;background:#E2E8F0;margin-top:17px}.fs-sline.done{background:#2F9E44}
.fs-req{display:flex;align-items:flex-start;gap:10px;font-size:13.5px;padding:6px 0;color:#1F2937}
.fs-req i{width:20px;height:20px;border-radius:50%;flex:none;display:inline-flex;align-items:center;justify-content:center;font-style:normal;
font-size:11px;font-weight:800;color:#fff;margin-top:1px}
.fs-req small{display:block;color:var(--muted);font-size:12px}
[data-testid="stFileUploaderDropzone"]{border:2px dashed #9DB7F0!important;background:#F5F8FF!important;border-radius:12px!important;min-height:120px}
/* ---------- nút, tab ---------- */
.stButton button,.stDownloadButton button,.stFormSubmitButton button{border-radius:10px;font-weight:600;min-height:42px}
button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primaryFormSubmit"]{background:#1E56C8!important;border-color:#1E56C8!important;color:#fff!important}
button[data-testid="stBaseButton-pillsActive"]{background:#E8EEFC!important;border-color:#1E56C8!important;color:#1E3A8A!important}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:none;flex-wrap:wrap;margin-bottom:6px}
.stTabs [data-baseweb="tab"]{padding:7px 16px;border-radius:9px;background:#fff;border:1px solid var(--line);height:auto}
.stTabs [data-baseweb="tab"] p{font-size:13.5px;font-weight:600;color:#475467}
.stTabs [aria-selected="true"]{background:#1E56C8!important;border-color:#1E56C8!important}
.stTabs [aria-selected="true"] p{color:#fff!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
[data-baseweb="tag"]{background:#E8EEFC!important}
[data-baseweb="tag"] span,[data-baseweb="tag"] svg{color:#1E3A8A!important;fill:#1E3A8A!important}
.fs-foot{margin-top:24px;padding-top:12px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);
display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
@media (max-width:1100px){.fs-kgrid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.fs-kgrid{grid-template-columns:1fr 1fr;gap:10px}.fs-k .v{font-size:25px}
.fs-top .ttl{font-size:18px}.fs-steps{padding:14px 10px}.fs-stp{width:70px;font-size:11.5px}}
</style>
"""
st.markdown(re.sub(r"\s*\n\s*", " ", CSS), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- tiện ích hiển thị
def H(s):
    """Hiển thị HTML (gộp dòng để Markdown không biến phần thụt lề thành khối code)."""
    st.markdown(re.sub(r"\s*\n\s*", " ", s), unsafe_allow_html=True)


def e(x):
    return html.escape("" if x is None else str(x))


def n0(x):
    return f"{int(round(float(x))):,}".replace(",", ".")


def d(x, k=3):
    # làm tròn nửa lên (0,9145 → 0,915) cho khớp số liệu trong hồ sơ
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP)
    return f"{q:.{k}f}".replace(".", ",")


def pct(x, k=1):
    return f"{float(x) * 100:.{k}f}%".replace(".", ",")


def money(x):
    return n0(x) + " ₫"


def dt(x):
    return pd.Timestamp(x).strftime("%d/%m/%Y %H:%M")


def badge(level):
    lv = str(level)
    cls = lv.split()[0] if lv else "LOW"
    return f'<span class="fs-badge b-{e(cls)}">{e(lv)}</span>'


def kpi(label, value, sub="", kind=""):
    return (f'<div class="fs-kpi {kind}"><div class="lab">{label}</div><div class="val">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def kpis(cards, small=False):
    H(f'<div class="fs fs-kpis{" sm" if small else ""}">' + "".join(cards) + "</div>")


def title(text, sub=None):
    H(f'<div class="fs-h">{text}</div>' + (f'<div class="fs-sub">{sub}</div>' if sub else ""))


def table(rows, cols, num_cols=(), hl_last=False, best=None, raw_cols=(), hl_row=None):
    """rows: list[dict]; cols: list[(key, header)]."""
    head = "".join(f'<th class="{"num" if k in num_cols else ""}">{e(h)}</th>' for k, h in cols)
    body = []
    for i, r in enumerate(rows):
        cls = ' class="hl"' if (hl_last and i == len(rows) - 1) or hl_row == i else ""
        tds = []
        for k, _ in cols:
            v = r.get(k, "")
            txt = v if k in raw_cols else e(v)
            c = ["num"] if k in num_cols else []
            if best and best.get(k) == i:
                c.append("best")
            tds.append(f'<td class="{" ".join(c)}">{txt}</td>')
        body.append(f"<tr{cls}>" + "".join(tds) + "</tr>")
    H(f'<div class="fs fs-tw"><table class="fs-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def card(key):
    return st.container(border=False, key=f"card_{key}")


def style(fig, h=320, legend=True, **kw):
    kw.setdefault("bargap", 0.28)
    fig.update_layout(
        height=h, margin=dict(l=6, r=10, t=30 if legend else 8, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", separators=",.",
        font=dict(family="Be Vietnam Pro, system-ui, sans-serif", size=12, color="#334155"),
        hoverlabel=dict(bgcolor=NAVY, bordercolor=NAVY, font=dict(color="#fff", family="Be Vietnam Pro, sans-serif")),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11.5)),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#D5DCE6", tickfont=dict(size=11.5), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#D5DCE6", tickfont=dict(size=11.5), automargin=True)
    return fig


def chart(fig, key=None):
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False, "responsive": True})


def grid(df, **kw):
    return st.dataframe(df, hide_index=True, width="stretch", **kw)


def money_short(x):
    x = float(x)
    if x >= 1e9:
        return f"{x / 1e9:.2f} tỷ ₫".replace(".", ",")
    if x >= 1e6:
        return f"{x / 1e6:.1f} triệu ₫".replace(".", ",")
    return money(x)


# ----------------------------------------------------------------------------- lưu quyết định xử lý (SQLite)
def _db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("CREATE TABLE IF NOT EXISTS decisions (ds TEXT, transaction_id TEXT, case_id TEXT, decision TEXT, "
                "note TEXT, user TEXT, time TEXT, model TEXT, threshold REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS actions (ds TEXT, case_id TEXT, target TEXT, action TEXT, user TEXT, time TEXT)")
    return con


def save_action(ds, case_id, target, action, user="chuyên viên"):
    """Tác vụ nghiệp vụ MÔ PHỎNG (tạm khóa thiết bị, yêu cầu sinh trắc học, gửi thông báo) – chỉ ghi nhật ký."""
    with closing(_db()) as con:
        con.execute("INSERT INTO actions VALUES (?,?,?,?,?,?)", (ds, case_id, target, action, user, now_vn()))
        con.commit()


def load_actions(ds):
    try:
        with closing(_db()) as con:
            return pd.read_sql_query("SELECT case_id, target, action, user, time FROM actions WHERE ds = ? ORDER BY rowid",
                                     con, params=(ds,))
    except Exception:  # noqa
        return pd.DataFrame(columns=["case_id", "target", "action", "user", "time"])


def analyst_labels(ds):
    """Nhãn do chuyên viên kết luận (kết luận mới nhất của từng giao dịch): gian lận = 1, báo động giả = 0."""
    lg = load_log(ds)
    if lg.empty:
        return pd.DataFrame(columns=["transaction_id", "is_fraud"])
    last = lg.drop_duplicates("transaction_id", keep="last")
    last = last[last.decision.isin(["Xác nhận gian lận", "Báo động giả"])]
    return pd.DataFrame({"transaction_id": last.transaction_id.values,
                         "is_fraud": (last.decision == "Xác nhận gian lận").astype(int).values})


def save_decisions(ds, rows):
    with closing(_db()) as con:
        con.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
                        [(ds, r["transaction_id"], r.get("case_id") or "", r["decision"], r.get("note") or "",
                          r.get("user") or "chuyên viên", r["time"], F.VERSION, float(r["threshold"])) for r in rows])
        con.commit()


def load_log(ds):
    try:
        with closing(_db()) as con:
            return pd.read_sql_query("SELECT transaction_id, case_id, decision, note, user, time, model, threshold "
                                     "FROM decisions WHERE ds = ? ORDER BY rowid", con, params=(ds,))
    except Exception:  # noqa
        return pd.DataFrame(columns=LOG_COLS)


def clear_decisions(ds):
    with closing(_db()) as con:
        con.execute("DELETE FROM decisions WHERE ds = ?", (ds,))
        con.execute("DELETE FROM actions WHERE ds = ?", (ds,))
        con.commit()


def now_vn():
    return datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M:%S")


def status_frame(res):
    """Mọi giao dịch đáng ngờ (HIGH + CRITICAL) kèm trạng thái xử lý.
    Đã xử lý = có kết luận: nhãn đã xác minh trong Data Pack hoặc quyết định chuyên viên ghi trong ứng dụng."""
    df = res["df"]
    a = df[df.predicted_fraud == 1].copy()
    a["status"] = "Chưa xử lý"
    a["source"] = ""
    a["handler"] = ""
    a["handled_at"] = ""
    a["note"] = ""
    lv = res.get("lab_view")
    if lv is not None:
        m = a.transaction_id.map(dict(zip(lv.transaction_id, lv.y)))
        a.loc[m == 1, "status"] = "Xác nhận gian lận"
        a.loc[m == 0, "status"] = "Báo động giả"
        a.loc[m.notna(), "source"] = "Nhãn BTC (đã xác minh)"
    log = load_log(res["ds"])
    if len(log):
        last = log.drop_duplicates("transaction_id", keep="last").set_index("transaction_id")
        hit = a.transaction_id.isin(last.index)
        ids = a.loc[hit, "transaction_id"]
        a.loc[hit, "status"] = ids.map(last["decision"]).map(DEC2STATUS).fillna("Đang xác minh").values
        a.loc[hit, "source"] = "Chuyên viên"
        a.loc[hit, "handler"] = ids.map(last["user"]).values
        a.loc[hit, "handled_at"] = ids.map(last["time"]).values
        a.loc[hit, "note"] = ids.map(last["note"]).values
    return a


# ----------------------------------------------------------------------------- chạy mô hình (có cache)
@st.cache_data(show_spinner=False, max_entries=8, ttl=6 * 3600)
def engine(files: dict, extra=None):
    t0 = time.perf_counter()
    tables, checks, ok = F.validate(files)
    if not ok:
        return {"ok": False, "checks": checks}
    res = F.run(files, tables=tables, extra_labels=extra)
    res.update(ok=True, checks=checks, seconds=round(time.perf_counter() - t0, 1))
    return res


@st.cache_data(show_spinner=False)
def sample_files():
    return F.collect_dir(SAMPLE_DIR)


def store_result(res, files, notes, source, label=None):
    checks = [{"group": "Tệp tải lên", "check": n, "status": "CẢNH BÁO", "detail": ""} for n in notes] + res["checks"]
    ds = hashlib.sha1(files.get("transactions.csv", b"")).hexdigest()[:16]
    st.session_state.files = files
    st.session_state.res = {**res, "checks": checks, "source": source, "ds": ds,
                            "version_label": label or "Bản gốc (nhãn BTC)",
                            "started": datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M"),
                            "size": sum(len(v) for v in files.values() if isinstance(v, (bytes, bytearray))),
                            "rid": f"{abs(hash((source, time.time()))) % 10**8:08d}"}
    st.session_state.pop("failed", None)


def start_run(files, notes, source):
    with st.status("Đang chạy Fraud Shield…", expanded=True) as s:
        st.write("Kiểm tra cấu trúc tệp, đối chiếu manifest và SHA-256")
        st.write("Dựng hồ sơ hành vi, chấm điểm Rule – Logistic Regression – Isolation Forest")
        st.write("Huấn luyện/đánh giá 5-fold (nếu có nhãn), 3 tầng hành động, thí nghiệm ẩn từng kiểu, gom hồ sơ")
        try:
            res = engine(files)
        except Exception as ex:  # noqa
            s.update(label="Lỗi khi chạy mô hình", state="error")
            st.error(f"Lỗi khi chạy mô hình: {ex}")
            return
        if not res["ok"]:
            s.update(label="Dữ liệu chưa đạt các kiểm tra bắt buộc", state="error")
            st.session_state.failed = {"checks": [*({"group": "Tệp tải lên", "check": n, "status": "CẢNH BÁO",
                                                      "detail": ""} for n in notes), *res["checks"]]}
            st.rerun()
        s.update(label="Hoàn tất", state="complete")
    store_result(res, files, notes, source)
    st.rerun()


def auto_load_sample():
    """Mở link là có Dashboard ngay: tự chạy Data Pack BTC có sẵn (kết quả được cache cho mọi người xem)."""
    if not (SAMPLE_DIR / "transactions.csv").exists():
        return False
    with st.spinner("Đang chấm điểm Data Pack BTC… (lần đầu mất vài giây)"):
        files = sample_files()
        res = engine(files)
    if not res.get("ok"):
        return False
    store_result(res, files, [], "Data Pack BTC (có sẵn trong ứng dụng)")
    return True


# ----------------------------------------------------------------------------- bảng kiểm tra, hồ sơ, liên kết
def checks_table(checks):
    rows = [{"group": c["group"], "check": c["check"], "status": badge(c["status"]), "detail": c.get("detail", "")}
            for c in checks]
    table(rows, [("group", "Nhóm"), ("check", "Hạng mục"), ("status", "Kết quả"), ("detail", "Chi tiết")],
          raw_cols=("status",))


def profile_of(df, cust, exclude_ids):
    hist = df[(df.customer_id == cust) & (~df.transaction_id.isin(exclude_ids))]
    top = lambda c: hist[c].value_counts().head(3)  # noqa: E731
    return hist, {k: top(k) for k in ["device_id", "location", "ip_country", "payment_channel"]}


def network(tx, cust, df=None):
    """Sơ đồ liên kết đa khách hàng: khách hàng của hồ sơ – thiết bị – merchant, kèm các khách hàng KHÁC đã dùng
    chung thiết bị (dấu hiệu số 8, Phụ lục 02 TT 17/2024 sửa đổi bởi TT 25/2025)."""
    devs = tx.groupby("device_id").agg(n=("transaction_id", "size"), reg=("registered_customer_id", "first"),
                                        risk=("device_risk_level", "first")).reset_index()
    mers = tx.groupby("merchant_id").agg(n=("transaction_id", "size"), cat=("merchant_category", "first"),
                                          risk=("merchant_risk_level", "first")).reset_index()
    others = []
    if df is not None:
        o = df[df.device_id.isin(devs.device_id) & (df.customer_id != cust)]
        others = (o.groupby(["customer_id", "device_id"]).size().reset_index(name="n")).itertuples()
        others = list(others)
    ys = lambda k: np.linspace(0, 1, k + 2)[1:-1] if k > 1 else np.array([0.5])  # noqa: E731
    custs = [cust] + sorted({r.customer_id for r in others})
    pos = {}
    for y, c in zip(ys(len(custs)), custs):
        pos[("c", c)] = (0, y)
    for y, r in zip(ys(len(devs)), devs.itertuples()):
        pos[("d", r.device_id)] = (1, y)
    for y, r in zip(ys(len(mers)), mers.itertuples()):
        pos[("m", r.merchant_id)] = (2, y)
    fig = go.Figure()
    ex, ey = [], []
    for r in tx.itertuples():
        for a, b in [(("c", cust), ("d", r.device_id)), (("d", r.device_id), ("m", r.merchant_id))]:
            ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="#C5CEDC", width=1.6), hoverinfo="skip")
    if others:
        ox, oy = [], []
        for r in others:
            a, b = ("c", r.customer_id), ("d", r.device_id)
            ox += [pos[a][0], pos[b][0], None]; oy += [pos[a][1], pos[b][1], None]
        fig.add_scatter(x=ox, y=oy, mode="lines", line=dict(color=LEVEL_COLOR["HIGH"], width=1.6, dash="dot"),
                        hoverinfo="skip")
    nodes = [(pos[("c", cust)], cust, NAVY, f"Khách hàng {cust} (hồ sơ này)", 30)]
    seen = set()
    for r in others:
        if r.customer_id in seen:
            continue
        seen.add(r.customer_id)
        k = sum(x.n for x in others if x.customer_id == r.customer_id)
        nodes.append((pos[("c", r.customer_id)], r.customer_id, LEVEL_COLOR["HIGH"],
                      f"Khách hàng khác {r.customer_id}<br>dùng chung thiết bị · {k} giao dịch", 22))
    for r in devs.itertuples():
        own = str(r.reg) == str(cust)
        col = "#2B8A3E" if own else LEVEL_COLOR["CRITICAL"]
        nodes.append((pos[("d", r.device_id)], r.device_id, col,
                      f"Thiết bị {r.device_id}<br>{'Đăng ký cho khách này' if own else 'Không đăng ký cho khách này'}"
                      f"<br>Rủi ro nền {r.risk} · {r.n} giao dịch trong hồ sơ", 22))
    for r in mers.itertuples():
        nodes.append((pos[("m", r.merchant_id)], r.merchant_id, LEVEL_COLOR.get(r.risk, "#2557D6") if r.risk != "LOW" else BLUE,
                      f"Merchant {r.merchant_id}<br>{r.cat} · rủi ro {r.risk}<br>{r.n} giao dịch", 22))
    fig.add_scatter(x=[p[0][0] for p in nodes], y=[p[0][1] for p in nodes], mode="markers+text",
                    marker=dict(size=[p[4] for p in nodes], color=[p[2] for p in nodes], line=dict(color="#fff", width=2)),
                    text=[p[1] for p in nodes], textposition="bottom center", textfont=dict(size=11, color="#334155"),
                    hovertext=[p[3] for p in nodes], hovertemplate="%{hovertext}<extra></extra>")
    style(fig, 280 if len(custs) <= 3 else 340, legend=False)
    fig.update_xaxes(visible=False, range=[-0.35, 2.35])
    fig.update_yaxes(visible=False, range=[-0.15, 1.1])
    for x, t in [(0, "Khách hàng"), (1, "Thiết bị"), (2, "Merchant")]:
        fig.add_annotation(x=x, y=1.08, text=f"<b>{t}</b>", showarrow=False, font=dict(size=11.5, color=MUTED))
    return fig, len(custs) - 1


FIELD_LABELS = {"customer_id": "Khách hàng", "customer_segment": "Phân khúc", "home_location": "Nơi ở",
                "datetime": "Thời gian", "amount": "Số tiền", "avg_transaction_amount": "Mức nền của khách",
                "amt_ratio": "Gấp mức nền", "payment_channel": "Kênh", "transaction_status": "Trạng thái",
                "authentication_method": "Xác thực", "device_id": "Thiết bị", "device_type": "Loại thiết bị",
                "registered_customer_id": "Thiết bị đăng ký cho", "dev_age_days": "Tuổi thiết bị (ngày)",
                "location": "Tỉnh/thành giao dịch", "ip_country": "Quốc gia IP", "merchant_id": "Merchant",
                "merchant_category": "Ngành hàng", "merchant_risk_level": "Rủi ro nền merchant",
                "confidence": "Độ tin cậy", "case_id": "Case"}


def fmt_field(k, v):
    if v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and v in ("", "nan", "None")):
        return "–"
    if k in ("amount", "avg_transaction_amount"):
        return money(v)
    if k == "amt_ratio":
        return d(v, 2) + "×"
    if k == "datetime":
        return pd.Timestamp(v).strftime("%d/%m/%Y %H:%M:%S")
    if k == "dev_age_days":
        return n0(v)
    if k == "confidence":
        return pct(v)
    return str(v)


@st.cache_data(show_spinner=False, max_entries=8)
def build_files(rid, _res):
    out = {}
    b = io.BytesIO(); _res["submission"].to_csv(b, index=False, encoding="utf-8-sig"); out["submission_output.csv"] = b.getvalue()
    b = io.BytesIO(); _res["cases"].to_csv(b, index=False, encoding="utf-8-sig"); out["cases.csv"] = b.getvalue()
    b = io.BytesIO(); _res["df"].drop(columns=["ts"], errors="ignore").to_csv(b, index=False, encoding="utf-8-sig")
    out["scored_transactions_detail.csv"] = b.getvalue()
    out["kpi_report.json"] = json.dumps(_res["report"], ensure_ascii=False, indent=2, default=str).encode("utf-8")
    b = io.BytesIO(); pd.DataFrame(_res["checks"]).to_csv(b, index=False, encoding="utf-8-sig"); out["data_checks.csv"] = b.getvalue()
    return out




# ----------------------------------------------------------------------------- khối giao diện dùng chung
def kcard(value, label, color, sub="", tag="", tag_bg=None):
    t = f'<div class="tag" style="--cbg:{tag_bg or "#F1F3F5"}">{tag}</div>' if tag else ""
    s = f'<div class="s">{sub}</div>' if sub else ""
    return (f'<div class="fs-k" style="--c:{color}"><div class="v">{value}</div><div class="l"><i></i>{label}</div>{s}{t}</div>')


def kgrid(cards):
    H('<div class="fs fs-kgrid">' + "".join(cards) + "</div>")


def stdot(s):
    return f'<span class="fs-st"><i style="background:{STATUS_COLOR.get(s, "#98A2B3")}"></i>{e(s)}</span>'


def topbar(res, page_title):
    if res is not None:
        df = res["df"]
        src = (f'{e(res["source"])} · {n0(len(df))} giao dịch {df.datetime.min():%d/%m/%Y} – {df.datetime.max():%d/%m/%Y}')
        run = f'<div class="fs-run">Lần chạy gần nhất<b>{e(res["started"])}</b></div><span class="fs-ok"><i>✓</i>Hoàn thành</span>'
    else:
        src, run = "Chưa có dữ liệu – tải Data Pack để bắt đầu", '<span class="fs-ok" style="background:#FFF4E6;color:#D9480F;border-color:#FFD8A8">Chưa chạy</span>'
    H(f'<div class="fs fs-top"><div><div class="ttl">{e(page_title)}</div><div class="src">{src}</div></div>'
      f'<div class="right">{run}<div class="fs-av" title="Chuyên viên">{USER_SVG}</div></div></div>')


def sidebar(res):
    cur = st.session_state.get("nav", "cases")
    with st.sidebar:
        H(f'<div class="fs fs-brand">{SHIELD.format(w=44)}<div><div class="nm">Fraud Shield</div>'
          f'<div class="sb">Digital Payment Risk Console</div></div></div>')
        for k, lab, ic, _ in NAV:
            st.button(lab, icon=ic, key=f"nav_{k}", type="primary" if k == cur else "secondary", width="stretch",
                      on_click=lambda k=k: st.session_state.update(nav=k))
    return cur


def add_types(df):
    """Loại rủi ro chính của từng giao dịch (luật đầu tiên bật theo RISK_TYPES)."""
    conds = [df[c].astype(bool).values for c, _ in RISK_TYPES]
    return np.select(conds, [t for _, t in RISK_TYPES], default=OTHER_TYPE)


def case_types(res):
    df, cases = res["df"], res["cases"]
    rank = {t: i for i, t in enumerate(TYPE_ORDER)}
    tmap = dict(zip(df.transaction_id, add_types(df)))
    return cases.transaction_ids.map(lambda s: min((tmap[t] for t in s.split(";")), key=rank.get))


def case_status_series(res):
    stt = status_frame(res).set_index("transaction_id")["status"]

    def one(ids):
        ss = [stt.get(t, "Chưa xử lý") for t in ids.split(";")]
        if all(x in DONE for x in ss):
            return "Đã xử lý"
        if any(x == "Đang xác minh" for x in ss):
            return "Đang xác minh"
        if any(x in DONE for x in ss):
            return "Xử lý một phần"
        return "Chưa xử lý"
    return res["cases"].transaction_ids.map(one)


def status_legend(counts, total, keys=STATUSES):
    bar = "".join(f'<i style="width:{counts.get(k, 0) / max(1, total) * 100:.2f}%;background:{STATUS_COLOR[k]}" '
                  f'title="{k}"></i>' for k in keys)
    leg = "".join(f'<span><i class="fs-dot" style="background:{STATUS_COLOR[k]}"></i>{k} <b>{n0(counts.get(k, 0))}</b></span>'
                  for k in keys)
    return f'<div class="fs"><div class="fs-prog">{bar}</div><div class="fs-legend">{leg}</div></div>'


def no_data():
    st.info("Chưa có dữ liệu. Vào mục **Dữ liệu & Chạy mô hình** để tải Data Pack và chạy mô hình.",
            icon=":material/info:")


# ============================================================================= 1. TỔNG QUAN
def page_overview(res):
    df, rep = res["df"], res["report"]
    n = len(df)
    n_alert = int(df.predicted_fraud.sum())
    cs = case_status_series(res)
    open_cases = int((cs != "Đã xử lý").sum())
    train = res["mode"] == "train"
    b = rep["comparison_nested_cv"]["Gộp 3 tầng – có hồ sơ (B, chính thức)"] if train else None
    n_step = int(rep.get("n_stepup", (df.risk_level == "MEDIUM").sum()))
    k = rep.get("kpi") if train else None
    kgrid([
        kcard(n0(n), "Giao dịch đã chấm điểm", BLUE, f"{df.datetime.min():%d/%m/%Y} – {df.datetime.max():%d/%m/%Y}"),
        kcard(n0(n_alert), "Tạm dừng giao dịch", LEVEL_COLOR["CRITICAL"],
              f"CRITICAL + HIGH · gom thành {n0(len(res['cases']))} hồ sơ ({n0(open_cases)} chưa xử lý xong)",
              pct(n_alert / n), "#FFE3E3"),
        kcard(n0(n_step), "Yêu cầu xác thực thêm", LEVEL_COLOR["MEDIUM"], "MEDIUM · không chặn, hỏi OTP/sinh trắc học",
              pct(n_step / n), "#FFF3BF"),
        kcard(d(k["intervene_recall"]), "Recall can thiệp", GREEN,
              f"{k['intervene_tp']}/{k['n_fraud']} gian lận bị chặn hoặc phải xác thực thêm · kiểm định 5-fold",
              "Đạt ≥ 0,90" if k["intervene_recall"] >= .9 else "Chưa đạt 0,90", "#E6F4EA")
        if k else kcard("–", "Recall can thiệp", GREEN, "Cần tệp nhãn để đánh giá")])
    if k:
        with card("ov_kpi"):
            title("Bộ chỉ số 3 tầng hành động",
                  "Kiểm định 5-fold, ngưỡng chọn trên phần huấn luyện của từng fold · chỉ dùng dữ liệu quá khứ")
            H('<div class="fs fs-kgrid" style="margin:4px 0 4px">' + "".join([
                kcard(pct(k["intervene_recall"]), "Can thiệp (chặn hoặc xác thực thêm)", GREEN,
                      f"đổi lại {pct(k['stepup_legit_rate'], 2)} khách thật phải xác thực thêm vài giây"),
                kcard(pct(k["value_recall_hard"]), "Giá trị tiền gian lận bị chặn", BLUE,
                      f"{pct(k['value_recall_intervene'])} nếu tính cả xác thực thêm"),
                kcard(pct(k["case_recall"]), "Vụ tấn công bị phát hiện", PURPLE,
                      f"{k['attacks_detected']}/{k['n_attacks']} vụ có ít nhất 1 giao dịch bị chặn ngay"),
                kcard(pct(k["hard_recall"]), "Chặn ngay tại thời điểm giao dịch", LEVEL_COLOR["HIGH"],
                      f"{k['hard_tp']}/{k['n_fraud']} gian lận · {k['hard_fp']} báo nhầm / {n0(k['n_legit'])} giao dịch thật")])
              + "</div>")

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1, card("ov_donut"):
        title("Phân bố mức rủi ro", f"Trên toàn bộ {n0(n)} giao dịch")
        cnt = rep["risk_level_counts"]
        vals = [cnt.get(l, 0) for l in LEVELS]
        fig = go.Figure(go.Pie(labels=[f"{l.title()} · {n0(v)} ({pct(v / n)})" for l, v in zip(LEVELS, vals)], values=vals,
                               hole=0.58, sort=False, direction="clockwise", textinfo="none",
                               marker=dict(colors=[LEVEL_COLOR[l] for l in LEVELS], line=dict(color="#fff", width=2)),
                               hovertemplate="%{label}<extra></extra>"))
        fig.add_annotation(text=f"<b>{n0(n)}</b><br><span style='font-size:11px'>giao dịch</span>", showarrow=False,
                           font=dict(size=17, color=INK))
        style(fig, 300, legend=True)
        fig.update_layout(legend=dict(orientation="v", x=1.0, xanchor="left", y=0.5, yanchor="middle", font=dict(size=12.5)),
                          margin=dict(l=6, r=10, t=8, b=6))
        chart(fig, "ov_pie")
    with c2, card("ov_prf"):
        title("Khả năng phát hiện ở mức chặn (HIGH/CRITICAL)")
        if train:
            names, vals, cols = ["Precision", "Recall", "F1-score", "PR-AUC"], \
                [b["precision"], b["recall"], b["f1"], b["pr_auc"]], [BLUE, "#3B7DDD", TEAL, PURPLE]
            fig = go.Figure(go.Bar(x=names, y=vals, marker_color=cols, text=[d(v) for v in vals], textposition="outside",
                                   cliponaxis=False, width=0.55, hovertemplate="%{x}: %{text}<extra></extra>",
                                   textfont=dict(size=14, color=INK)))
            style(fig, 300, legend=False)
            fig.update_yaxes(range=[0, 1.12], tickformat=".1f", dtick=0.2)
            chart(fig, "ov_prf")
        else:
            st.caption("Data Pack này không có nhãn nên chưa đánh giá được độ chính xác.")

    c1, c2 = st.columns([1.25, 1], gap="medium")
    with c1, card("ov_trend"):
        title("Xu hướng giao dịch cảnh báo theo thời gian", "Theo tuần · trục phải là số giao dịch cảnh báo")
        wk = df.datetime.dt.to_period("W").dt.start_time
        tot = df.groupby(wk).size()
        al = df[df.predicted_fraud == 1].groupby(wk[df.predicted_fraud == 1]).size().reindex(tot.index, fill_value=0)
        x = [pd.Timestamp(w).strftime("%d/%m") for w in tot.index]
        fig = go.Figure()
        fig.add_scatter(x=x, y=tot.values, name="Tổng giao dịch", mode="lines+markers", line=dict(color=BLUE, width=2.5),
                        marker=dict(size=5), hovertemplate="Tuần %{x}<br>%{y} giao dịch<extra></extra>")
        fig.add_scatter(x=x, y=al.values, name="Giao dịch cảnh báo", mode="lines+markers", yaxis="y2",
                        line=dict(color=LEVEL_COLOR["CRITICAL"], width=2.5), marker=dict(size=5),
                        hovertemplate="Tuần %{x}<br>%{y} cảnh báo<extra></extra>")
        style(fig, 300, legend=True)
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, rangemode="tozero",
                                      tickfont=dict(size=11.5, color=LEVEL_COLOR["CRITICAL"])),
                          legend=dict(orientation="h", y=-0.18, yanchor="top", x=0.5, xanchor="center"))
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(type="category", nticks=9, tickangle=0)
        chart(fig, "ov_trend")
    with c2, card("ov_types"):
        title("Top 5 loại rủi ro phổ biến", "Trong các giao dịch cảnh báo, theo dấu hiệu chính")
        a = df[df.predicted_fraud == 1]
        vc = pd.Series(add_types(a)).value_counts().head(5).sort_values()
        pal = ["#3B7DDD", "#F03E3E", "#FF922B", "#FCC419", "#22B8CF"][:len(vc)][::-1]
        fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", marker_color=pal, text=vc.values,
                               textposition="outside", cliponaxis=False, width=0.62,
                               hovertemplate="%{y}<br>%{x} giao dịch<extra></extra>"))
        style(fig, 300, legend=False)
        fig.update_xaxes(range=[0, max(1, vc.max() if len(vc) else 1) * 1.18])
        chart(fig, "ov_types")

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1, card("ov_prog"):
        allst = status_frame(res)
        cnt = allst.status.value_counts().to_dict()
        n_done = sum(cnt.get(k, 0) for k in DONE)
        title("Tiến độ xử lý cảnh báo", f"Đã có kết luận {n0(n_done)}/{n0(len(allst))} giao dịch cảnh báo – từ nhãn đã xác minh "
              "trong Data Pack hoặc quyết định của chuyên viên")
        H(status_legend(cnt, len(allst)))
        H('<div class="fs" style="margin-top:12px">' + "".join(
            f'<div class="fs-mini"><span>{badge(l)}</span><span class="why">{LEVEL_INFO[l][1]}</span>'
            f'<b>{n0(rep["risk_level_counts"].get(l, 0))}</b></div>' for l in LEVELS) + "</div>")
    with c2, card("ov_urgent"):
        title("Hồ sơ cần xử lý trước", "5 hồ sơ chưa xử lý xong có điểm cao nhất – mở ở mục Hồ sơ cảnh báo")
        cases = res["cases"].assign(st_=cs.values)
        top = cases[cases.st_ != "Đã xử lý"].head(5)
        if top.empty:
            st.success("Mọi hồ sơ đã có kết luận.", icon=":material/check_circle:")
        else:
            H('<div class="fs">' + "".join(
                f'<div class="fs-mini"><span class="id">{e(r.case_id)}</span>{badge(r.case_level)}'
                f'<span class="why">{e(r.customer_id)} · {int(r.n_transactions)} GD · {e(str(r.top_reason).split(";")[0])}</span>'
                f'<b>{d(r.max_risk_score, 4)}</b></div>' for r in top.itertuples()) + "</div>")
            st.button("Mở hàng đợi của tôi", icon=":material/arrow_forward:", key="ov_go_cases",
                      on_click=lambda: st.session_state.update(nav="cases"))


# ============================================================================= 2. GIAO DỊCH
def tx_status(res):
    df = res["df"]
    nov = df["novelty"].values == 1 if "novelty" in df else np.zeros(len(df), dtype=bool)
    s = pd.Series(np.where(df.risk_level == "MEDIUM", "Xác thực thêm", np.where(nov, "Nghi vấn mẫu mới", "Bình thường")),
                  index=df.transaction_id.values)
    a = status_frame(res)
    s.loc[a.transaction_id.values] = a.status.values
    return s


RULE_SHORT = {k: v.split(" – ")[0] for k, v in F.RULE_NAMES.items()}


def advanced_filters(df, rid):
    """Bộ lọc đa chiều (ẩn trong ngăn kéo) – trả về bộ lọc dạng tuple để làm chữ ký phân trang."""
    rl = st.multiselect("Luật đã bật (giao dịch phải bật tất cả)", list(F.RULE_W), format_func=lambda k: F.RULE_NAMES[k],
                        key=f"adv_rules_{rid}")
    no_rule = st.toggle("Chỉ giao dịch không bật luật chính nào nhưng điểm học máy cao (≥ 0,5)", key=f"adv_norule_{rid}")
    c = st.columns(2)
    amin = c[0].number_input("Số tiền từ (VND)", min_value=0, value=0, step=500000, key=f"adv_amin_{rid}")
    amax = c[1].number_input("đến (VND, 0 = không giới hạn)", min_value=0, value=0, step=500000, key=f"adv_amax_{rid}")
    ratio = st.slider("Gấp mức nền của khách ít nhất", 0.0, 10.0, 0.0, 0.5, key=f"adv_ratio_{rid}")
    hours = st.slider("Khung giờ", 0, 23, (0, 23), key=f"adv_hour_{rid}")
    ips = st.multiselect("Quốc gia IP", sorted(df.ip_country.astype(str).unique()), key=f"adv_ip_{rid}")
    dev = st.slider("Số chiều lệch thói quen ít nhất", 0, 5, 0, key=f"adv_dev_{rid}")
    return (tuple(rl), no_rule, int(amin), int(amax), float(ratio), tuple(hours), tuple(ips), int(dev))


def apply_advanced(v, adv):
    rl, no_rule, amin, amax, ratio, hours, ips, dev = adv
    for k in rl:
        v = v[v[k] == 1]
    if no_rule:
        main = ["flag_R1_strong", "flag_R2", "flag_R3", "flag_R4", "flag_R5", "flag_R6", "flag_R7"]
        v = v[(v[main].sum(axis=1) == 0) & (v.ml_score >= 0.5)]
    if amin:
        v = v[v.amount >= amin]
    if amax:
        v = v[v.amount <= amax]
    if ratio:
        v = v[v.amt_ratio >= ratio]
    if tuple(hours) != (0, 23):
        v = v[v.datetime.dt.hour.between(hours[0], hours[1])]
    if ips:
        v = v[v.ip_country.astype(str).isin(ips)]
    if dev:
        v = v[v.profile_dev_count >= dev]
    return v


def page_tx(res):
    df, rid = res["df"], res["rid"]
    stt = tx_status(res)
    d0, d1 = df.datetime.min().date(), df.datetime.max().date()
    with card("tx_filter"):
        c = st.columns([1, 1, 1, 1, 1], vertical_alignment="bottom")
        lv = c[0].selectbox("Mức rủi ro", ["Tất cả"] + LEVELS, key=f"tx_lv_{rid}")
        stv = c[1].selectbox("Trạng thái", ["Tất cả"] + STATUSES + ["Xác thực thêm", "Nghi vấn mẫu mới", "Bình thường"],
                             key=f"tx_st_{rid}")
        f0 = c[2].date_input("Từ ngày", value=d0, min_value=d0, max_value=d1, format="DD/MM/YYYY", key=f"tx_d0_{rid}")
        f1 = c[3].date_input("Đến ngày", value=d1, min_value=d0, max_value=d1, format="DD/MM/YYYY", key=f"tx_d1_{rid}")
        srt = c[4].selectbox("Sắp xếp", ["Điểm rủi ro cao nhất", "Mới nhất", "Số tiền lớn nhất"], key=f"tx_sort_{rid}")
        c = st.columns([2.6, 0.75, 0.95, 1], vertical_alignment="bottom")
        q = c[0].text_input("Tìm kiếm", placeholder="Tìm theo mã giao dịch, khách hàng, merchant…", key=f"tx_q_{rid}",
                            label_visibility="collapsed").strip().upper()
        c[1].button("Tìm kiếm", type="primary", width="stretch", key=f"tx_go_{rid}")
        with c[2].popover("Bộ lọc nâng cao", icon=":material/tune:", width="stretch"):
            adv = advanced_filters(df, rid)
    v = df.assign(status=df.transaction_id.map(stt))
    v = apply_advanced(v, adv)
    if lv != "Tất cả":
        v = v[v.risk_level == lv]
    if stv != "Tất cả":
        v = v[v.status == stv]
    f0 = f0 if f0 else d0
    f1 = f1 if f1 else d1
    v = v[(v.datetime.dt.date >= f0) & (v.datetime.dt.date <= f1)]
    if q:
        v = v[v.transaction_id.str.upper().str.contains(q, regex=False) | v.customer_id.str.upper().str.contains(q, regex=False)
              | v.merchant_id.astype(str).str.upper().str.contains(q, regex=False)
              | v.merchant_category.astype(str).str.upper().str.contains(q, regex=False)]
    v = v.sort_values({"Điểm rủi ro cao nhất": "priority_rank", "Mới nhất": "datetime", "Số tiền lớn nhất": "amount"}[srt],
                      ascending=srt == "Điểm rủi ro cao nhất")
    b = io.BytesIO()
    v[["transaction_id", "datetime", "customer_id", "amount", "merchant_id", "merchant_category", "payment_channel",
       "location", "risk_score", "risk_level", "status", "case_id", "alert_reason"]].to_csv(b, index=False, encoding="utf-8-sig")
    c[3].download_button("Tải xuống CSV", b.getvalue(), "giao_dich_loc.csv", "text/csv", icon=":material/download:",
                         width="stretch", key=f"tx_dl_{rid}")

    # phân trang
    sig = f"{lv}|{stv}|{f0}|{f1}|{q}|{srt}|{adv}"
    pk, sk = f"tx_page_{rid}", f"tx_sig_{rid}"
    if st.session_state.get(sk) != sig:
        st.session_state[sk] = sig
        st.session_state[pk] = 1
    size = st.session_state.get(f"tx_size_{rid}", 10)
    pages = max(1, -(-len(v) // size))
    page = min(max(1, st.session_state.get(pk, 1)), pages)
    view = v.iloc[(page - 1) * size: page * size]

    with card("tx_table"):
        title("Danh sách giao dịch", f"{n0(len(v))} giao dịch khớp bộ lọc · điểm rủi ro tô màu theo mức")
        if view.empty:
            st.info("Không có giao dịch nào khớp bộ lọc.", icon=":material/info:")
        else:
            rows = []
            for r in view.itertuples():
                rows.append(
                    f'<tr><td class="mono">{e(r.transaction_id)}</td><td>{r.datetime:%d/%m/%Y %H:%M}</td>'
                    f'<td class="num">{n0(r.amount)}</td><td>{e(r.merchant_id)} <span style="color:#64748B">· '
                    f'{e(r.merchant_category)}</span></td><td>{e(r.customer_id)}</td><td>{e(r.payment_channel)}</td>'
                    f'<td class="num" style="color:{LEVEL_COLOR[r.risk_level] if r.risk_level != "LOW" else "#475467"};'
                    f'font-weight:700">{d(r.risk_score, 2)}</td><td>{badge(r.risk_level)}</td>'
                    f'<td>{ACTION_SHORT[r.risk_level]}</td><td>{stdot(r.status)}</td></tr>')
            H('<div class="fs fs-tw"><table class="fs-table"><thead><tr><th>Mã giao dịch</th><th>Thời gian</th>'
              '<th class="num">Số tiền (VND)</th><th>Merchant</th><th>Khách hàng</th><th>Kênh</th><th class="num">Điểm rủi ro</th>'
              '<th>Mức rủi ro</th><th>Hành động</th><th>Trạng thái</th></tr></thead><tbody>' + "".join(rows)
              + "</tbody></table></div>")
        c = st.columns([0.9, 0.9, 2.4, 0.55, 0.9, 0.55], vertical_alignment="center")
        c[0].selectbox("Hiển thị", [10, 20, 50, 100], index=[10, 20, 50, 100].index(size), key=f"tx_size_{rid}",
                       format_func=lambda x: f"{x} dòng/trang", label_visibility="collapsed")
        c[1].caption(f"{n0((page - 1) * size + min(1, len(view)))}–{n0((page - 1) * size + len(view))} / {n0(len(v))}")
        c[3].button("Trước", icon=":material/chevron_left:", key=f"tx_prev_{rid}", disabled=page <= 1, width="stretch",
                    on_click=lambda: st.session_state.update({pk: page - 1}))
        c[4].markdown(f"<div style='text-align:center;font-size:13px;color:#475467'>Trang <b>{page}</b> / {n0(pages)}</div>",
                      unsafe_allow_html=True)
        c[5].button("Sau", icon=":material/chevron_right:", icon_position="right", key=f"tx_next_{rid}", disabled=page >= pages, width="stretch",
                    on_click=lambda: st.session_state.update({pk: page + 1}))

    if not view.empty:
        with card("tx_pick"):
            ids = view.transaction_id.tolist()
            lab = dict(zip(view.transaction_id, [f"{t} · {l} · điểm {d(s, 4)}" for t, l, s in
                                                 zip(view.transaction_id, view.risk_level, view.risk_score)]))
            tid = st.selectbox("Xem chi tiết giao dịch (trong trang đang hiển thị)", ids, format_func=lambda t: lab.get(t, t),
                               key=f"tx_pick_{rid}_{page}_{sig}")
        tx_detail(res, tid)


def tx_detail(res, tid):
    df, rep = res["df"], res["report"]
    thr, med = rep["threshold"], rep["medium_threshold"]
    r = df[df.transaction_id == tid].iloc[0]
    with card("tx_head"):
        H(f"""<div class="fs"><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px">
          <span style="font-size:19px;font-weight:800;color:{NAVY}">{e(r.transaction_id)}</span>{badge(r.risk_level)}
          <span class="fs-sub" style="margin:0">Hạng ưu tiên #{int(r.priority_rank)} / {n0(len(df))}</span></div>
          <div class="fs-case">
            <div class="item"><div class="k">Điểm rủi ro</div><div class="v">{d(r.risk_score, 4)}</div></div>
            <div class="item"><div class="k">Ngưỡng chặn · xác thực thêm</div><div class="v">{d(thr, 4)} · {d(med, 4)}</div></div>
            <div class="item"><div class="k">Hành động</div><div class="v">{e(F.ACTIONS[r.risk_level])}</div></div>
            <div class="item"><div class="k">Số tiền</div><div class="v">{money(r.amount)}</div></div>
            <div class="item"><div class="k">Hồ sơ</div><div class="v">{e(r.case_id) if isinstance(r.case_id, str) else '–'}</div></div>
          </div><div class="fs-reason"><b>Lý do:</b> {e(r.alert_reason)}</div></div>""")
    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1, card("tx_score"):
        w = F.W_FUSION
        title("Điểm được tạo thành thế nào",
              f"RiskScore = 0,6 × Logistic + 0,3 × Rule + 0,1 × Isolation Forest = {d(r.risk_score, 4)}")
        parts = [("Hồi quy logistic", w[0] * r.ml_score, r.ml_score, BLUE),
                 ("Hệ luật", w[1] * r.rule_score, r.rule_score, LEVEL_COLOR["HIGH"]),
                 ("Isolation Forest", w[2] * r.anomaly_score, r.anomaly_score, PURPLE)]
        fig = go.Figure()
        for nm, v, raw, col in parts:
            fig.add_bar(y=["Điểm"], x=[v], name=f"{nm}: {d(raw, 3)}", orientation="h", marker_color=col,
                        hovertemplate=f"{nm}<br>điểm gốc {d(raw, 4)} · đóng góp {d(v, 4)}<extra></extra>")
        fig.add_vline(x=thr, line=dict(color=NAVY, dash="dash", width=1.5))
        fig.add_annotation(x=thr, y=1.0, yref="paper", text=f"Chặn {d(thr, 2)}", showarrow=False, yanchor="bottom",
                           font=dict(size=11, color=NAVY))
        fig.add_vline(x=med, line=dict(color="#A06A00", dash="dot", width=1.5))
        fig.add_annotation(x=med, y=1.0, yref="paper", text=f"Xác thực thêm {d(med, 2)}", showarrow=False,
                           yanchor="bottom", xanchor="right", font=dict(size=11, color="#A06A00"))
        style(fig, 190, barmode="stack", legend=True)
        fig.update_layout(legend=dict(y=-0.25, yanchor="top", traceorder="normal"), margin=dict(t=24, b=6))
        fig.update_xaxes(range=[0, 1], dtick=0.1)
        fig.update_yaxes(visible=False)
        chart(fig, f"sc_{tid}")
        H('<div class="fs" style="margin-top:4px">' + "".join(
            f'<span class="fs-flag {"on" if r[k] else "off"}">{"●" if r[k] else "○"} {e(F.RULE_NAMES[k])}</span>'
            for k in F.RULE_W) + "</div>")
    with c2, card("tx_prof"):
        title("Lệch so với thói quen của khách", "Chỉ dùng các giao dịch trước đó của chính khách hàng")
        yn = lambda v: '<span class="new">Có</span>' if v else "Không"  # noqa: E731
        rows = [{"k": "Số tiền lệch (z-score so với chính khách)", "v": d(r.prof_amt_z, 2)},
                {"k": "Lần đầu dùng thiết bị này", "v": yn(r.prof_new_device)},
                {"k": "Lần đầu giao dịch ở tỉnh này", "v": yn(r.prof_new_location)},
                {"k": "Lần đầu dùng kênh này", "v": yn(r.prof_new_channel)},
                {"k": "Lần đầu dùng IP nước này", "v": yn(r.prof_new_ip)},
                {"k": "Giao dịch của khách trong 5 phút gần nhất", "v": str(int(r.n_txn_5m))},
                {"k": "Giao dịch thất bại trong 30 phút gần nhất", "v": str(int(r.n_fail_30m))}]
        table(rows, [("k", "Chỉ báo"), ("v", "Giá trị")], num_cols=("v",), raw_cols=("v",))
    with card("tx_fields"):
        title("Thông tin giao dịch")
        H('<div class="fs fs-kv">' + "".join(
            f"<div><span>{e(lab_)}</span><b>{e(fmt_field(k, r[k]))}</b></div>"
            for k, lab_ in FIELD_LABELS.items() if k in r.index) + "</div>")


# ============================================================================= 3. HỒ SƠ CẢNH BÁO
QCHIPS = ["Tất cả hồ sơ", "Khẩn cấp chưa xử lý", "Thiết bị dùng chung (R7)", "Xác thực thêm · merchant HIGH",
          "Nghi vấn mẫu mới"]


@st.dialog("Xử lý hàng loạt", width="large")
def bulk_dialog(res, sel, rtype):
    ids = [(c, t) for c, s in zip(sel.case_id, sel.transaction_ids) for t in s.split(";")]
    H(f'<div class="fs fs-sub" style="font-size:14px;color:#1F2937">Bạn sắp ghi <b>một kết luận chung</b> cho '
      f'<b>{len(sel)} hồ sơ</b> (gồm <b>{len(ids)} giao dịch</b>) cùng loại rủi ro <b>{e(rtype)}</b>: '
      f'{e(", ".join(sel.case_id.head(8)))}{" …" if len(sel) > 8 else ""}</div>')
    dec = st.radio("Kết luận", DECISIONS, horizontal=True, key="bulk_dec")
    a, b = st.columns(2)
    who = a.text_input("Người xử lý", placeholder="Tên chuyên viên", key="bulk_who")
    note = b.text_input("Ghi chú", placeholder="VD: cùng một mẫu tấn công", key="bulk_note")
    c1, c2 = st.columns(2)
    if c1.button(f"Xác nhận ghi cho {len(sel)} hồ sơ", type="primary", width="stretch", key="bulk_ok"):
        save_decisions(res["ds"], [{"transaction_id": t, "case_id": c, "decision": dec, "note": f"[Hàng loạt] {note}"[:500],
                                    "user": (who or "chuyên viên")[:60], "time": now_vn(),
                                    "threshold": res["report"]["threshold"]} for c, t in ids])
        st.session_state.flash = f"Đã ghi “{dec}” cho {len(sel)} hồ sơ ({len(ids)} giao dịch)"
        st.rerun()
    if c2.button("Huỷ", width="stretch", key="bulk_cancel"):
        st.rerun()


def sim_actions(res, case_id, target, key):
    """Nút tác vụ nghiệp vụ – chỉ MÔ PHỎNG, ghi vào nhật ký."""
    H('<div class="fs-sub" style="margin:6px 0 4px">Tác vụ nhanh <b>(mô phỏng – chỉ ghi nhật ký, không tác động hệ thống '
      'thật)</b></div>')
    cols = st.columns(3)
    for col, (k, lab, ic) in zip(cols, SIM_ACTIONS):
        if col.button(lab, icon=ic, width="stretch", key=f"act_{k}_{key}"):
            save_action(res["ds"], case_id, target, lab)
            st.session_state.flash = f"{lab} · {target} (mô phỏng) – đã ghi nhật ký"
            st.rerun()


def txn_queue(res, kind):
    """Hai hàng đợi theo giao dịch: vùng xác thực thêm tại merchant HIGH, và nghi vấn mẫu mới."""
    df, rid = res["df"], res["rid"]
    if kind == QCHIPS[3]:
        v = df[(df.risk_level == "MEDIUM") & (df.merchant_high == 1)]
        sub = ("Giao dịch tại merchant rủi ro cao ở vùng MEDIUM: hệ thống không chặn mà tự động yêu cầu xác thực thêm. "
               "Chuyên viên rà soát để phát hiện mẫu lừa đảo tại merchant.")
    else:
        v = df[df.get("novelty", pd.Series(0, index=df.index)) == 1]
        sub = (f"Giao dịch không bị chặn và không phải xác thực thêm nhưng dị biệt nhất so với giai đoạn quá khứ "
               f"(Isolation Forest). Giới hạn khoảng {F.NOV_BUDGET} ca cho cả kỳ để không quá tải – dùng để khám phá "
               "kiểu gian lận chưa có trong luật.")
    v = v.sort_values("anomaly_score" if kind == QCHIPS[4] else "risk_score", ascending=False).reset_index(drop=True)
    if v.empty:
        st.info("Hàng đợi này đang trống.", icon=":material/inbox:")
        return
    log = load_log(res["ds"])
    last = log.drop_duplicates("transaction_id", keep="last").set_index("transaction_id").decision \
        if len(log) else pd.Series(dtype=str)
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1, card("tq_list"):
        title(f"{kind} ({n0(len(v))} giao dịch)", sub)
        tbl = pd.DataFrame({"Giao dịch": v.transaction_id, "Thời gian": v.datetime.dt.strftime("%d/%m %H:%M"),
                            "Khách hàng": v.customer_id, "Số tiền": v.amount.map(money),
                            "Merchant": v.merchant_id + " · " + v.merchant_category.astype(str),
                            "Điểm": v.risk_score, "Dị biệt": v.anomaly_score,
                            "Kết luận": v.transaction_id.map(last).fillna("Chưa xem")})
        ev = grid(tbl, height=460, on_select="rerun", selection_mode="single-row", key=f"tq_tbl_{rid}_{kind}",
                  column_config={"Điểm": st.column_config.NumberColumn(format="%.3f"),
                                 "Dị biệt": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
    try:
        rows = ev.selection.rows
    except Exception:  # noqa
        rows = []
    r = v.iloc[rows[0] if rows and rows[0] < len(v) else 0]
    with c2, card("tq_panel"):
        H(f"""<div class="fs"><div class="fs-phead"><div class="fs-h" style="font-size:17px">{e(r.transaction_id)}</div>
          {badge(r.risk_level)}</div><div class="fs-dl">
          <span>Khách hàng</span><b>{e(r.customer_id)} · {e(r.customer_segment)} · {e(r.home_location)}</b>
          <span>Thời gian</span><b>{dt(r.datetime)}</b><span>Số tiền</span><b>{money(r.amount)} (gấp {d(r.amt_ratio, 1)} lần mức nền)</b>
          <span>Merchant</span><b>{e(r.merchant_id)} · {e(r.merchant_category)} · rủi ro {e(r.merchant_risk_level)}</b>
          <span>Thiết bị · tỉnh · IP</span><b>{e(r.device_id)} · {e(r.location)} · {e(r.ip_country)}</b>
          <span>Điểm · dị biệt</span><b>{d(r.risk_score, 4)} · {d(r.anomaly_score, 3)}</b>
          <span>Hành động tự động</span><b>{e(F.ACTIONS[r.risk_level])}</b></div>
          <div class="fs-reason"><b>Lý do:</b> {e(r.alert_reason)}</div></div>""")
        with st.form(f"tqf_{rid}_{r.transaction_id}", clear_on_submit=True, border=False):
            dec = st.radio("Kết luận của chuyên viên", DECISIONS, horizontal=True)
            a, b = st.columns(2)
            who = a.text_input("Người xử lý", placeholder="Tên chuyên viên")
            note = b.text_input("Ghi chú", placeholder="VD: mẫu mới, cần thêm luật")
            if st.form_submit_button("Ghi kết luận", type="primary", icon=":material/task_alt:", width="stretch"):
                save_decisions(res["ds"], [{"transaction_id": r.transaction_id, "case_id": "", "decision": dec,
                                            "note": note[:500], "user": (who or "chuyên viên")[:60], "time": now_vn(),
                                            "threshold": res["report"]["threshold"]}])
                st.session_state.flash = f"Đã ghi kết luận cho {r.transaction_id}"
                st.rerun()
        sim_actions(res, "", r.device_id, f"tq_{r.transaction_id}")


def page_cases(res):
    df, cases, rid = res["df"], res["cases"], res["rid"]
    if cases.empty:
        st.info("Không có cảnh báo nào ở ngưỡng hiện tại.")
        return
    cases = cases.assign(case_status=case_status_series(res).values, risk_type=case_types(res).values)
    tx_r7 = set(df.loc[df.flag_R7 == 1, "transaction_id"])
    cases = cases.assign(has_r7=cases.transaction_ids.map(lambda s: any(t in tx_r7 for t in s.split(";"))),
                         sla=cases.case_level.map(SLA_TXT))
    cnt = cases.case_status.value_counts()
    urgent = cases[(cases.case_level == "CRITICAL") & (cases.case_status != "Đã xử lý")]
    hub = live_hub(hub_key(res))
    if hub.get("ready"):
        st.fragment(queue_live_box, run_every=2.0)(hub, res)
    kgrid([kcard(n0(len(cases)), "Tổng hồ sơ", LEVEL_COLOR["CRITICAL"], f"gom từ {n0(res['report']['n_alerts'])} giao dịch bị tạm dừng"),
           kcard(n0(cnt.get("Chưa xử lý", 0)), "Chưa xử lý", LEVEL_COLOR["HIGH"],
                 f"trong đó {n0(len(urgent))} hồ sơ khẩn cấp (SLA 5 phút)"),
           kcard(n0(cnt.get("Đang xác minh", 0) + cnt.get("Xử lý một phần", 0)), "Đang xem xét", BLUE,
                 "đang xác minh hoặc xử lý một phần"),
           kcard(n0(cnt.get("Đã xử lý", 0)), "Đã xử lý", GREEN, "mọi giao dịch trong hồ sơ đã có kết luận")])

    n_med_mh = int(((df.risk_level == "MEDIUM") & (df.merchant_high == 1)).sum())
    n_nov = int(df["novelty"].sum()) if "novelty" in df else 0
    counts = {QCHIPS[0]: len(cases), QCHIPS[1]: len(urgent), QCHIPS[2]: int(cases.has_r7.sum()), QCHIPS[3]: n_med_mh,
              QCHIPS[4]: n_nov}
    chip = st.pills("Lọc nhanh", QCHIPS, selection_mode="single", default=QCHIPS[0], key=f"chip_{rid}",
                    format_func=lambda x: f"{x} · {n0(counts[x])}", label_visibility="collapsed") or QCHIPS[0]
    if chip in (QCHIPS[3], QCHIPS[4]):
        txn_queue(res, chip)
        return

    with card("cs_filter"):
        c = st.columns([1, 1.2, 1, 2.2, 0.8], vertical_alignment="bottom")
        lv = c[0].selectbox("Mức rủi ro", ["Tất cả", "CRITICAL", "HIGH"], key=f"cs_lv_{rid}")
        ty = c[1].selectbox("Loại rủi ro", ["Tất cả"] + [t for t in TYPE_ORDER if t in set(cases.risk_type)], key=f"cs_ty_{rid}")
        stv = c[2].selectbox("Trạng thái", ["Tất cả", "Chưa xử lý", "Đang xác minh", "Xử lý một phần", "Đã xử lý"],
                             key=f"cs_st_{rid}")
        q = c[3].text_input("Tìm kiếm", placeholder="Tìm theo mã hồ sơ, khách hàng, giao dịch…", key=f"cs_q_{rid}"
                            ).strip().upper()
        c[4].button("Tìm kiếm", type="primary", width="stretch", key=f"cs_go_{rid}")
    view = cases
    if chip == QCHIPS[1]:
        view = urgent
    elif chip == QCHIPS[2]:
        view = view[view.has_r7]
    if lv != "Tất cả":
        view = view[view.case_level == lv]
    if ty != "Tất cả":
        view = view[view.risk_type == ty]
    if stv != "Tất cả":
        view = view[view.case_status == stv]
    if q:
        view = view[view.customer_id.str.upper().str.contains(q, regex=False) | view.case_id.str.contains(q, regex=False)
                    | view.transaction_ids.str.upper().str.contains(q, regex=False)]
    view = view.reset_index(drop=True)
    if view.empty:
        st.warning("Không có hồ sơ nào khớp bộ lọc.", icon=":material/search_off:")
        return

    bulk = ty != "Tất cả"
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1, card("cs_list"):
        title("Danh sách hồ sơ", f"{n0(len(view))} hồ sơ · sắp theo điểm cao nhất · "
              + ("tích nhiều dòng để xử lý hàng loạt cùng loại rủi ro" if bulk
                 else "bấm một dòng để xem chi tiết · chọn một loại rủi ro để xử lý hàng loạt"))
        tbl = pd.DataFrame({"Mã hồ sơ": view.case_id, "Khách hàng": view.customer_id, "Số GD": view.n_transactions,
                            "Mức rủi ro": view.case_level, "SLA": view.sla, "Loại rủi ro": view.risk_type,
                            "Trạng thái": view.case_status, "Điểm": view.max_risk_score})
        lvl_css = {l: f"color:{LEVEL_COLOR[l]};font-weight:700" for l in LEVELS}
        sty = (tbl.style.map(lambda x: lvl_css.get(x, ""), subset=["Mức rủi ro"])
               .map(lambda x: f"color:{STATUS_COLOR.get(x, '#475467')};font-weight:600", subset=["Trạng thái"])
               .format({"Điểm": lambda x: d(x, 4)}))
        ev = grid(sty, height=520, on_select="rerun", selection_mode="multi-row" if bulk else "single-row",
                  key=f"cs_tbl_{rid}_{chip}_{ty}",
                  column_config={"Mã hồ sơ": st.column_config.TextColumn(width="small"),
                                 "Số GD": st.column_config.NumberColumn(width="small"),
                                 "SLA": st.column_config.TextColumn(width="small")})
        try:
            rows = [i for i in ev.selection.rows if i < len(view)]
        except Exception:  # noqa
            rows = []
        if bulk:
            if st.button(f"Xử lý hàng loạt {len(rows)} hồ sơ đã chọn" if rows else "Xử lý hàng loạt (tích chọn hồ sơ trước)",
                         icon=":material/done_all:", type="primary", disabled=not rows, width="stretch",
                         key=f"bulk_{rid}_{ty}"):
                bulk_dialog(res, view.iloc[rows], ty)
    i = rows[0] if rows else 0
    c = view.iloc[i]
    ids = c.transaction_ids.split(";")
    tx = df[df.transaction_id.isin(ids)].sort_values("datetime")
    first = tx.iloc[0]
    with c2, card("cs_panel"):
        st_color = STATUS_COLOR.get(c.case_status, "#475467")
        reasons = [x.strip().rstrip(".") for x in str(c.top_reason).split(";") if x.strip()]
        on = [(F.RULE_NAMES[k], int(tx[k].sum())) for k in F.RULE_W if tx[k].sum() > 0]
        H(f"""<div class="fs"><div class="fs-phead"><div class="fs-h" style="font-size:17px">Chi tiết hồ sơ</div>{badge(c.case_level)}</div>
          <div class="fs-dl">
            <span>Mã hồ sơ</span><b>{e(c.case_id)} · hạng #{int(c.case_rank)}</b>
            <span>Khách hàng</span><b>{e(c.customer_id)} · {e(first.customer_segment)} · {e(first.home_location)}</b>
            <span>Số giao dịch liên quan</span><b>{int(c.n_transactions)}</b>
            <span>Tổng số tiền</span><b>{money(c.total_amount)}</b>
            <span>Loại rủi ro</span><b>{e(c.risk_type)}</b>
            <span>Thời gian</span><b>{dt(c.start_time)} → {pd.Timestamp(c.end_time):%H:%M}</b>
            <span>Điểm cao nhất · SLA</span><b>{d(c.max_risk_score, 4)} · phản ứng trong {e(c.sla)}</b>
            <span>Trạng thái</span><b style="color:{st_color}">{e(c.case_status)}</b>
          </div>
          <div class="fs-cause"><div class="t">Nguyên nhân chính</div><ul>{"".join(f"<li>{e(x)}</li>" for x in reasons)}</ul></div>
          <div style="margin-top:8px">{"".join(f'<span class="fs-flag on">{e(nm)} · {k} GD</span>' for nm, k in on)}</div></div>""")
        with st.form(f"f_{rid}_{c.case_id}", clear_on_submit=True, border=False):
            dec = st.radio("Kết luận của chuyên viên", DECISIONS, horizontal=True)
            a, b = st.columns(2)
            who = a.text_input("Người xử lý", placeholder="Tên chuyên viên")
            note = b.text_input("Ghi chú", placeholder="VD: đã gọi xác minh")
            if st.form_submit_button("Ghi quyết định", type="primary", icon=":material/task_alt:", width="stretch"):
                save_decisions(res["ds"], [{"transaction_id": t, "case_id": c.case_id, "decision": dec, "note": note[:500],
                                            "user": (who or "chuyên viên")[:60], "time": now_vn(),
                                            "threshold": res["report"]["threshold"]} for t in ids])
                st.session_state.flash = f"Đã ghi quyết định cho {c.case_id} ({len(ids)} giao dịch)"
                st.rerun()
        sim_actions(res, c.case_id, ";".join(sorted(set(tx.device_id))), f"cs_{c.case_id}")
        st.caption("Mô hình chỉ cảnh báo và giải thích; quyết định thuộc về chuyên viên và được ghi vào nhật ký kèm phiên bản mô hình.")
    case_detail(res, c, tx)


def case_detail(res, c, tx):
    df = res["df"]
    ids = c.transaction_ids.split(";")
    cust = c.customer_id
    with card("case_tx"):
        title(f"Các giao dịch trong hồ sơ {c.case_id} ({len(tx)})")
        grid(pd.DataFrame({"Thời gian": tx.datetime.dt.strftime("%d/%m %H:%M:%S"), "Giao dịch": tx.transaction_id,
                           "Số tiền": tx.amount.map(money), "Trạng thái GD": tx.transaction_status,
                           "Kênh": tx.payment_channel, "Thiết bị": tx.device_id, "Tỉnh": tx.location,
                           "IP": tx.ip_country, "Merchant": tx.merchant_id + " · " + tx.merchant_risk_level.astype(str),
                           "Điểm": tx.risk_score, "Mức": tx.risk_level, "Lý do": tx.alert_reason}),
             column_config={"Điểm": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                            "Lý do": st.column_config.TextColumn(width="large")})
    c1, c2 = st.columns([1, 1.05], gap="medium")
    with c1, card("case_prof"):
        hist, prof = profile_of(df, cust, ids)
        title("Thói quen của khách so với hồ sơ này",
              f"Thói quen tính từ {n0(len(hist))} giao dịch khác của {e(cust)} · cái mới tô cam")
        rows = []
        for col, lab in [("device_id", "Thiết bị"), ("location", "Tỉnh/thành"), ("ip_country", "Quốc gia IP"),
                         ("payment_channel", "Kênh")]:
            usual = ", ".join(f"{e(k)} ({v})" for k, v in prof[col].items()) or "–"
            now = []
            for v in sorted(tx[col].astype(str).unique()):
                now.append(f'<span class="new">{e(v)} · mới</span>' if v not in set(hist[col].astype(str)) else e(v))
            rows.append({"k": lab, "usual": usual, "now": ", ".join(now)})
        med = hist.amount.median() if len(hist) else np.nan
        rows.append({"k": "Số tiền", "usual": f"trung vị {money(med)}" if med == med else "–",
                     "now": f"cao nhất {money(tx.amount.max())} (gấp {d(tx.amt_ratio.max(), 1)} lần mức nền)"})
        table(rows, [("k", "Chiều"), ("usual", "Thường dùng"), ("now", "Trong hồ sơ")], raw_cols=("usual", "now"))
    with c2, card("case_net"):
        fig, n_other = network(tx, cust, df)
        title("Liên kết khách hàng – thiết bị – merchant",
              "Xanh lá: thiết bị đăng ký cho khách · Đỏ: thiết bị lạ · Cam: khách hàng khác dùng chung thiết bị"
              + (f" – {n_other} khách (dấu hiệu số 8, TT 17/2024 sửa đổi bởi TT 25/2025)" if n_other else ""))
        chart(fig, f"net_{c.case_id}")
    with card("case_log"):
        title("Nhật ký quyết định và tác vụ của hồ sơ")
        lg = load_log(res["ds"])
        lg = lg[lg.case_id == c.case_id].drop_duplicates(["time", "decision", "user", "note"])
        ac = load_actions(res["ds"])
        ac = ac[ac.case_id == c.case_id]
        log = [{"time": r.time, "user": r.user, "decision": r.decision, "note": r.note} for r in lg.itertuples()] + \
              [{"time": r.time, "user": r.user, "decision": f"{r.action} (mô phỏng)", "note": r.target}
               for r in ac.itertuples()]
        log = sorted(log, key=lambda x: datetime.strptime(x["time"], "%d/%m/%Y %H:%M:%S"))
        if log:
            table(log[::-1], [("time", "Thời điểm"), ("user", "Người xử lý"), ("decision", "Kết luận / tác vụ"),
                              ("note", "Ghi chú")])
        else:
            st.caption("Chưa có quyết định hay tác vụ nào cho hồ sơ này.")


# ============================================================================= 4. KIỂM ĐỊNH
def page_eval(res):
    rep, df = res["report"], res["df"]
    if res["mode"] != "train":
        st.info("Kiểm định cần nhãn. Data Pack vừa chạy không có fraud_labels_sample.csv nên ứng dụng chỉ chấm điểm bằng "
                "mô hình học sẵn – hãy tải kèm tệp nhãn để xem kết quả kiểm định.", icon=":material/info:")
        return
    lv = res["lab_view"]
    b = rep["comparison_nested_cv"]["Gộp 3 tầng – có hồ sơ (B, chính thức)"]
    n_lab, n_fr = len(lv), int(lv.y.sum())
    ok = lambda x, t: f"mục tiêu ≥ {d(t, 2)} · {'đạt ✓' if x >= t else 'chưa đạt'}"  # noqa: E731
    kgrid([kcard(d(b["precision"]), "Precision", BLUE, f"{b['fp']} báo nhầm / {n0(n_lab - n_fr)} giao dịch thật"),
           kcard(d(b["recall"]), "Recall", GREEN, f"{b['tp']}/{n_fr} gian lận được bắt · {ok(b['recall'], .8)}"),
           kcard(d(b["f1"]), "F1-score", PURPLE, ok(b["f1"], .75)),
           kcard(d(b["pr_auc"]), "PR-AUC", LEVEL_COLOR["HIGH"], "chất lượng xếp hạng, không phụ thuộc ngưỡng")])
    H(f'<div class="fs-sub" style="margin:-6px 0 8px">{n0(n_lab)} giao dịch có nhãn ({n0(n_fr)} gian lận) · 5-fold phân tầng · '
      f'mô hình, trọng số luật và ngưỡng chỉ học trên phần huấn luyện của từng fold · seed {rep["seed"]}</div>')
    tabs = st.tabs(["Tổng quan", "3 tầng hành động", "Kiểu gian lận mới", "Thời gian thực", "Hiệu quả từng luật",
                    "Phân tích lỗi dự báo", "So sánh cấu hình", "Độ nhạy ngưỡng", "Công bằng"])
    with tabs[0]:
        eval_overview(res, b)
    with tabs[1]:
        eval_tiers(res)
    with tabs[2]:
        eval_loto(rep)
    with tabs[3]:
        eval_realtime(res)
    with tabs[4]:
        eval_rules(rep)
    with tabs[5]:
        eval_errors(rep)
    with tabs[6]:
        eval_compare(rep)
    with tabs[7]:
        eval_threshold(res)
    with tabs[8]:
        eval_fair(rep)


def eval_overview(res, b):
    rep, lv = res["report"], res["lab_view"]
    cnt = {k: v for k, v in rep["labeled_counts"].items() if k != "NORMAL"}
    A = rep["detected_by_type"]["Gộp 3 tầng – có hồ sơ (B, chính thức)"]
    B = rep["kpi"]["intervene_by_type"]
    types = sorted(cnt, key=lambda t: -A.get(t, 0) / cnt[t])
    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1, card("ev_types"):
        title("Kết quả theo nhóm gian lận", "Bắt được = bị chặn (HIGH/CRITICAL) · Can thiệp = bị chặn hoặc phải xác thực "
              "thêm (MEDIUM) · mục tiêu 0,80 cho từng nhóm")
        rows = []
        for t in types:
            r = A.get(t, 0) / cnt[t]
            good = r >= .8
            rows.append(f'<tr class="{"" if good else "bad"}"><td><b>{e(FRAUD_VN.get(t, t))}</b><br>'
                        f'<span style="font-size:11px;color:#64748B">{e(t)}</span></td><td class="num">{cnt[t]}</td>'
                        f'<td class="num">{A.get(t, 0)}</td><td class="num">{d(r, 2)}</td>'
                        f'<td class="num">{B.get(t, 0)}/{cnt[t]}</td>'
                        f'<td class="num"><span class="{"yes" if good else "no"}">{"✓" if good else "✗"}</span></td></tr>')
        tot = sum(A.get(t, 0) for t in types)
        rows.append(f'<tr class="hl"><td>Tổng</td><td class="num">{sum(cnt.values())}</td><td class="num">{tot}</td>'
                    f'<td class="num">{d(b["recall"], 2)}</td><td class="num">{sum(B.get(t, 0) for t in types)}/{sum(cnt.values())}</td>'
                    f'<td class="num"><span class="{"yes" if b["recall"] >= .8 else "no"}">{"✓" if b["recall"] >= .8 else "✗"}</span></td></tr>')
        H('<div class="fs fs-tw"><table class="fs-table"><thead><tr><th>Nhóm gian lận</th><th class="num">Số mẫu</th>'
          '<th class="num">Bắt được</th><th class="num">Recall</th><th class="num">Can thiệp</th>'
          '<th class="num">Đạt</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")
    with c2, card("ev_cm"):
        title("Ma trận nhầm lẫn", "Dự đoán ở ngưỡng cảnh báo, gộp 5 fold")
        tp, fn, fp = b["tp"], b["fn"], b["fp"]
        tn = len(lv) - int(lv.y.sum()) - fp
        z = [[tp, fn], [fp, tn]]
        fig = go.Figure(go.Heatmap(z=[[np.log10(v + 1) for v in row] for row in z], x=["Gian lận", "Hợp lệ"],
                                   y=["Gian lận", "Hợp lệ"], colorscale=[[0, "#E7EEFC"], [1, "#0B3B9B"]], showscale=False,
                                   hovertemplate="Thực tế %{y} · dự đoán %{x}<extra></extra>", xgap=3, ygap=3))
        labels = [["Bắt đúng", "Bỏ sót"], ["Báo nhầm", "Đúng là thật"]]
        for i, row in enumerate(z):
            for j, v in enumerate(row):
                dark = np.log10(v + 1) > 1.9
                fig.add_annotation(x=j, y=i, text=f"<b>{n0(v)}</b><br><span style='font-size:11px'>{labels[i][j]}</span>",
                                   showarrow=False, font=dict(size=20, color="#fff" if dark else NAVY))
        style(fig, 280, legend=False)
        fig.update_yaxes(autorange="reversed", title="Thực tế", showgrid=False)
        fig.update_xaxes(side="bottom", title="Dự đoán", showgrid=False)
        chart(fig, "ev_cm")
        weak = [t for t in types if A.get(t, 0) / cnt[t] < .8]
        if weak:
            items = "; ".join(f"{FRAUD_VN.get(t, t)} {A.get(t, 0)}/{cnt[t]} ({pct(A.get(t, 0) / cnt[t], 0)})" for t in weak)
            k = rep["kpi"]
            H(f'<div class="fs fs-warnbox"><div class="ic">!</div><div><div class="t">Điểm cần cải thiện</div>'
              f'<div class="d">Recall chung ở mức chặn là {pct(b["recall"])}, nhưng tính riêng từng nhóm thì còn dưới mục tiêu '
              f'80%: {e(items)}. Merchant rủi ro trông giống giao dịch thường; giao dịch đầu tiên của chuỗi dồn dập chưa có '
              f'quá khứ để so. Tầng xác thực thêm nâng tỉ lệ can thiệp chung lên {pct(k["intervene_recall"])}. Hướng xử lý: '
              f'thêm dữ liệu về tài khoản nhận tiền, học lại khi có nhãn mới.</div></div></div>')


def eval_tiers(res):
    rep = res["report"]
    k, ts = rep["kpi"], rep.get("time_split")
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1, card("ev_tiers"):
        title("Bộ chỉ số 3 tầng hành động",
              "Ngưỡng chặn và ngưỡng xác thực thêm đều chọn trên phần huấn luyện của từng fold rồi áp lên phần kiểm tra "
              "(ngưỡng lồng) – con số dưới đây là kết quả ngoài mẫu")
        rows = [{"m": "Recall can thiệp (chặn hoặc xác thực thêm)", "v": f'{pct(k["intervene_recall"])} ({k["intervene_tp"]}/{k["n_fraud"]})',
                 "c": f'{k["stepup_legit"]} khách thật ({pct(k["stepup_legit_rate"], 2)}) phải xác thực thêm'},
                {"m": "Giá trị tiền gian lận bị chặn", "v": pct(k["value_recall_hard"]),
                 "c": f'{pct(k["value_recall_intervene"])} nếu tính cả xác thực thêm; ca sót lớn nhất '
                      f'{money_short(k["missed_value_top"][0]) if k["missed_value_top"] else "–"}'},
                {"m": "Recall theo vụ tấn công", "v": f'{pct(k["case_recall"])} ({k["attacks_detected"]}/{k["n_attacks"]} vụ)',
                 "c": "vụ = các giao dịch gian lận của cùng khách cách nhau ≤ 30 phút; tính từ cảnh báo tại thời điểm giao dịch"},
                {"m": "Recall chặn ngay tại thời điểm giao dịch", "v": f'{pct(k["hard_recall"])} ({k["hard_tp"]}/{k["n_fraud"]})',
                 "c": f'{k["hard_fp"]} báo nhầm / {n0(k["n_legit"])} giao dịch thật'}]
        if ts:
            rows.append({"m": f"Kiểm tra theo thời gian (học {ts['train_months']}, kiểm tra {ts['test_months']})",
                         "v": f'can thiệp {pct(ts["intervene_recall"])} · chặn {pct(ts["recall"])}',
                         "c": f'{pct(ts["stepup_legit_rate"], 2)} khách thật phải xác thực thêm · {ts["fp"]} báo nhầm'})
        table(rows, [("m", "Chỉ số"), ("v", "Kết quả"), ("c", "Ghi chú")], num_cols=("v",))
        st.caption(f"Ngưỡng xác thực thêm đặt để Recall can thiệp trên dữ liệu học ≥ {pct(k['target_intervene'], 0)} "
                   f"(biên an toàn để ngoài mẫu ≥ 90%), nhưng không quá {pct(k['stepup_cap'], 0)} khách thật phải xác thực "
                   f"thêm. Ngưỡng chọn ở 5 fold: {', '.join(d(x, 3) for x in k['med_folds'])} · ngưỡng dùng chung: "
                   f"{d(rep['medium_threshold'], 4)}.")
    with c2, card("ev_tiers_type"):
        title("Theo nhóm gian lận", "Số ca bị chặn và số ca được can thiệp (chặn hoặc xác thực thêm)")
        cnt = {t: v for t, v in rep["labeled_counts"].items() if t not in ("NORMAL",)}
        tt = sorted(cnt, key=lambda t: -cnt[t])
        fig = go.Figure()
        fig.add_bar(y=[FRAUD_VN.get(t, t) for t in tt], x=[k["hard_by_type"].get(t, 0) / cnt[t] for t in tt],
                    orientation="h", name="Bị chặn", marker_color=LEVEL_COLOR["CRITICAL"],
                    text=[f'{k["hard_by_type"].get(t, 0)}/{cnt[t]}' for t in tt], textposition="inside")
        fig.add_bar(y=[FRAUD_VN.get(t, t) for t in tt], x=[k["intervene_by_type"].get(t, 0) / cnt[t] for t in tt],
                    orientation="h", name="Can thiệp", marker_color=LEVEL_COLOR["MEDIUM"],
                    text=[f'{k["intervene_by_type"].get(t, 0)}/{cnt[t]}' for t in tt], textposition="inside")
        style(fig, 330, barmode="group")
        fig.update_xaxes(tickformat=".0%", range=[0, 1.05])
        fig.add_vline(x=0.8, line=dict(color=NAVY, dash="dash", width=1))
        chart(fig, "ev_tier_type")
    with card("ev_tiers_note"):
        title("Tầng xác thực thêm chặn được gì, không chặn được gì")
        H('<div class="fs fs-sub" style="font-size:13.5px;color:#1F2937;line-height:1.6">Yêu cầu xác thực thêm (OTP/sinh '
          'trắc học) ngăn được kẻ gian chiếm đoạt tài khoản, dùng thông tin thẻ đánh cắp hoặc giao dịch từ thiết bị lạ. '
          'Với lừa đảo thao túng tâm lý – dụ chính chủ tài khoản tự chuyển tiền và tự xác thực – bước này không tự chặn '
          'được; cần kết hợp cảnh báo tài khoản nhận tiền (hướng phát triển khi có dữ liệu luồng tiền nhận). Quyết định '
          '2345/QĐ-NHNN bắt buộc sinh trắc học theo ngưỡng tiền (trên 10 triệu/lần, 20 triệu/ngày) và khi giao dịch lần đầu '
          'trên thiết bị mới; tầng xác thực thêm theo điểm rủi ro là biện pháp quản trị rủi ro bổ sung của ngân hàng.</div>')


def eval_loto(rep):
    lo = rep.get("loto")
    if not lo:
        st.caption("Không có dữ liệu.")
        return
    tot = lo[-1]
    with card("ev_loto"):
        title("Thí nghiệm ẩn từng kiểu gian lận",
              "Coi mỗi kiểu là một thủ đoạn mô hình chưa từng gặp: bỏ toàn bộ nhãn của kiểu đó khỏi dữ liệu học và tắt luật "
              "tương ứng, học lại hồi quy logistic và trọng số luật trên phần còn lại; Isolation Forest giữ nguyên (không dùng "
              "nhãn); giữ nguyên ngưỡng hiện hành; đếm số ca của kiểu bị ẩn mà hệ thống vẫn can thiệp")
        rows = [{"t": FRAUD_VN.get(o["type"], o["type"]) if o["type"] != "TỔNG" else "Tổng", "r": o["rules_off"],
                 "n": o["n"], "h": f'{o["high"]} ({pct(o["high"] / o["n"], 0)})',
                 "m": f'{o["medium_plus"]} ({pct(o["medium_plus"] / o["n"], 0)})', "q": o["queue"]} for o in lo]
        table(rows, [("t", "Kiểu bị ẩn"), ("r", "Luật bị tắt"), ("n", "Số ca"), ("h", "Bị chặn"),
                     ("m", "Chặn hoặc xác thực thêm"), ("q", "Vào hàng đợi mẫu mới")],
              num_cols=("n", "h", "m", "q"), hl_last=True)
        H(f'<div class="fs fs-reason"><b>Kết luận:</b> với kiểu gian lận mô hình chưa từng học, hệ thống vẫn chặn '
          f'{tot["high"]}/{tot["n"]} ca và can thiệp (chặn hoặc bắt xác thực thêm) {tot["medium_plus"]}/{tot["n"]} ca '
          f'({pct(tot["medium_plus"] / tot["n"], 0)}), nhờ các dấu vết lệch thói quen chung. Điểm yếu: chuỗi giao dịch '
          f'dồn dập khi tắt luật R6. Hàng đợi nghi vấn mẫu mới (Isolation Forest, giới hạn ~{F.NOV_BUDGET} ca/kỳ) chỉ thêm '
          f'{tot["queue"]} ca – trên dữ liệu này dị biệt đơn thuần chưa phải tín hiệu mạnh.</div>')


def eval_realtime(res):
    with card("ev_rt"):
        title("Chấm điểm thời gian thực", "Kết quả kiểm tra trùng khớp giữa chấm theo luồng và chấm theo lô")
        p = st.session_state.get("parity", {}).get(res["rid"])
        if p:
            parity_view(p)
        else:
            st.caption("Chưa chạy kiểm tra trùng khớp. Vào mục Luồng thời gian thực để chạy (khoảng 5–10 giây).")
            st.button("Mở Luồng thời gian thực", icon=":material/bolt:", key="ev_go_live",
                      on_click=lambda: st.session_state.update(nav="live"))


def eval_rules(rep):
    pr = rep.get("per_rule")
    if not pr:
        st.caption("Không có dữ liệu.")
        return
    with card("perrule"):
        title("Hiệu quả từng luật",
              "Đúng = luật bật và giao dịch đúng là gian lận (trên nhãn). Trọng số = cận dưới tin cậy 95% của tỉ lệ đúng – "
              "ít mẫu thì trọng số thấp hơn tỉ lệ đúng. Cột “PR-AUC giảm khi bỏ”: mức giảm chất lượng xếp hạng nếu bỏ luật")
        order = sorted(pr, key=lambda k: (-round(pr[k]["pr_auc_drop"], 3), -pr[k]["precision"]))
        table([{"r": F.RULE_NAMES[k], "b": pr[k].get("basis", ""), "f": f'{pr[k]["tp"]}/{pr[k]["fired_labeled"]}',
                "p": pct(pr[k]["precision"], 0), "w": d(pr[k]["weight"], 2), "a": n0(pr[k]["fired_all"]),
                "dr": d(max(0.0, pr[k]["pr_auc_drop"]), 3), "n": pr[k].get("note", "")} for k in order],
              [("r", "Luật"), ("b", "Căn cứ"), ("f", "Đúng / bật (nhãn)"), ("p", "Tỉ lệ đúng"), ("w", "Trọng số"),
               ("a", "Bật trên toàn bộ"), ("dr", "PR-AUC giảm khi bỏ"), ("n", "Đánh giá")],
              num_cols=("f", "p", "w", "a", "dr"))


def eval_errors(rep):
    cols = {"transaction_id": "Giao dịch", "fraud_type": "Mẫu", "amt_ratio": "Gấp mức nền", "location": "Tỉnh",
            "home_location": "Nơi ở", "ip_country": "IP", "merchant_risk_level": "Merchant", "risk_oof": "Điểm"}
    c1, c2 = st.columns(2, gap="medium")
    for col, key, ttl, sub in [(c1, "false_negatives", "Gian lận bị bỏ sót", "Điểm dưới ngưỡng – chủ yếu merchant rủi ro với "
                                "số tiền không lệch và giao dịch đầu tiên của chuỗi dồn dập (chưa có quá khứ để so)"),
                               (c2, "false_positives", "Báo động giả", "Giao dịch bình thường nhưng số tiền rất lệch mức nền "
                                "hoặc ở merchant rủi ro")]:
        with col, card(key):
            items = rep.get(key, [])
            title(f"{ttl} ({len(items)})", sub)
            if items:
                x = pd.DataFrame(items)[list(cols)].rename(columns=cols)
                x["Mẫu"] = x["Mẫu"].fillna("NORMAL").map(lambda t: FRAUD_VN.get(t, "Bình thường" if t == "NORMAL" else t))
                grid(x, height=420, column_config={"Gấp mức nền": st.column_config.NumberColumn(format="%.2f×"),
                                                   "Điểm": st.column_config.NumberColumn(format="%.3f")})


def eval_compare(rep):
    with card("cmp"):
        title("So sánh với các cấu hình đơn giản hơn", "Cùng 5 fold, cùng cách chọn ngưỡng – chỉ khác cấu hình mô hình")
        comp = rep["comparison_nested_cv"]
        names = list(comp)
        label = {"Chỉ Rule": "Chỉ hệ luật", "Chỉ Isolation Forest": "Chỉ Isolation Forest",
                 "Chỉ Logistic Regression": "Chỉ hồi quy logistic",
                 "Gộp 3 tầng – không hồ sơ (A)": "Gộp 3 lớp, không hồ sơ hành vi",
                 "Gộp 3 tầng – có hồ sơ (B, chính thức)": "Fraud Shield (gộp 3 lớp)"}
        rows = [{"name": label.get(k, k), "tp": v["tp"], "fn": v["fn"], "fp": v["fp"], "p": d(v["precision"]),
                 "r": d(v["recall"]), "f1": d(v["f1"]), "auc": d(v["pr_auc"])} for k, v in comp.items()]
        ts = rep.get("time_split")
        if ts:
            rows.append({"name": f"Theo thời gian: học {ts['train_months']}, kiểm tra {ts['test_months']}",
                         "tp": f"{ts['fraud_test'] - ts['fn']}/{ts['fraud_test']}", "fn": ts["fn"], "fp": ts["fp"],
                         "p": d(ts["precision"]), "r": d(ts["recall"]), "f1": d(ts["f1"]), "auc": d(ts["pr_auc"])})
        best = {c: int(np.argmax([comp[k][m] for k in names])) for c, m in
                [("p", "precision"), ("r", "recall"), ("f1", "f1"), ("auc", "pr_auc")]}
        table(rows, [("name", "Cấu hình"), ("tp", "Bắt đúng"), ("fn", "Bỏ sót"), ("fp", "Báo nhầm"), ("p", "Precision"),
                     ("r", "Recall"), ("f1", "F1"), ("auc", "PR-AUC")],
              num_cols=("tp", "fn", "fp", "p", "r", "f1", "auc"), hl_row=len(comp) - 1, best=best)
        mi = rep["model_info"]
        if mi.get("n_labels") == 2000 and mi.get("n_fraud") == 128:
            st.caption("Mọi đặc trưng chỉ dùng dữ liệu quá khứ. Bản cũ có dùng dữ liệu tương lai cho F1 0,923 – đã bỏ, "
                       "không dùng làm kết quả.")
    with card("topk"):
        title("Danh sách ưu tiên (top-k)", "Nếu chỉ đủ người xem k giao dịch điểm cao nhất")
        table([{"k": n0(t["k"]), "f": t["fraud"], "p": d(t["precision"], 2), "r": d(t["recall"])} for t in rep["topk"]],
              [("k", "Xem k giao dịch đầu"), ("f", "Gian lận thật"), ("p", "Precision@k"), ("r", "Recall@k")],
              num_cols=("k", "f", "p", "r"))


def eval_threshold(res):
    rep, df, lv = res["report"], res["df"], res["lab_view"]
    with card("thr"):
        title("Thử thay đổi ngưỡng cảnh báo",
              "Kéo thanh trượt để thấy đánh đổi giữa bắt được nhiều gian lận và báo nhầm. Điểm dùng ở đây là điểm "
              "out-of-fold tại một ngưỡng cố định nên có thể chênh nhẹ so với bảng so sánh (ngưỡng lồng).")
        key = f"thr_{res['rid']}"
        cost = rep["cost_threshold"]["threshold"]
        if key not in st.session_state:
            st.session_state[key] = float(rep["threshold"])
        b1, b2, b3, _ = st.columns([1, 1.15, 1.3, 0.6])
        b1.button(f"Ngưỡng chặn · {d(rep['threshold'], 4)}", width="stretch", key=f"pf_{res['rid']}",
                  on_click=lambda: st.session_state.update({key: float(rep["threshold"])}))
        b2.button(f"Ngưỡng xác thực thêm · {d(rep['medium_threshold'], 4)}", width="stretch", key=f"pm_{res['rid']}",
                  on_click=lambda: st.session_state.update({key: float(rep["medium_threshold"])}))
        b3.button(f"Theo chi phí (bỏ sót = 10 × báo nhầm) · {d(cost, 4)}", width="stretch", key=f"pc_{res['rid']}",
                  on_click=lambda: st.session_state.update({key: float(cost)}))
        t = st.slider("Ngưỡng", 0.05, 0.99, step=0.0005, format="%.4f", key=key)
        y, s = lv.y.values, lv.score.values
        pred = s >= t
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F1 = 2 * P * R / max(1e-9, P + R)
        kgrid([kcard(d(P), "Precision", BLUE), kcard(d(R), "Recall", GREEN), kcard(d(F1), "F1-score", PURPLE),
               kcard(f"{fn} / {fp}", "Bỏ sót / báo nhầm", LEVEL_COLOR["HIGH"],
                     f"{n0((df.risk_score >= t).sum())} cảnh báo trên {n0(len(df))} giao dịch")])
        pr, rc, th = precision_recall_curve(y, s)
        fig = go.Figure()
        fig.add_scatter(x=rc, y=pr, mode="lines", line=dict(color=BLUE, width=2.5, shape="hv"), name="Đường PR",
                        hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>")
        fig.add_scatter(x=[R], y=[P], mode="markers", marker=dict(size=13, color=LEVEL_COLOR["HIGH"],
                                                                  line=dict(color="#fff", width=2)),
                        name=f"Ngưỡng {d(t, 4)}", hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>")
        style(fig, 330, xaxis=dict(title="Recall", range=[0, 1.02]), yaxis=dict(title="Precision", range=[0, 1.04]))
        chart(fig, "pr")


def eval_fair(rep):
    c1, c2 = st.columns(2, gap="medium")
    with c1, card("fair"):
        title("Công bằng theo phân khúc khách hàng", "Phân khúc và nơi ở không dùng làm đặc trưng; bảng này để theo dõi thiên lệch")
        seg = pd.Series(rep["alert_rate_by_segment"]); fpr = pd.Series(rep["fpr_by_segment_labeled"]).reindex(seg.index)
        fig = go.Figure()
        fig.add_bar(y=seg.index, x=seg.values, orientation="h", name="Tỉ lệ cảnh báo (toàn bộ)", marker_color=BLUE,
                    hovertemplate="%{y}<br>%{x:.1%}<extra></extra>")
        fig.add_bar(y=fpr.index, x=fpr.values, orientation="h", name="Tỉ lệ báo nhầm (có nhãn)",
                    marker_color=LEVEL_COLOR["HIGH"], hovertemplate="%{y}<br>%{x:.2%}<extra></extra>")
        style(fig, 300, barmode="group")
        fig.update_xaxes(tickformat=".1%")
        chart(fig, "fair")
    with c2, card("geo"):
        title("Tỉ lệ cảnh báo theo tỉnh/thành nơi ở của khách")
        geo = pd.Series(rep["alert_rate_by_home_location"]).sort_values()
        fig = go.Figure(go.Bar(y=geo.index, x=geo.values, orientation="h", marker_color=BLUE, text=[pct(v) for v in geo.values],
                               textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>%{x:.1%}<extra></extra>"))
        style(fig, 300, legend=False)
        fig.update_xaxes(tickformat=".0%", range=[0, geo.max() * 1.22])
        chart(fig, "geo")


# ============================================================================= LUỒNG THỜI GIAN THỰC
# Trung tâm dùng chung cho mọi phiên trên máy chủ: điện thoại khách (kể cả mở trên máy thật qua link ?view=khach)
# gửi giao dịch vào đây, màn hình giám sát đọc ra mỗi giây. Một bảng trạng thái, một luồng, một khóa.
PROV_VN = {"Hanoi": "Hà Nội", "Ho Chi Minh City": "TP. Hồ Chí Minh", "Da Nang": "Đà Nẵng", "Hai Phong": "Hải Phòng",
           "Can Tho": "Cần Thơ", "Thanh Hoa": "Thanh Hóa", "Nghe An": "Nghệ An", "Hung Yen": "Hưng Yên",
           "Bac Ninh": "Bắc Ninh", "Quang Ninh": "Quảng Ninh"}
IP_VN = {"VN": "Việt Nam", "SG": "Singapore", "TH": "Thái Lan", "US": "Mỹ", "JP": "Nhật Bản", "KR": "Hàn Quốc",
         "MY": "Malaysia"}
BG_SPEEDS = {"Tạm dừng": 0.0, "1 GD/giây": 1.0, "3 GD/giây": 3.0}
ST_PENDING = {"CRITICAL": "Tạm giữ · chờ giám sát viên", "HIGH": "Tạm giữ · chờ giám sát viên",
              "MEDIUM": "Chờ khách xác thực OTP", "LOW": "Thành công"}
DONE_LIVE = {"block": "Đã chặn giao dịch", "cancel": "Đã huỷ · hoàn tiền cho khách", "ok": "Đã xác thực · cho qua"}
LIVE_CSS = """<style>
.lv-strip{display:flex;align-items:center;gap:18px;flex-wrap:wrap;background:#0B2552;color:#fff;border-radius:14px;
padding:12px 18px;margin-bottom:14px}
.lv-on{display:inline-flex;align-items:center;gap:8px;font-weight:800;letter-spacing:.06em;font-size:13px}
.lv-on i{width:11px;height:11px;border-radius:50%;background:#2FD36B;box-shadow:0 0 0 0 rgba(47,211,107,.7);animation:lvp 1.4s infinite}
.lv-on.off i{background:#98A2B3;animation:none}
@keyframes lvp{0%{box-shadow:0 0 0 0 rgba(47,211,107,.7)}70%{box-shadow:0 0 0 10px rgba(47,211,107,0)}100%{box-shadow:0 0 0 0 rgba(47,211,107,0)}}
.lv-clock{font-variant-numeric:tabular-nums;font-size:13px;color:#C9D6EA}.lv-clock b{color:#fff;font-size:15px}
.lv-cnt{display:flex;gap:22px;margin-left:auto;flex-wrap:wrap}
.lv-cnt div{text-align:right;line-height:1.15}.lv-cnt b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.lv-cnt span{font-size:11.5px;color:#9FB3D1}
.lv-feed{display:flex;flex-direction:column;gap:8px}
.lv-it{display:grid;grid-template-columns:62px 1fr auto;gap:10px;align-items:center;background:#fff;border:1px solid #E3E8F0;
border-left:5px solid var(--c);border-radius:10px;padding:8px 12px;font-size:13px}
.lv-it.new{animation:lvin .9s ease-out}
@keyframes lvin{0%{transform:translateY(-8px);background:#FFF8E1;opacity:.2}100%{transform:none;background:#fff;opacity:1}}
.lv-it .t{color:#64748B;font-variant-numeric:tabular-nums;font-size:12px}
.lv-it .m{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lv-it .m b{color:#0B2552}.lv-it .r{text-align:right;white-space:nowrap}
.lv-it .act{display:block;font-size:11.5px;color:#475467;margin-top:2px}
.lv-src{display:inline-block;background:#E8EEFC;color:#1E3A8A;font-size:10.5px;font-weight:700;border-radius:6px;padding:1px 6px;margin-left:6px}
.lv-alert{border:1px solid #FFD6D6;border-left:5px solid var(--c);background:#FFF8F8;border-radius:10px;padding:10px 12px;margin-bottom:4px}
.lv-alert .h{display:flex;justify-content:space-between;gap:8px;font-size:13.5px}
.lv-alert .d{font-size:12.5px;color:#334155;margin-top:4px;line-height:1.45}
.lv-sla{font-weight:800;font-variant-numeric:tabular-nums}
[class*="st-key-card_phone"]{max-width:400px;margin:0 auto;border:10px solid #0F172A!important;border-radius:34px!important;
padding:18px 16px 14px!important;box-shadow:0 18px 40px rgba(15,23,42,.18)!important}
.ph-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.ph-top .b{font-weight:800;color:#0B2552;font-size:17px}.ph-top .s{font-size:11.5px;color:#64748B}
.ph-res{border-radius:14px;padding:14px;margin-top:8px;text-align:center}
.ph-res .ic{font-size:30px;line-height:1}.ph-res .t{font-weight:800;font-size:16px;margin-top:6px}
.ph-res .d{font-size:12.5px;color:#334155;margin-top:4px;line-height:1.45}
</style>"""


@st.cache_resource(show_spinner=False)
def live_hub(key):
    return {"lock": threading.Lock(), "ready": False}


def hub_key(res):
    return f'{res["ds"]}|{res["model"].name}|{res.get("version_label", "")}|app-only'


def hub_get(res):
    """Luồng thời gian thực CHỈ nhận giao dịch từ ứng dụng khách FS Pay. Dữ liệu BTC không được phát lên luồng:
    toàn bộ lịch sử chỉ dùng để dựng hồ sơ thói quen của khách (bảng trạng thái) và huấn luyện mô hình."""
    hub = live_hub(hub_key(res))
    m = res["model"]
    if not hub["ready"]:
        with hub["lock"]:
            if not hub["ready"]:
                T = res["tables"]
                hist = RT.stream_order(T["transactions.csv"])
                sc = RT.OnlineScorer(m, T["customers.csv"], T["merchants.csv"], T["devices.csv"])
                sc.warm(hist.to_dict("records"))
                hub.update(sc=sc, model=m, rows=[], i=0, events=[], by_id={}, seq=0,
                           clock=pd.Timestamp(hist.datetime.iloc[-1]), speed=0.0, last_tick=time.time(),
                           n_hist=len(hist), labels={}, merch=T["merchants.csv"], cust=T["customers.csv"], hist=hist,
                           ready=True)
    return hub


def _push(hub, t, src):
    r = hub["sc"].score(t)
    hub["seq"] += 1
    mm = hub["sc"].merch.get(t["merchant_id"], {})
    ev = {"id": hub["seq"], "src": src, "tid": t["transaction_id"], "dt": pd.Timestamp(t["datetime"]),
          "cust": t["customer_id"], "amount": float(t["amount"]), "merchant": t["merchant_id"],
          "cat": mm.get("merchant_category", ""), "loc": t["location"], "ip": t["ip_country"], "dev": t["device_id"],
          "tx_status": t["transaction_status"], "score": r["risk_score"], "level": r["risk_level"],
          "reason": r["alert_reason"], "ms": r["latency_ms"], "wall": time.time(), "nov": r["novelty"],
          "status": ST_PENDING[r["risk_level"]], "y": hub["labels"].get(t["transaction_id"]),
          "recv": datetime.now(VN_TZ)}
    if src == "Dữ liệu BTC" and r["risk_level"] == "MEDIUM":
        ev["status"] = "Đã yêu cầu xác thực thêm"
    if t["transaction_status"] != "SUCCESS" and r["risk_level"] in ("LOW", "MEDIUM"):
        ev["status"] = "Thất bại (sai OTP)"
    hub["clock"] = max(hub["clock"], ev["dt"])
    hub["events"].append(ev)
    hub["by_id"][ev["id"]] = ev
    if len(hub["events"]) > 3000:
        for x in hub["events"][:500]:
            hub["by_id"].pop(x["id"], None)
        del hub["events"][:500]
    return ev


def hub_tick(hub):
    """Phát giao dịch nền từ dữ liệu BTC theo tốc độ đã chọn (theo thời gian thật, dùng chung cho mọi người xem)."""
    with hub["lock"]:
        now = time.time()
        if hub["speed"] <= 0 or hub["i"] >= len(hub["rows"]):
            hub["last_tick"] = now
            return
        due = int((now - hub["last_tick"]) * hub["speed"])
        if due <= 0:
            return
        for _ in range(min(due, 20)):
            if hub["i"] >= len(hub["rows"]):
                break
            _push(hub, hub["rows"][hub["i"]], "Dữ liệu BTC")
            hub["i"] += 1
        hub["last_tick"] = now


def hub_action(hub, ev_id, act, ds):
    with hub["lock"]:
        ev = hub["by_id"].get(ev_id)
        if ev is None or ev["status"] in DONE_LIVE.values():
            return
        ev["status"] = DONE_LIVE[act]
        ev["handled_wall"] = time.time()
    save_action(ds, "LIVE", ev["tid"], DONE_LIVE[act])


def pending_events(hub):
    with hub["lock"]:
        evs = [e for e in hub["events"] if e["status"].startswith("Tạm giữ")
               or (e["status"].startswith("Chờ khách") and e["src"] != "Dữ liệu BTC")]
    return sorted(evs, key=lambda e: (e["src"] == "Dữ liệu BTC", -e["wall"]))


def sla_left(ev):
    s = SLA_S.get(ev["level"], 600)
    left = s - (time.time() - ev["wall"])
    mm, ss = divmod(int(abs(left)), 60)
    hh, mm = divmod(mm, 60)
    txt = f"{hh}:{mm:02d}:{ss:02d}" if hh else f"{mm}:{ss:02d}"
    col = "#2F9E44" if left > s * 0.4 else "#F59F00" if left > 0 else "#E03131"
    return ("còn " if left > 0 else "quá hạn ") + txt, col


def pending_panel(hub, res, kp, limit=5):
    evs = pending_events(hub)
    if not evs:
        st.caption("Không có giao dịch nào đang chờ quyết định.")
        return 0
    for ev in evs[:limit]:
        sla, col = sla_left(ev)
        src = ""
        H(f'<div class="fs lv-alert" style="--c:{LEVEL_COLOR[ev["level"]]}"><div class="h"><span>{badge(ev["level"])} '
          f'<b>{e(ev["tid"])}</b> · {e(who(ev))} · <b>{money(ev["amount"])}</b>{src}</span>'
          f'<span class="lv-sla" style="color:{col}">{sla}</span></div>'
          f'<div class="d">{e(ev["reason"])}</div></div>')
        c = st.columns(3)
        c[0].button("Chặn", key=f"{kp}_b_{ev['id']}", icon=":material/block:", type="primary", width="stretch",
                    on_click=hub_action, args=(hub, ev["id"], "block", res["ds"]))
        c[1].button("Huỷ", key=f"{kp}_c_{ev['id']}", icon=":material/cancel:", width="stretch",
                    on_click=hub_action, args=(hub, ev["id"], "cancel", res["ds"]))
        c[2].button("Xác thực", key=f"{kp}_o_{ev['id']}", icon=":material/verified:", width="stretch",
                    on_click=hub_action, args=(hub, ev["id"], "ok", res["ds"]))
    if len(evs) > limit:
        st.caption(f"Còn {len(evs) - limit} giao dịch đang chờ khác.")
    return len(evs)


def queue_live_box(hub, res):
    if pending_events(hub):
        with card("q_live"):
            title("Cảnh báo mới từ luồng thời gian thực",
                  "Giao dịch đang bị tạm giữ chờ quyết định · Chặn = từ chối giao dịch · Huỷ = huỷ và hoàn tiền · "
                  "Xác thực = đã xác minh với khách, cho qua")
            pending_panel(hub, res, "q", limit=4)


def parity_view(p):
    ok = p["n_score_mismatch"] == 0 and p["n_level_mismatch"] == 0
    H('<div class="fs fs-kgrid" style="margin:4px 0 6px">' + "".join([
        kcard(n0(p["n"]), "Giao dịch chạy qua luồng", BLUE, "từng giao dịch một, đúng thứ tự thời gian"),
        kcard(n0(p["n_score_mismatch"]), "Giao dịch lệch điểm", GREEN if ok else LEVEL_COLOR["CRITICAL"],
              f"chênh lớn nhất {p['max_abs_diff']:.1e} · lệch mức: {p['n_level_mismatch']}",
              "Trùng 100%" if ok else "Có lệch", "#E6F4EA" if ok else "#FFE3E3"),
        kcard(f"{d(p['latency_mean_ms'], 2)} ms", "Thời gian chấm 1 giao dịch", PURPLE,
              f"p95 {d(p['latency_p95_ms'], 2)} ms · p99 {d(p['latency_p99_ms'], 2)} ms"),
        kcard(f"{d(p['total_s'], 1)} s", "Tổng thời gian chạy", TEAL,
              f"đặc trưng lệch lớn nhất {p['max_feature_diff']:.1e}")]) + "</div>")
    H(f'<div class="fs fs-reason"><b>Ý nghĩa:</b> bộ chấm luồng chỉ biết các giao dịch đã xảy ra trước đó (bảng trạng thái '
      f'trong bộ nhớ). Điểm của nó trùng với bản chấm theo lô cho {"toàn bộ" if ok else "phần lớn"} {n0(p["n"])} giao dịch – '
      f'bằng chứng mô hình không dùng dữ liệu tương lai.</div>')


# ---------------------------------------------------------------- màn hình giám sát
def monitor_body(res):
    hub = hub_get(res)
    hub_tick(hub)
    with hub["lock"]:
        evs = list(hub["events"])
    lv = pd.Series([x["level"] for x in evs]).value_counts() if evs else pd.Series(dtype=int)
    ms = [x["ms"] for x in evs[-200:]] or [0.0]
    n_pend = len(pending_events(hub))
    H(f'<div class="fs lv-strip"><span class="lv-on"><i></i>ĐANG HOẠT ĐỘNG</span>'
      f'<span class="lv-clock">Bây giờ<br><b>{datetime.now(VN_TZ):%d/%m/%Y %H:%M:%S}</b></span><div class="lv-cnt">'
      f'<div><b>{n0(len(evs))}</b><span>đã chấm</span></div>'
      f'<div><b style="color:#FF8787">{n0(lv.get("CRITICAL", 0) + lv.get("HIGH", 0))}</b><span>tạm giữ</span></div>'
      f'<div><b style="color:#FFD43B">{n0(lv.get("MEDIUM", 0))}</b><span>xác thực thêm</span></div>'
      f'<div><b style="color:#8CE99A">{n0(lv.get("LOW", 0))}</b><span>cho qua</span></div>'
      f'<div><b>{d(float(np.mean(ms)), 2)}</b><span>ms / giao dịch</span></div>'
      f'<div><b style="color:#FFA8A8">{n0(n_pend)}</b><span>chờ quyết định</span></div></div></div>')
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1, card("mon_chart"):
        title("Điểm rủi ro của các giao dịch vừa đến",
              f"Mỗi điểm là một giao dịch từ ứng dụng khách · trên đường đỏ: tạm giữ · giữa hai đường: xác thực thêm")
        last = evs[-90:]
        fig = go.Figure()
        if last:
            fig.add_scatter(x=[x["id"] for x in last], y=[x["score"] for x in last], mode="lines",
                            line=dict(color="#CBD5E1", width=1), hoverinfo="skip")
            for lvl in LEVELS:
                pts = [x for x in last if x["level"] == lvl]
                if pts:
                    fig.add_scatter(x=[x["id"] for x in pts], y=[x["score"] for x in pts], mode="markers", name=lvl,
                                    marker=dict(color=LEVEL_COLOR[lvl], size=14, symbol="circle",
                                                line=dict(color="#fff", width=1)),
                                    hovertext=[f'{x["tid"]} · {x["cust"]} · {money(x["amount"])}' for x in pts],
                                    hovertemplate="%{hovertext}<br>điểm %{y:.3f}<extra></extra>")
        m = hub["model"]
        fig.add_hline(y=m.thr, line=dict(color=LEVEL_COLOR["CRITICAL"], dash="dash", width=1.5))
        fig.add_hline(y=m.med, line=dict(color="#A06A00", dash="dot", width=1.5))
        style(fig, 300, legend=False)
        fig.update_yaxes(range=[0, 1.05], dtick=0.2)
        fig.update_xaxes(showticklabels=False, title=None)
        chart(fig, "mon_fig")
    with c2, card("mon_pending"):
        title("Cần xử lý ngay", "Chặn = từ chối · Huỷ = huỷ và hoàn tiền · Xác thực = đã xác minh với khách, cho qua · "
              "SLA: CRITICAL 5 phút, HIGH 2 giờ")
        pending_panel(hub, res, "mon", limit=3)
    with card("mon_feed"):
        title("Giao dịch đang đến", "Mới nhất ở trên")
        now = time.time()
        items = []
        for x in evs[-10:][::-1]:
            src = ""
            items.append(f'<div class="lv-it {"new" if now - x["wall"] < 2.5 else ""}" style="--c:{LEVEL_COLOR[x["level"]]}">'
                         f'<span class="t">{x.get("recv", x["dt"]):%H:%M:%S}</span><span class="m"><b>{e(who(x))}</b> → {e(x["merchant"])} '
                         f'{e(x["cat"])} · {e(PROV_VN.get(x["loc"], x["loc"]))} · IP {e(x["ip"])}{src}</span>'
                         f'<span class="r"><b>{money(x["amount"])}</b> {badge(x["level"])}'
                         f'<span class="act">{e(x["status"])}</span></span></div>')
        H('<div class="fs lv-feed">' + ("".join(items) or '<div class="fs-sub">Đang chờ giao dịch từ ứng dụng khách FS Pay…</div>') + "</div>")


def customer_url():
    try:
        u = st.context.url
        if u:
            return u.split("?")[0].rstrip("/") + "/?view=khach"
    except Exception:  # noqa
        pass
    return "?view=khach"


def page_live(res):
    H(re.sub(r"\s*\n\s*", " ", LIVE_CSS))
    hub = hub_get(res)
    m = hub["model"]
    url = customer_url()
    with card("live_head"):
        c = st.columns([3.2, 1.1], vertical_alignment="center")
        c[0].markdown(f"**Trung tâm giám sát thời gian thực** · chỉ nhận giao dịch phát sinh từ ứng dụng khách FS Pay, "
                      f"chấm từng giao dịch ngay khi đến · ngưỡng tạm giữ {d(m.thr, 2)}, xác thực thêm {d(m.med, 2)}")
        c[1].link_button("Mở ứng dụng khách FS Pay", url, icon=":material/smartphone:", type="primary", width="stretch")
        st.caption(f"Lịch sử {n0(hub['n_hist'])} giao dịch BTC không được phát lên luồng – chỉ dùng để dựng hồ sơ thói quen "
                   f"của khách và huấn luyện mô hình. Gửi link sang điện thoại: {url}")
    st.fragment(monitor_body, run_every=1.0)(res)
    with st.expander("Bằng chứng kỹ thuật: kiểm tra trùng khớp chấm theo luồng và chấm theo lô", icon=":material/fact_check:"):
        par = st.session_state.setdefault("parity", {})
        st.caption(f"Cho toàn bộ {n0(len(res['df']))} giao dịch đi qua bộ chấm luồng từng giao dịch một và đối chiếu với chấm "
                   "theo lô bằng cùng một mô hình đóng băng (khoảng 10 giây).")
        if st.button("Chạy kiểm tra trùng khớp", icon=":material/fact_check:", key="btn_parity"):
            bar = st.progress(0.0, text="Đang chạy từng giao dịch…")
            par[res["rid"]] = RT.parity_check(res["model"], res["tables"],
                                              progress=lambda x: bar.progress(min(1.0, x), text="Đang chạy từng giao dịch…"))
            bar.empty()
        if res["rid"] in par:
            parity_view(par[res["rid"]])


# ---------------------------------------------------------------- ứng dụng khách FS Pay (trang riêng ?view=khach)
CAT_VN = {"DIGITAL_SERVICE": "Dịch vụ số", "DINING": "Ăn uống", "EDUCATION": "Giáo dục", "ELECTRONICS": "Điện máy",
          "ENTERTAINMENT": "Giải trí", "FASHION": "Thời trang", "FUEL": "Xăng dầu", "GAMING": "Game",
          "GROCERY": "Siêu thị", "HEALTHCARE": "Y tế", "HOTEL": "Khách sạn", "MONEY_TRANSFER": "Chuyển tiền",
          "ONLINE_MARKETPLACE": "Sàn TMĐT", "OTHER": "Khác", "TRAVEL": "Du lịch"}
CAT_ICON = {"DIGITAL_SERVICE": "📶", "DINING": "🍜", "EDUCATION": "🎓", "ELECTRONICS": "💻", "ENTERTAINMENT": "🎬",
            "FASHION": "🛍️", "FUEL": "⛽", "GAMING": "🎮", "GROCERY": "🛒", "HEALTHCARE": "🏥", "HOTEL": "🏨",
            "MONEY_TRANSFER": "💸", "ONLINE_MARKETPLACE": "📦", "OTHER": "🧾", "TRAVEL": "✈️"}
QUICK = [("💸", "Chuyển tiền", ["MONEY_TRANSFER"]), ("🧾", "Hoá đơn", ["DIGITAL_SERVICE", "EDUCATION", "HEALTHCARE"]),
         ("🛍️", "Mua sắm", ["FASHION", "ELECTRONICS", "ONLINE_MARKETPLACE", "GROCERY"]),
         ("📱", "Nạp ĐT", ["DIGITAL_SERVICE"]), ("✈️", "Du lịch", ["TRAVEL", "HOTEL"]),
         ("🎮", "Giải trí", ["GAMING", "ENTERTAINMENT"]), ("🍜", "Ăn uống", ["DINING"]), ("⋯", "Tất cả", None)]
# Tên minh hoạ (hư cấu) cho khách demo – dữ liệu BTC chỉ có mã khách đã ẩn danh
DEMO_NAMES = ["Nguyễn Minh Anh", "Trần Quốc Bảo", "Lê Thu Hà", "Phạm Đức Huy", "Hoàng Ngọc Lan", "Vũ Thanh Tùng",
              "Đặng Mai Phương", "Bùi Gia Khánh", "Đỗ Hải Yến", "Ngô Tuấn Kiệt", "Dương Thảo Vy", "Lý Hoàng Nam"]


def cust_name(cid, custs):
    return DEMO_NAMES[custs.index(cid) % len(DEMO_NAMES)] if cid in custs else cid


def who(ev):
    return f'{ev["name"]} ({ev["cust"]})' if ev.get("name") else ev["cust"]


DEVICES = ["📱 Máy quen", "🆕 Máy lạ, vừa đăng nhập", "⚠️ Máy đã root / giả lập"]
CHANNELS = {"MOBILE_BANKING": "App ngân hàng", "E_WALLET": "Ví điện tử", "CARD": "Thẻ", "QR": "Quét QR",
            "BANK_TRANSFER": "Chuyển khoản"}
OTP_MODES = ["Nhập đúng", "Nhập sai 2 lần trước"]
GAPS = {"Vài giờ trước": 3 * 3600, "3 phút trước": 180}
PRESETS = {
    "🛒 Mua sắm bình thường": dict(desc="Máy quen, ở nhà, số tiền như mọi khi", dev=0, home=True, ip="VN", otp=0,
                                   mult=1.0, cat=["FASHION", "GROCERY", "ONLINE_MARKETPLACE"], mrisk="LOW"),
    "🌏 Máy lạ ở nước ngoài": dict(desc="Máy mới, IP Singapore, gấp 5 lần mức thường", dev=1, home=False, ip="SG", otp=0,
                                  mult=5.0, cat=["ELECTRONICS", "ONLINE_MARKETPLACE"], mrisk="LOW"),
    "🔑 Thử sai OTP liên tục": dict(desc="Máy lạ, nhập sai OTP 2 lần rồi đúng trong 1 phút", dev=1, home=True, ip="VN",
                                    otp=1, mult=3.0, cat=["GAMING"], mrisk="LOW"),
    "💰 Chuyển gấp 10 lần": dict(desc="Merchant rủi ro cao, số tiền bất thường", dev=0, home=True, ip="VN", otp=0,
                                mult=10.0, cat=["MONEY_TRANSFER", "GAMING", "HOTEL"], mrisk="HIGH"),
}

PHONE_CSS = """<style>
section[data-testid="stSidebar"],header[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1180px!important;padding-top:1.4rem!important}
.stApp{background:linear-gradient(150deg,#E6EDFB 0%,#F4F7FC 55%,#EEF2F8 100%)!important}
.st-key-fsphone{width:380px!important;max-width:100%;margin:0 auto;min-height:790px;background:#F5F7FB;border:12px solid #0B1220;
border-radius:52px;padding:0 0 14px!important;box-shadow:0 30px 60px rgba(15,23,42,.25);overflow:hidden;gap:.55rem!important}
.st-key-fsphone [data-testid="stVerticalBlock"]{gap:.5rem}
.st-key-fsphone .stElementContainer,.st-key-fsphone [data-testid="stHorizontalBlock"]{padding:0 14px}
.st-key-fsphone .stElementContainer:has(.ph-full){padding:0}
.st-key-fsphone [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"]{padding:0}
.ph-sb{height:44px;display:flex;justify-content:space-between;align-items:flex-end;padding:0 28px 6px;font-size:13px;font-weight:600;position:relative}
.ph-sb:after{content:"";position:absolute;top:9px;left:50%;transform:translateX(-50%);width:104px;height:28px;background:#0B1220;border-radius:18px}
.ph-blue{background:linear-gradient(135deg,#123A8C,#1E56C8 60%,#3B7BF0);color:#fff}
.ph-hero{padding:6px 20px 64px}.ph-hero .hi{font-size:12.5px;opacity:.85}.ph-hero .nm{font-size:16.5px;font-weight:700}
.ph-hero .row{display:flex;justify-content:space-between;align-items:center}
.ph-hero .bell{width:38px;height:38px;border-radius:50%;background:rgba(255,255,255,.2);display:grid;place-items:center}
.ph-bal{margin:-54px 14px 0;background:#fff;border-radius:20px;padding:14px 16px;box-shadow:0 10px 24px rgba(30,86,200,.15);position:relative}
.ph-bal .l{font-size:12px;color:#64748B}.ph-bal .v{font-size:25px;font-weight:800;letter-spacing:-.02em;margin:3px 0 2px}
.ph-bal .a{font-size:11.5px;color:#64748B;display:flex;justify-content:space-between}.ph-bal .a b{color:#1E56C8}
.ph-sec{display:flex;justify-content:space-between;font-size:14px;font-weight:700;margin-top:4px}
.ph-sec span{font-size:12px;color:#1E56C8;font-weight:600}
.ph-tx{background:#fff;border-radius:14px;padding:10px 12px;display:flex;align-items:center;gap:10px;font-size:12.5px;margin-top:6px}
.ph-tx i{width:34px;height:34px;border-radius:11px;background:#EEF3FF;display:grid;place-items:center;font-style:normal;font-size:16px}
.ph-tx .m{flex:1;min-width:0}.ph-tx .m b{display:block;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ph-tx .m small{color:#64748B;font-size:11px}.ph-tx .x{font-weight:700;white-space:nowrap}
.ph-nav{display:flex;justify-content:space-around;align-items:center;background:#fff;border-top:1px solid #E6EAF2;
padding:8px 4px 6px;margin-top:8px;font-size:10.5px;color:#94A3B8;text-align:center}
.ph-nav b{display:block;font-size:18px;font-weight:400}.ph-nav .on{color:#1E56C8;font-weight:700}
.ph-nav .qr{width:48px;height:48px;border-radius:16px;background:#1E56C8;color:#fff;display:grid;place-items:center;font-size:20px;
margin-top:-22px;box-shadow:0 8px 18px rgba(30,86,200,.35)}
.ph-hdr{font-weight:700;font-size:16px;padding-top:2px}
.ph-card{background:#fff;border-radius:16px;padding:12px 14px;font-size:13px}
.ph-card .r{display:flex;justify-content:space-between;padding:4px 0}.ph-card .r span{color:#64748B}
.ph-warn{background:#FFF8E1;border-radius:14px;padding:10px 12px;font-size:12.5px;color:#7A5200;line-height:1.5}
.ph-otp{display:flex;gap:7px;justify-content:center;margin:4px 0}
.ph-otp b{width:38px;height:46px;border-radius:12px;background:#fff;border:1.5px solid #CBD5E1;display:grid;place-items:center;font-size:20px}
.ph-res{text-align:center;padding:26px 6px 6px}.ph-res .ic{width:88px;height:88px;border-radius:50%;display:grid;place-items:center;
font-size:42px;margin:0 auto 14px}.ph-res h3{font-size:19px;margin:0 0 6px;color:#0F172A}
.ph-res p{font-size:13px;color:#475569;line-height:1.5;margin:0}
.st-key-fsphone [class*="st-key-qa_"] button{height:76px;border-radius:16px;border:none;background:#fff;
box-shadow:0 2px 8px rgba(15,23,42,.05);padding:4px 2px;font-size:11px;line-height:1.2;white-space:normal}
.st-key-fsphone [class*="st-key-qa_"] button p{font-size:11px!important;white-space:normal!important;overflow:visible!important;
text-overflow:clip!important;line-height:1.25!important}
.st-key-fsphone [class*="st-key-qa_"] button div{overflow:visible!important;white-space:normal!important}
.st-key-fsphone [class*="st-key-chip_"] button{border-radius:20px;background:#EEF3FF;color:#1E56C8;border:none;
font-size:11.5px;min-height:30px;padding:2px 6px}
.st-key-fspanel{background:#fff;border-radius:22px;border:1px solid #E3E8F0;box-shadow:0 14px 34px rgba(15,23,42,.08);
padding:20px 22px!important}
.pn-h{font-size:18px;font-weight:800;display:flex;align-items:center;gap:10px;color:#0F172A}
.pn-h em{font-style:normal;font-size:11px;background:#FFF3BF;color:#8A6100;border-radius:8px;padding:3px 8px;font-weight:700}
.pn-sub{font-size:12.5px;color:#64748B;margin:2px 0 4px}
.pn-g{font-size:11.5px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.05em;margin-top:6px}
.st-key-fspanel [class*="st-key-sc_"] button{min-height:70px;border-radius:14px;white-space:normal;text-align:left;justify-content:flex-start}
.pn-log div{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:7px 0;border-bottom:1px dashed #E6EAF2;font-size:12.5px}
.pn-bd{font-size:10.5px;font-weight:800;border-radius:6px;padding:3px 7px;color:#fff;white-space:nowrap}
@media (max-width:640px){.st-key-fsphone{border-width:0;border-radius:0;box-shadow:none;width:100%!important}}
</style>"""


@st.cache_data(show_spinner=False)
def demo_customers(_hist, ds):
    h = _hist
    if not len(h):
        return []
    cnt = h.customer_id.value_counts()
    return cnt[cnt >= 8].index[:12].tolist()


def cust_profile(hub, cid):
    k = hub["sc"].cust.get(cid, {})
    h = hub["hist"][hub["hist"].customer_id == cid]
    return {"home": k.get("home_location", "Hanoi"), "avg": float(k.get("avg_transaction_amount", 1e6) or 1e6),
            "dev": h.device_id.mode().iloc[0] if len(h) else f"DEV-{cid}",
            "chan": h.payment_channel.mode().iloc[0] if len(h) else "MOBILE_BANKING",
            "seg": k.get("customer_segment", ""), "hist": h}


def _ph_reset_cust():
    st.session_state.pop("ph_newdev", None)
    st.session_state.pop("ph_loc", None)
    st.session_state.pop("ph_chan", None)


def _ph_defaults(hub, custs):
    ss = st.session_state
    ss.setdefault("ph_cust", custs[0])
    prof = cust_profile(hub, ss.ph_cust)
    ss.setdefault("ph_screen", "home")
    ss.setdefault("ph_dev", DEVICES[0])
    ss.setdefault("ph_loc", prof["home"])
    ss.setdefault("ph_ip", "VN")
    ss.setdefault("ph_chan", prof["chan"] if prof["chan"] in CHANNELS else "MOBILE_BANKING")
    ss.setdefault("ph_otp", OTP_MODES[0])
    ss.setdefault("ph_gap", list(GAPS)[0])
    ss.setdefault("ph_amt", int(round(prof["avg"] / 1000) * 1000))
    ss.setdefault("ph_cat", None)
    ss.setdefault("ph_mer", hub["merch"].merchant_id.iloc[0])
    ss.setdefault("ph_note", "")
    ss.setdefault("ph_mine", [])
    return prof


def _ph_go(screen, cat=None, hub=None):
    st.session_state.ph_screen = screen
    if screen == "pay":
        st.session_state.ph_cat = cat
        if hub is not None:
            ms = hub["merch"]
            pool = ms if not cat else ms[ms.merchant_category.isin(cat)]
            if len(pool) and st.session_state.get("ph_mer") not in set(pool.merchant_id):
                st.session_state.ph_mer = pool.merchant_id.iloc[0]


def _ph_preset(name, hub):
    p = PRESETS[name]
    ss = st.session_state
    prof = cust_profile(hub, ss.ph_cust)
    ss.ph_dev = DEVICES[p["dev"]]
    ss.pop("ph_newdev", None)
    other = [x for x in PROV_VN if x != prof["home"]]
    ss.ph_loc = prof["home"] if p["home"] else other[sum(map(ord, ss.ph_cust)) % len(other)]
    ss.ph_ip = p["ip"]
    ss.ph_otp = OTP_MODES[p["otp"]]
    ss.ph_gap = list(GAPS)[0]
    ss.ph_chan = prof["chan"] if prof["chan"] in CHANNELS else "MOBILE_BANKING"
    ss.ph_amt = int(round(prof["avg"] * p["mult"] / 1000) * 1000)
    ms = hub["merch"]
    pool = ms[ms.merchant_category.isin(p["cat"]) & (ms.merchant_risk_level == p["mrisk"])]
    if not len(pool):
        pool = ms[ms.merchant_risk_level == p["mrisk"]]
    ss.ph_cat = p["cat"]
    ss.ph_mer = pool.merchant_id.iloc[0]
    ss.ph_screen = "pay"


def _ph_submit(hub):
    ss = st.session_state
    ss.ph_dev = ss.get("ph_dev") or DEVICES[0]
    ss.ph_otp = ss.get("ph_otp") or OTP_MODES[0]
    cid = ss.ph_cust
    prof = cust_profile(hub, cid)
    with hub["lock"]:
        if ss.ph_dev == DEVICES[0]:
            dev = prof["dev"]
        else:
            dev = ss.get("ph_newdev")
            if not dev or not dev.startswith("DEV-ROOT" if ss.ph_dev == DEVICES[2] else "DEV-LA"):
                rooted = ss.ph_dev == DEVICES[2]
                dev = f"{'DEV-ROOT' if rooted else 'DEV-LA'}-{hub['seq'] + 1:04d}"
                hub["sc"].dev[dev] = {"first_seen_date": hub["clock"].normalize(), "registered_customer_id": None,
                                      "device_risk_level": "HIGH" if rooted else "MEDIUM",
                                      "rooted_or_jailbroken": rooted, "emulator_flag": False}
                ss.ph_newdev = dev
        statuses = ["FAILED", "FAILED", "SUCCESS"] if ss.ph_otp == OTP_MODES[1] else ["SUCCESS"]
        # Thời điểm giao dịch: sau đồng hồ luồng, và cách giao dịch trước của chính khách đúng khoảng đã chọn
        # (mặc định vài giờ – để các lần bấm thử liên tiếp không bị coi là "dồn dập trong 5 phút").
        last = hub["sc"].C.get(cid, {}).get("last")
        base = hub["clock"] + pd.Timedelta(seconds=15)
        if last is not None:
            base = max(base, pd.Timestamp("2000-01-01") + pd.Timedelta(seconds=last + GAPS.get(ss.get("ph_gap"), 3 * 3600)))
        ids = []
        for k, sts in enumerate(statuses):
            t0 = base + pd.Timedelta(seconds=20 * k)
            hub["clock"] = max(hub["clock"], t0)
            t = {"transaction_id": f"LIVE-{hub['seq'] + 1:04d}", "customer_id": cid, "merchant_id": ss.ph_mer,
                 "datetime": t0, "amount": float(ss.ph_amt), "currency": "VND", "payment_channel": ss.ph_chan,
                 "device_id": dev, "location": ss.ph_loc, "transaction_status": sts, "ip_country": ss.ph_ip,
                 "is_card_present": False, "authentication_method": "BIOMETRIC" if ss.ph_dev == DEVICES[0] else "OTP"}
            ev = _push(hub, t, "Điện thoại khách")
            ev["note"] = ss.get("ph_note", "")
            ev["name"] = ss.get("ph_name")
            ids.append(ev["id"])
    ss.ph_mine = (ss.get("ph_mine", []) + ids)[-30:]
    ss.ph_last = ids[-1]
    lvl = hub["by_id"][ids[-1]]["level"]
    ss.ph_screen = "otp" if lvl == "MEDIUM" else "result"


def _ph_otp(hub, ev_id, ok):
    with hub["lock"]:
        ev = hub["by_id"].get(ev_id)
        if ev and ev["status"].startswith("Chờ khách"):
            ev["status"] = "Thành công sau xác thực OTP" if ok else "Khách báo không phải mình · đã huỷ"
    st.session_state.ph_screen = "result"


def _ph_report(hub, ev_id):
    with hub["lock"]:
        ev = hub["by_id"].get(ev_id)
        if ev and (ev["status"].startswith("Tạm giữ") or ev["status"].startswith("Chờ khách")):
            ev["status"] = "Khách báo không phải mình · đã huỷ"


def _status_bar(blue=False):
    H(f'<div class="fs ph-full ph-sb {"ph-blue" if blue else ""}"><span>{datetime.now(VN_TZ):%H:%M}</span>'
      '<span>●●● ᯤ ▮</span></div>')


def ph_home(hub, prof):
    ss = st.session_state
    mine = [hub["by_id"][i] for i in ss.ph_mine if i in hub["by_id"]]
    spent = sum(x["amount"] for x in mine if x["status"].startswith("Thành công") or x["status"].startswith("Đã xác thực"))
    _status_bar(True)
    H(f'<div class="fs ph-full ph-blue ph-hero"><div class="row"><div><div class="hi">Xin chào,</div>'
      f'<div class="nm">{e(ss.get("ph_name") or ss.ph_cust)}</div></div><div class="bell">🔔</div></div></div>'
      f'<div class="fs ph-full"><div class="ph-bal"><div class="l">Tài khoản thanh toán</div>'
      f'<div class="v">{money(25_430_000 - spent)}</div><div class="a"><span>STK {e(ss.ph_cust[-4:])} 2026 01</span>'
      f'<b>Ẩn số dư</b></div></div></div>')
    for row in (QUICK[:4], QUICK[4:]):
        cols = st.columns(4, gap="small")
        for c, (ic, lab, cat) in zip(cols, row):
            c.button(f"{ic}  \n{lab}", key=f"qa_{lab}", width="stretch", on_click=_ph_go, args=("pay", cat, hub))
    rows = []
    for x in mine[::-1][:2]:
        ok = x["status"].startswith("Thành công") or x["status"].startswith("Đã xác thực")
        rows.append((CAT_ICON.get(x["cat"], "🧾"), x["merchant"], f'{CAT_VN.get(x["cat"], "")} · {x["dt"]:%H:%M}',
                     f'{"−" if ok else ""}{money(x["amount"])}', "#0F172A" if ok else "#E8590C",
                     "" if ok else x["status"].split(" · ")[0]))
    h = prof["hist"].tail(4).iloc[::-1]
    mm = hub["sc"].merch
    for t in h.itertuples():
        if len(rows) >= 3:
            break
        cat = mm.get(t.merchant_id, {}).get("merchant_category", "")
        rows.append((CAT_ICON.get(cat, "🧾"), t.merchant_id, f"{CAT_VN.get(cat, '')} · {pd.Timestamp(t.datetime):%d/%m}",
                     f"−{money(t.amount)}", "#0F172A", ""))
    H('<div class="fs"><div class="ph-sec">Giao dịch gần đây <span>Xem tất cả</span></div>' + "".join(
        f'<div class="ph-tx"><i>{ic}</i><div class="m"><b>{e(m)}</b><small>{e(sub)}'
        f'{" · " + e(st_) if st_ else ""}</small></div><div class="x" style="color:{col}">{a}</div></div>'
        for ic, m, sub, a, col, st_ in rows) + "</div>")
    H('<div class="fs ph-full ph-nav"><div class="on"><b>⌂</b>Trang chủ</div><div><b>☰</b>Lịch sử</div>'
      '<div><div class="qr">⌗</div></div><div><b>🔔</b>Thông báo</div><div><b>☺</b>Cá nhân</div></div>')


def ph_pay(hub, prof):
    ss = st.session_state
    _status_bar()
    c = st.columns([1, 3.2], vertical_alignment="center")
    c[0].button("‹", key="ph_back", on_click=_ph_go, args=("home",))
    c[1].markdown('<div class="fs ph-hdr">Thanh toán</div>', unsafe_allow_html=True)
    ms = hub["merch"]
    pool = ms if not ss.ph_cat else ms[ms.merchant_category.isin(ss.ph_cat)]
    if ss.ph_mer not in set(pool.merchant_id):
        pool = ms
    info = ms.set_index("merchant_id")
    st.selectbox("Thanh toán cho", pool.merchant_id.tolist(), key="ph_mer",
                 format_func=lambda m: f"{CAT_ICON.get(info.merchant_category.get(m), '🧾')} {m} · "
                                       f"{CAT_VN.get(info.merchant_category.get(m), '')}")
    st.markdown(f'<div class="fs" style="font-size:12px;color:#64748B">Từ tài khoản STK {e(ss.ph_cust[-4:])} 2026 01</div>',
                unsafe_allow_html=True)
    st.number_input("Số tiền (VND)", min_value=10000, step=50000, key="ph_amt", format="%d")
    cols = st.columns(4, gap="small")
    for cc, v in zip(cols, [500_000, 1_000_000, 2_000_000, 5_000_000]):
        cc.button(f"{v // 1000:,}k".replace(",", "."), key=f"chip_{v}", width="stretch",
                  on_click=lambda v=v: st.session_state.update(ph_amt=v))
    st.text_input("Nội dung", key="ph_note", placeholder="VD: thanh toán đơn hàng")
    st.button("Tiếp tục", type="primary", width="stretch", key="ph_pay", on_click=_ph_submit, args=(hub,))


def ph_otp(hub):
    ev = hub["by_id"].get(st.session_state.get("ph_last"))
    _status_bar()
    st.markdown('<div class="fs ph-hdr">Xác nhận giao dịch</div>', unsafe_allow_html=True)
    if not ev:
        st.button("Về trang chủ", on_click=_ph_go, args=("home",), key="ph_home0")
        return
    H(f'<div class="fs ph-card"><div class="r"><span>Người nhận</span><b>{e(ev["merchant"])}</b></div>'
      f'<div class="r"><span>Số tiền</span><b>{money(ev["amount"])}</b></div><div class="r"><span>Phí</span><b>Miễn phí</b></div></div>'
      '<div class="fs ph-warn">🔐 Giao dịch có dấu hiệu khác thói quen của bạn. Vui lòng nhập mã OTP vừa gửi tới số điện '
      'thoại đăng ký.</div><div class="fs ph-otp"><b>4</b><b>8</b><b>2</b><b>9</b><b>1</b><b>6</b></div>')
    st.button("Xác nhận", type="primary", width="stretch", key="ph_otp_ok", on_click=_ph_otp, args=(hub, ev["id"], True))
    st.button("Không phải tôi thực hiện", width="stretch", key="ph_otp_no", on_click=_ph_otp, args=(hub, ev["id"], False))


def ph_result(hub):
    ev = hub["by_id"].get(st.session_state.get("ph_last"))
    if not ev:
        return
    s = ev["status"]
    if s.startswith("Thành công") or s.startswith("Đã xác thực"):
        bg, ic, t, p, col = "#E6F4EA", "✅", "Thanh toán thành công", "Cảm ơn bạn đã sử dụng FS Pay", "#2F9E44"
    elif s.startswith("Tạm giữ"):
        bg, ic, t, col = "#FFF4E6", "⏳", "Giao dịch đang được kiểm tra", "#E8590C"
        p = ("Để bảo vệ tài khoản, ngân hàng đang xác minh giao dịch này. Tiền chưa bị trừ. Bạn có thể nhận cuộc gọi "
             "xác minh trong ít phút.")
    elif s.startswith("Chờ khách"):
        bg, ic, t, p, col = "#FFF8E1", "🔐", "Cần xác thực thêm", "Vui lòng xác thực để hoàn tất giao dịch.", "#A06A00"
    else:
        bg, ic, t, col = "#FFE3E3", "⛔", "Giao dịch không thành công", "#C92A2A"
        p = ("Ngân hàng đã từ chối giao dịch để bảo vệ tài khoản của bạn. Nếu đây đúng là bạn, vui lòng liên hệ tổng đài "
             "hỗ trợ." if "chặn" in s.lower() or "huỷ" in s.lower() else s)
    H(f'<div class="fs ph-res"><div class="ic" style="background:{bg}">{ic}</div><h3>{t}</h3><p>{e(p)}</p></div>'
      f'<div class="fs ph-card"><div class="r"><span>Số tiền</span><b>{money(ev["amount"])}</b></div>'
      f'<div class="r"><span>Người nhận</span><b>{e(ev["merchant"])}</b></div>'
      f'<div class="r"><span>Mã giao dịch</span><b>{e(ev["tid"])}</b></div>'
      f'<div class="r"><span>Trạng thái</span><b style="color:{col}">{e(s)}</b></div></div>')
    if s.startswith("Tạm giữ"):
        st.button("Tôi không thực hiện giao dịch này", width="stretch", key=f"ph_rep_{ev['id']}",
                  on_click=_ph_report, args=(hub, ev["id"]))
    else:
        if st.button("Về trang chủ", type="primary", width="stretch", key=f"ph_home_{ev['id']}"):
            st.session_state.ph_screen = "home"
            st.rerun()


def control_panel(hub, custs, prof):
    ss = st.session_state
    with st.container(key="fspanel"):
        H('<div class="fs pn-h">🎛️ Bảng điều khiển giả lập <em>CHỈ DÙNG KHI DEMO</em></div>'
          '<div class="fs pn-sub">Đặt hoàn cảnh của giao dịch tiếp theo. Khách không nhìn thấy bảng này – đây là những thông '
          'tin ngân hàng thu được ngầm từ thiết bị và mạng.</div>')
        H('<div class="fs pn-g">Khách hàng đăng nhập</div>')
        st.selectbox("Khách hàng", custs, key="ph_cust", label_visibility="collapsed", on_change=_ph_reset_cust,
                     format_func=lambda c: f"{cust_name(c, custs)} · {c} · {PROV_VN.get(cust_profile(hub, c)['home'], '')} · chi thường "
                                           f"~{money(cust_profile(hub, c)['avg'])}")
        H('<div class="fs pn-g">Thiết bị</div>')
        st.segmented_control("Thiết bị", DEVICES, key="ph_dev", label_visibility="collapsed",
                             format_func=lambda x: x if x != DEVICES[0] else f"{x} ({prof['dev']})")
        c = st.columns(2)
        with c[0]:
            H('<div class="fs pn-g">Vị trí giao dịch</div>')
            st.selectbox("Vị trí", list(PROV_VN), key="ph_loc", label_visibility="collapsed",
                         format_func=lambda x: f"📍 {PROV_VN[x]}{' (nơi ở)' if x == prof['home'] else ''}")
        with c[1]:
            H('<div class="fs pn-g">Mạng / IP</div>')
            st.selectbox("IP", list(IP_VN), key="ph_ip", label_visibility="collapsed", format_func=lambda x: f"🌐 {IP_VN[x]}")
        c = st.columns(2)
        with c[0]:
            H('<div class="fs pn-g">Kênh thanh toán</div>')
            st.selectbox("Kênh", list(CHANNELS), key="ph_chan", label_visibility="collapsed",
                         format_func=lambda x: f"{CHANNELS[x]}{' (quen dùng)' if x == prof['chan'] else ''}")
        with c[1]:
            H('<div class="fs pn-g">Khi được hỏi OTP</div>')
            st.segmented_control("OTP", OTP_MODES, key="ph_otp", label_visibility="collapsed")
        H('<div class="fs pn-g">Giao dịch trước của khách này diễn ra</div>')
        st.segmented_control("Khoảng cách", list(GAPS), key="ph_gap", label_visibility="collapsed",
                             help="Mặc định vài giờ trước. Chọn 3 phút trước để thử tình huống giao dịch dồn dập (luật R6).")
        H('<div class="fs pn-g">Kịch bản nhanh (bấm là điền sẵn cả app lẫn bảng này)</div>')
        names = list(PRESETS)
        for i in range(0, 4, 2):
            cc = st.columns(2)
            for j, col in enumerate(cc):
                nm = names[i + j]
                col.button(f"**{nm}**  \n{PRESETS[nm]['desc']}", key=f"sc_{i + j}", width="stretch",
                           on_click=_ph_preset, args=(nm, hub))
        H('<div class="fs pn-g">Giao dịch vừa gửi → kết quả hệ thống trả về</div>')
        mine = [hub["by_id"][i] for i in ss.ph_mine if i in hub["by_id"]][::-1][:5]
        if mine:
            H('<div class="fs pn-log">' + "".join(
                f'<div><span>{x.get("recv", x["dt"]):%H:%M:%S} · {e(x["tid"])} · {money(x["amount"])} · {e(x["dev"])} · IP {e(x["ip"])}'
                f'{" · " + ("thất bại" if x["tx_status"] != "SUCCESS" else "") if x["tx_status"] != "SUCCESS" else ""}</span>'
                f'<span class="pn-bd" style="background:{LEVEL_COLOR[x["level"]]}">{x["level"]} · '
                f'{e(x["status"].split(" · ")[0].lower())}</span></div>' for x in mine) + "</div>")
        else:
            st.caption("Chưa gửi giao dịch nào.")
        st.caption("Ứng dụng thanh toán mô phỏng “FS Pay” phục vụ trình diễn Fraud Shield – không phải ứng dụng của ngân "
                   "hàng thật. Tên khách là tên minh hoạ (hư cấu) gắn với mã khách đã ẩn danh trong dữ liệu BTC. Mô hình "
                   "chấm điểm không dùng giờ trong ngày nên bảng không có mục thời điểm.")


def phone_screen(hub, prof):
    sc = st.session_state.ph_screen
    if sc == "pay":
        ph_pay(hub, prof)
    elif sc == "otp":
        ph_otp(hub)
    elif sc == "result":
        _status_bar()
        ev = hub["by_id"].get(st.session_state.get("ph_last"))
        waiting = ev is not None and ev["status"].startswith("Tạm giữ")
        st.fragment(ph_result, run_every=1.0 if waiting else None)(hub)
    else:
        ph_home(hub, prof)


def page_customer(res):
    """Ứng dụng khách FS Pay – trang riêng, mở bằng link ?view=khach (tab khác hoặc điện thoại thật)."""
    H(re.sub(r"\s*\n\s*", " ", PHONE_CSS))
    hub = hub_get(res)
    custs = demo_customers(hub["hist"], res["ds"])
    if not custs:
        st.caption("Không có khách hàng mẫu.")
        return
    prof = _ph_defaults(hub, custs)
    st.session_state.ph_name = cust_name(st.session_state.ph_cust, custs)
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        with st.container(key="fsphone"):
            phone_screen(hub, prof)
    with right:
        control_panel(hub, custs, prof)


# ============================================================================= 5. DỮ LIỆU & CHẠY MÔ HÌNH
def stepper(states):
    labels = ["Tải dữ liệu", "Kiểm tra dữ liệu", "Chạy mô hình", "Xem kết quả"]
    out = []
    for i, (lab, s) in enumerate(zip(labels, states)):
        if i:
            out.append(f'<div class="fs-sline {"done" if states[i - 1] == "done" else ""}"></div>')
        sym = "✓" if s == "done" else "!" if s == "err" else str(i + 1)
        out.append(f'<div class="fs-stp {s}"><div class="c">{sym}</div>{lab}</div>')
    H('<div class="fs fs-steps">' + "".join(out) + "</div>")


GROUP_TEXT = {"Tệp bắt buộc": "Đủ 4 tệp bắt buộc và đủ cột (transactions, customers, merchants, devices)",
              "Tệp tùy chọn": "Tệp tùy chọn: nhãn, manifest, template nộp bài",
              "Đối chiếu manifest BTC": "Khớp số dòng, số cột và SHA-256 với manifest của BTC",
              "Chất lượng dữ liệu": "Mã giao dịch không trùng, số tiền dương, thời gian đọc được",
              "Liên kết bảng": "Mọi giao dịch liên kết được với khách hàng, merchant, thiết bị",
              "Nhãn": "Nhãn khớp với giao dịch"}


def requirements_list(checks):
    order = {"LỖI": 3, "CẢNH BÁO": 2, "THIẾU": 1, "ĐẠT": 0}
    groups = {}
    for c in checks:
        g = c["group"]
        if g == "Tệp tải lên":
            continue
        groups[g] = max(groups.get(g, "ĐẠT"), c["status"], key=order.get)
    color = {"ĐẠT": ("#2F9E44", "✓"), "CẢNH BÁO": ("#F59F00", "!"), "LỖI": ("#E03131", "✕"), "THIẾU": ("#98A2B3", "–")}
    H('<div class="fs">' + "".join(
        f'<div class="fs-req"><i style="background:{color[s][0]}">{color[s][1]}</i><div>{e(GROUP_TEXT.get(g, g))}'
        f'<small>{e(g)} · {e(s.lower())}</small></div></div>' for g, s in groups.items()) + "</div>")


def page_data(res):
    failed = st.session_state.get("failed")
    stepper(["done", "err", "", ""] if failed else (["done", "done", "done", "cur"] if res is not None else ["cur", "", "", ""]))
    if failed:
        st.error("Data Pack vừa tải lên chưa đạt các kiểm tra bắt buộc nên chưa chạy mô hình – xem chi tiết bên dưới.",
                 icon=":material/error:")
    c1, c2 = st.columns([1.25, 1], gap="medium")
    with c1, card("dt_upload"):
        title("Tải Data Pack", "Tệp .zip của Ban tổ chức hoặc chọn nhiều tệp .csv cùng lúc. Bắt buộc: transactions, customers, "
              "merchants, devices. Có fraud_labels_sample thì huấn luyện và kiểm định; không có thì chấm điểm bằng mô hình học sẵn.")
        ups = st.file_uploader("Data Pack (.zip hoặc .csv)", type=["zip", "csv"], accept_multiple_files=True,
                               label_visibility="collapsed", key="uploader")
        H('<div class="fs-h" style="margin-top:14px">Yêu cầu dữ liệu</div>')
        if failed:
            requirements_list(failed["checks"])
        elif res is not None:
            requirements_list(res["checks"])
        else:
            H('<div class="fs">' + "".join(f'<div class="fs-req"><i style="background:#CBD5E1">•</i><div>{e(t)}</div></div>'
                                          for g, t in GROUP_TEXT.items() if g not in ("Tệp tùy chọn", "Nhãn")) + "</div>")
    with c2:
        with card("dt_info"):
            title("Data Pack đang xem")
            if res is not None:
                df = res["df"]
                H('<div class="fs fs-dl">' + "".join(f"<span>{k}</span><b>{v}</b>" for k, v in [
                    ("Nguồn", e(res["source"])), ("Dung lượng", f'{res.get("size", 0) / 1048576:.1f} MB'.replace(".", ",")),
                    ("Số giao dịch", n0(len(df))),
                    ("Khách hàng · merchant · thiết bị", f"{n0(df.customer_id.nunique())} · {n0(df.merchant_id.nunique())} · "
                                                         f"{n0(df.device_id.nunique())}"),
                    ("Ngày dữ liệu", f'{res["report"]["period"][0]} – {res["report"]["period"][1]}'),
                    ("Chế độ", "Huấn luyện + kiểm định" if res["mode"] == "train" else "Chỉ chấm điểm (mô hình học sẵn)"),
                    ("Thời gian chạy", f'{str(res.get("seconds", "–")).replace(".", ",")} giây'),
                    ("Phiên bản mô hình", e(res.get("version_label", "Bản gốc"))),
                    ("Trạng thái", '<span class="fs-badge b-ĐẠT">Hợp lệ</span>')]) + "</div>")
            else:
                st.caption("Chưa có dữ liệu.")
        with card("dt_run"):
            title("Chạy mô hình", "Hệ luật (R1–R7) + Hồi quy logistic + Isolation Forest (học trên giai đoạn quá khứ), "
                  "gộp điểm 0,6 – 0,3 – 0,1, xếp 3 tầng hành động và gom hồ sơ")
            if st.button("Chạy mô hình với dữ liệu tải lên" if ups else "Chạy mô hình với Data Pack BTC", type="primary",
                         icon=":material/play_arrow:", width="stretch", key="btn_run",
                         disabled=not ups and not (SAMPLE_DIR / "transactions.csv").exists()):
                if ups:
                    files, notes = F.collect_files([(u.name, u.getvalue()) for u in ups])
                    src = ", ".join(u.name for u in ups) if len(ups) <= 2 else f"{len(ups)} tệp CSV tải lên"
                    start_run(files, notes, src[:160])
                else:
                    start_run(sample_files(), [], "Data Pack BTC (có sẵn trong ứng dụng)")
            st.caption("Thời gian dự kiến: khoảng 10–20 giây cho 8.000 giao dịch. Chưa chọn tệp thì chạy lại Data Pack BTC.")

    if failed or res is not None:
        with st.expander("Kết quả kiểm tra chi tiết từng hạng mục", icon=":material/fact_check:", expanded=bool(failed)):
            checks_table(failed["checks"] if failed else res["checks"])
    if res is not None:
        downloads(res)
        retrain_section(res)
        with card("dt_admin"):
            title("Quản trị", "Kết luận của chuyên viên lưu trên máy chủ đến khi ứng dụng khởi động lại.")
            a, b = st.columns([2, 1], vertical_alignment="center")
            sure = a.checkbox("Tôi muốn xoá mọi kết luận đã ghi cho bộ dữ liệu này", key="chk_clear")
            if b.button("Xoá kết luận đã ghi", disabled=not sure, width="stretch", key="btn_clear", icon=":material/delete:"):
                clear_decisions(res["ds"])
                st.session_state.flash = "Đã xoá các kết luận đã ghi"
                st.rerun()


def retrain_section(res):
    with card("dt_retrain"):
        title("Học lại có kiểm soát (quản trị viên)",
              "Đưa kết luận của chuyên viên (Xác nhận gian lận = 1, Báo động giả = 0) vào dữ liệu học. Mô hình mới chỉ được "
              "áp dụng sau khi so sánh trước – sau trên tập kiểm tra độc lập theo thời gian, và luôn khôi phục được bản cũ.")
        ex = analyst_labels(res["ds"])
        btc = res["tables"].get("fraud_labels_sample.csv")
        new = ex[~ex.transaction_id.isin(btc.transaction_id)] if btc is not None else ex
        H('<div class="fs fs-dl">' + "".join(f"<span>{k}</span><b>{v}</b>" for k, v in [
            ("Phiên bản đang dùng", e(res.get("version_label", "Bản gốc"))),
            ("Kết luận của chuyên viên", f"{n0(len(ex))} giao dịch ({n0(int(ex.is_fraud.sum()))} gian lận, "
                                         f"{n0(int((ex.is_fraud == 0).sum()))} báo động giả)"),
            ("Nhãn mới (chưa có trong nhãn BTC)", n0(len(new)))]) + "</div>")
        H('<div class="fs fs-warnbox" style="background:#FFF9DB;border-color:#FFE066"><div class="ic" style="background:#F59F00">!'
          '</div><div><div class="t" style="color:#A06A00">Lưu ý độ lệch mẫu</div><div class="d">Chuyên viên chỉ kết luận trên '
          'các giao dịch đã bị cảnh báo, nên nhãn mới lệch về phía “điểm cao”. Vì vậy mô hình mới phải được so với mô hình cũ '
          'trên nhãn BTC của giai đoạn sau (không dùng để học) trước khi áp dụng.</div></div></div>')
        admin = st.toggle("Tôi là quản trị viên (mô phỏng phân quyền – bản trình diễn chưa có đăng nhập)", key="is_admin")
        vers = st.session_state.setdefault("versions", [])
        if not admin:
            return
        if res["mode"] != "train":
            st.caption("Cần Data Pack có nhãn BTC để kiểm tra trước – sau.")
        elif len(new) < 3:
            st.info("Cần ít nhất 3 kết luận mới của chuyên viên (ở mục Hàng đợi của tôi) để học lại.", icon=":material/info:")
        else:
            key = f"{res['rid']}|{len(new)}|{int(new.is_fraud.sum())}"
            if st.button("So sánh trước – sau", icon=":material/compare_arrows:", key="btn_cmp"):
                with st.spinner("Đang học lại và kiểm tra trên giai đoạn sau…"):
                    st.session_state.rt_cmp = (key, F.retrain_compare(res, new))
            cmp = st.session_state.get("rt_cmp")
            if cmp and cmp[0] == key and cmp[1]:
                c = cmp[1]
                b_, a_ = c["before"], c["after"]
                rows = [{"m": "Nhãn dùng để học", "b": f'{n0(b_["n_train"])} ({b_["fraud_train"]} gian lận)',
                         "a": f'{n0(a_["n_train"])} ({a_["fraud_train"]} gian lận)'},
                        {"m": "PR-AUC trên giai đoạn kiểm tra", "b": d(b_["pr_auc"]), "a": d(a_["pr_auc"])},
                        {"m": "Recall chặn", "b": d(b_["recall"]), "a": d(a_["recall"])},
                        {"m": "Báo nhầm", "b": b_["fp"], "a": a_["fp"]},
                        {"m": "Recall can thiệp", "b": d(b_["intervene_recall"]), "a": d(a_["intervene_recall"])},
                        {"m": "Khách thật phải xác thực thêm", "b": pct(b_["stepup_rate"], 2), "a": pct(a_["stepup_rate"], 2)}]
                table(rows, [("m", f"Học {c['train_period']} · kiểm tra {c['test_period']} ({c['fraud_test']} gian lận)"),
                             ("b", "Trước"), ("a", "Sau")], num_cols=("b", "a"))
                st.caption(f"Dùng {c['n_new_labels']} nhãn mới thuộc giai đoạn học; {c['n_ignored_future']} nhãn thuộc giai đoạn "
                           "kiểm tra không được dùng ở bước so sánh này để tránh rò rỉ.")
                if c["recommend"]:
                    st.success("Mô hình mới không làm giảm chất lượng trên giai đoạn kiểm tra – có thể áp dụng.",
                               icon=":material/check_circle:")
                else:
                    st.warning("Mô hình mới làm giảm chất lượng – không nên áp dụng.", icon=":material/warning:")
                sure = st.checkbox("Tôi đã xem bảng so sánh và đồng ý áp dụng", key="chk_apply")
                if st.button("Áp dụng mô hình mới", type="primary", icon=":material/published_with_changes:",
                             disabled=not (sure and c["recommend"]), key="btn_apply"):
                    with st.spinner("Đang học lại toàn bộ mô hình…"):
                        files = st.session_state.get("files") or sample_files()
                        r2 = engine(files, new.reset_index(drop=True))
                    vers.append(st.session_state.res)
                    store_result(r2, files, [], res["source"],
                                 label=f"Học lại · +{len(new)} nhãn chuyên viên · {now_vn()}")
                    st.session_state.flash = "Đã áp dụng mô hình học lại"
                    st.rerun()
        if vers:
            prev = vers[-1]
            st.caption(f"Phiên bản trước: {prev.get('version_label', 'Bản gốc')}")
            if st.button("Khôi phục phiên bản trước (Rollback)", icon=":material/undo:", key="btn_rollback"):
                st.session_state.res = vers.pop()
                st.session_state.flash = "Đã khôi phục phiên bản trước"
                st.rerun()


def downloads(res):
    files = dict(build_files(res["rid"], res))
    b = io.BytesIO()
    audit = load_log(res["ds"])
    audit.to_csv(b, index=False, encoding="utf-8-sig")
    files["audit_log.csv"] = b.getvalue()
    with card("dt_dl"):
        a, b2 = st.columns([3, 1], vertical_alignment="center")
        with a:
            title("Tải kết quả", "submission_output.csv theo đúng cột và thứ tự dòng của template BTC")
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for k, v in files.items():
                zf.writestr(k, v)
        b2.download_button("Tải tất cả (.zip)", z.getvalue(), "FraudShield_ketqua.zip", "application/zip", type="primary",
                           icon=":material/download:", width="stretch", key=f"dl_all_{res['rid']}")
        desc = [("submission_output.csv", "Tệp nộp theo template BTC (9 cột)", "text/csv"),
                ("cases.csv", "Danh sách hồ sơ cảnh báo", "text/csv"),
                ("scored_transactions_detail.csv", "Toàn bộ giao dịch kèm đặc trưng, cờ luật, điểm từng lớp", "text/csv"),
                ("kpi_report.json", "Báo cáo chỉ số kiểm định", "application/json"),
                ("data_checks.csv", "Kết quả kiểm tra dữ liệu", "text/csv"),
                ("audit_log.csv", f"Nhật ký quyết định của chuyên viên ({len(audit)} dòng)", "text/csv")]
        cols = st.columns(3, gap="small")
        for i, (nm, ds, mime) in enumerate(desc):
            with cols[i % 3]:
                st.download_button(nm, files[nm], nm, mime, icon=":material/description:", width="stretch",
                                   key=f"dl_{i}_{res['rid']}", help=f"{ds} · {n0(max(1, len(files[nm]) / 1024))} KB")


# ============================================================================= khung chính
PAGES = {"overview": page_overview, "tx": page_tx, "cases": page_cases, "eval": page_eval, "live": page_live}

if "res" not in st.session_state and "failed" not in st.session_state and "tried_sample" not in st.session_state:
    st.session_state.tried_sample = True
    auto_load_sample()
RES = st.session_state.get("res")
if st.query_params.get("view") == "khach" and RES is not None:
    page_customer(RES)
    st.stop()
cur = sidebar(RES)
if RES is None and cur != "data":
    cur = "data"
if st.session_state.get("flash"):
    st.toast(st.session_state.pop("flash"), icon=":material/check_circle:")
topbar(RES, dict((k, t) for k, _, _, t in NAV)[cur])
if cur == "data":
    page_data(RES)
elif RES is None:
    no_data()
else:
    PAGES[cur](RES)
