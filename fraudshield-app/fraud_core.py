"""
FRAUD SHIELD – LÕI MÔ HÌNH (dùng chung cho ứng dụng web và chạy dòng lệnh)
Festival Chuyển đổi số 2026 · Bảng B · Digital Payment Fraud Shield

Nguyên tắc: mọi đặc trưng của một giao dịch chỉ dùng thông tin CÓ TRƯỚC hoặc TẠI thời điểm giao dịch
(cửa sổ trượt lùi [t-5 phút, t], [t-30 phút, t]; hồ sơ hành vi chỉ từ các giao dịch trước đó).

Các lớp:
  1) Hồ sơ hành vi khách hàng (chỉ lịch sử quá khứ, không dùng nhãn)
  2) Chấm điểm 3 tầng: Rule (noisy-OR) + Logistic Regression + Isolation Forest
     risk_score = 0,60·ML + 0,30·Rule + 0,10·Anomaly
     Trọng số mỗi luật = cận dưới khoảng tin cậy Wilson 95% của tỉ lệ "luật bật thì đúng là gian lận",
     tính trên phần dữ liệu huấn luyện của từng fold.
  3) Lớp bất thường theo ngữ cảnh (v3.2, không dùng nhãn): mỗi đặc trưng rủi ro được so với phân bố của giai đoạn
     quá khứ theo đúng chiều rủi ro (số tiền cao hơn, nhịp nhanh hơn, thiết bị mới hơn…), cộng độ hiếm
     −log P(X ≥ x) của từng đặc trưng (dạng HBOS có chiều). Giao dịch lệch mạnh mà điểm tổng còn thấp được đưa
     sang tầng "xác thực thêm" – cửa này dành cho kiểu gian lận mô hình có giám sát chưa từng thấy.
  4) Ngưỡng / risk_level / alert_reason / priority_rank / gom case

Hai chế độ:
  - "train": Data Pack có fraud_labels_sample.csv -> huấn luyện + đánh giá đầy đủ
    (5-fold phân tầng, ngưỡng lồng, baseline, theo thời gian, top-k, chi phí, công bằng).
  - "score": Data Pack không có nhãn -> chấm điểm bằng mô hình đã huấn luyện sẵn
    (model/pretrained.joblib, học trên 2.000 nhãn công khai của BTC).
"""
import hashlib, io, json, math, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
VERSION = "FraudShield-v3.2 (thời gian thực · 3 tầng hành động · lớp ngữ cảnh)"
W_FUSION = (0.60, 0.30, 0.10)
# Trọng số mặc định (Wilson 95% cận dưới trên 2.000 nhãn BTC); khi có nhãn sẽ được tính lại từ dữ liệu
RULE_W = {"flag_R1_strong": .62, "flag_R1_mild": .24, "flag_R2": .84, "flag_R3": .92, "flag_R4": .30,
          "flag_R5": .82, "flag_R6": .90, "flag_R7": .77, "flag_C1": .07, "flag_C2": .02}
RULE_NAMES = {"flag_R1_strong": "R1 mạnh – số tiền ≥ 5 lần mức nền",
              "flag_R1_mild": "R1 nhẹ – số tiền 3–5 lần mức nền",
              "flag_R2": "R2 – thiết bị lạ, mới xuất hiện ≤ 30 ngày",
              "flag_R3": "R3 – IP nước ngoài và tỉnh khác nơi ở",
              "flag_R4": "R4 – merchant HIGH kèm tiền ≥ 2× hoặc tỉnh mới",
              "flag_R5": "R5 – ≥ 2 lần thất bại trong 30 phút gần nhất",
              "flag_R6": "R6 – ≥ 2 giao dịch trong 5 phút gần nhất",
              "flag_R7": "R7 – thiết bị đã dùng cho ≥ 2 khách hàng",
              "flag_C1": "C1 – thiết bị root/giả lập/rủi ro HIGH",
              "flag_C2": "C2 – không xác thực"}
RULE_BASIS = {"flag_R1_strong": "TT17 dấu hiệu 6", "flag_R1_mild": "TT17 dấu hiệu 6",
              "flag_R2": "QĐ 2345; TT17 dấu hiệu 6", "flag_R3": "TT17 dấu hiệu 6",
              "flag_R4": "NĐ 52/2024; TT17 dấu hiệu 6", "flag_R5": "TT17 dấu hiệu 6 (tần suất)",
              "flag_R6": "TT17 dấu hiệu 6 (tần suất)", "flag_R7": "TT17 sửa đổi bởi TT25/2025 – dấu hiệu 8",
              "flag_C1": "TT 50/2024, TT 77/2025", "flag_C2": "QĐ 2345"}
CASE_GAP_S = 1800
TARGET_INTERVENE = 0.91   # Recall can thiệp đặt trên dữ liệu học (biên an toàn 1 điểm để ngoài mẫu ≥ 90%)
STEPUP_CAP = 0.05         # tối đa 5% giao dịch thật phải xác thực thêm
NOV_BUDGET = 40           # hàng đợi nghi vấn mẫu mới: khoảng 40 ca cho cả kỳ
CTX_BUDGET = 0.03         # cửa ngữ cảnh: tối đa ~3% giao dịch thật bị đưa sang xác thực thêm vì lệch ngữ cảnh
CTX_GRID = (0.0, 0.01, 0.02, 0.03, 0.05)
# Lớp ngữ cảnh: (đặc trưng, chiều rủi ro). +1: càng cao càng rủi ro; -1: càng thấp càng rủi ro. Không dùng nhãn.
CTX_SPEC = [("amt_ratio", 1), ("prof_amt_z", 1), ("c_amt_over_max", 1), ("n_txn_5m", 1), ("c_n_1h", 1),
            ("c_gap_pct", 1), ("log_min_gap", -1), ("n_fail_30m", 1), ("merchant_high", 1),
            ("dev_not_registered_to_cust", 1), ("dev_age_days", -1), ("loc_mismatch", 1), ("ip_foreign", 1),
            ("auth_none", 1), ("dev_compromised", 1), ("dev_shared", 1), ("prof_new_device", 1),
            ("prof_new_location", 1), ("prof_new_ip", 1)]
CTX_FILL = {"dev_age_days": 9999.0}
ACTIONS = {"CRITICAL": "Tạm dừng giao dịch, gọi xác minh ngay", "HIGH": "Tạm dừng, chuyên viên kiểm tra trong ca",
           "MEDIUM": "Yêu cầu xác thực thêm (OTP/sinh trắc học)", "LOW": "Cho qua"}
COST_FN, COST_FP = 10, 1
F_BASE = ["amt_ratio", "dev_not_registered_to_cust", "dev_age_days", "dev_is_new", "loc_mismatch", "ip_foreign",
          "merchant_high", "auth_none", "dev_compromised", "n_txn_5m", "log_min_gap", "n_fail_30m", "n_succ_30m"]
F_PROFILE = ["prof_amt_z", "prof_new_device", "prof_new_location", "prof_new_channel", "prof_new_ip",
             "profile_dev_count"]
FEATURES = F_BASE + F_PROFILE + ["dev_shared"]
F_NOPROF = F_BASE + ["dev_shared"]
OUT_COLS = ["transaction_id", "risk_score", "risk_level", "predicted_fraud", "alert_reason", "priority_rank",
            "confidence", "model_name", "processing_time_ms"]

# Cột bắt buộc theo từng tệp (tên tệp theo Data Pack của BTC)
REQUIRED = {
    "transactions.csv": ["transaction_id", "customer_id", "merchant_id", "datetime", "amount", "payment_channel",
                         "device_id", "location", "transaction_status", "ip_country", "authentication_method"],
    "customers.csv": ["customer_id", "customer_segment", "home_location", "avg_transaction_amount"],
    "merchants.csv": ["merchant_id", "merchant_category", "merchant_risk_level"],
    "devices.csv": ["device_id", "first_seen_date", "registered_customer_id", "device_risk_level",
                    "rooted_or_jailbroken", "emulator_flag"],
}
OPTIONAL = {"fraud_labels_sample.csv": ["transaction_id", "is_fraud", "fraud_type"],
            "dataset_manifest.csv": ["file_name", "row_count", "column_count", "sha256"],
            "submission_output_template.csv": ["transaction_id"],
            "answer_key_sample.csv": ["transaction_id", "expected_is_fraud", "expected_risk_level"]}

PRETRAINED = Path(__file__).parent / "model" / "pretrained.json"


# ============================================================================ tiện ích
def vnd(x):
    return f"{int(round(x)):,}".replace(",", ".") + " VND"


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def dot(n):
    return f"{int(n):,}".replace(",", ".")


def _csv(b):
    return pd.read_csv(io.BytesIO(b), encoding="utf-8-sig")


# ============================================================================ 0. ĐỌC FILE TẢI LÊN
def collect_files(items):
    """items: list of (filename, bytes). Nhận .zip (đọc trong bộ nhớ, không giải nén ra đĩa) hoặc .csv.
    Trả về dict tên-tệp (chữ thường) -> bytes."""
    files, notes = {}, []
    for name, data in items:
        low = name.lower()
        if low.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        base = info.filename.replace("\\", "/").split("/")[-1].lower()
                        if info.is_dir() or not base.endswith(".csv") or base.startswith("._"):
                            continue
                        if info.file_size > 200 * 1024 * 1024:
                            notes.append(f"Bỏ qua {base}: quá lớn"); continue
                        files[base] = z.read(info)
            except zipfile.BadZipFile:
                notes.append(f"{name}: không phải tệp zip hợp lệ")
        elif low.endswith(".csv"):
            files[low.split("/")[-1]] = data
        else:
            notes.append(f"Bỏ qua {name}: chỉ nhận .zip hoặc .csv")
    return files, notes


def collect_dir(path):
    files = {}
    for p in Path(path).rglob("*.csv"):
        files[p.name.lower()] = p.read_bytes()
    return files


# ============================================================================ 1. KIỂM TRA DỮ LIỆU
def validate(files):
    """Kiểm tra mềm: trả về (bảng dữ liệu, danh sách kiểm tra, có chạy được không)."""
    checks, T = [], {}

    def add(group, check, status, detail=""):
        checks.append({"group": group, "check": check, "status": status, "detail": detail})

    for name, cols in REQUIRED.items():
        if name not in files:
            add("Tệp bắt buộc", name, "LỖI", "Không tìm thấy tệp này trong dữ liệu tải lên"); continue
        try:
            t = _csv(files[name])
        except Exception as e:  # noqa
            add("Tệp bắt buộc", name, "LỖI", f"Không đọc được CSV: {e}"); continue
        miss = [c for c in cols if c not in t.columns]
        if miss:
            add("Tệp bắt buộc", name, "LỖI", "Thiếu cột: " + ", ".join(miss)); continue
        T[name] = t
        add("Tệp bắt buộc", name, "ĐẠT", f"{dot(len(t))} dòng × {t.shape[1]} cột")
    for name, cols in OPTIONAL.items():
        if name not in files:
            add("Tệp tùy chọn", name, "THIẾU", "Không bắt buộc"); continue
        try:
            t = _csv(files[name])
            miss = [c for c in cols if c not in t.columns]
            if miss:
                add("Tệp tùy chọn", name, "CẢNH BÁO", "Thiếu cột: " + ", ".join(miss) + " – bỏ qua tệp"); continue
            T[name] = t
            add("Tệp tùy chọn", name, "ĐẠT", f"{dot(len(t))} dòng")
        except Exception as e:  # noqa
            add("Tệp tùy chọn", name, "CẢNH BÁO", f"Không đọc được: {e}")

    ok = all(n in T for n in REQUIRED)
    if not ok:
        return T, checks, False

    # Đối chiếu manifest của BTC (nếu có)
    if "dataset_manifest.csv" in T:
        man = T["dataset_manifest.csv"]
        for _, r in man.iterrows():
            fn = str(r.file_name).lower()
            if fn not in files:
                continue
            t = _csv(files[fn]); sha = hashlib.sha256(files[fn]).hexdigest()
            same = (len(t) == r.row_count) and (t.shape[1] == r.column_count) and (sha == r.sha256)
            add("Đối chiếu manifest BTC", fn, "ĐẠT" if same else "CẢNH BÁO",
                "Khớp số dòng, số cột và SHA-256" if same else
                f"Khác manifest (dòng {len(t)}/{r.row_count}, cột {t.shape[1]}/{r.column_count}, "
                f"SHA {'khớp' if sha == r.sha256 else 'khác'}) – vẫn chạy được")

    gd, kh, mc, tb = (T[n] for n in ["transactions.csv", "customers.csv", "merchants.csv", "devices.csv"])
    uniq = gd.transaction_id.is_unique
    add("Chất lượng dữ liệu", "transaction_id không trùng", "ĐẠT" if uniq else "LỖI",
        "" if uniq else f"{gd.transaction_id.duplicated().sum()} mã bị trùng")
    amt = pd.to_numeric(gd.amount, errors="coerce")
    bad_amt = int((amt.isna() | (amt <= 0)).sum())
    add("Chất lượng dữ liệu", "amount là số dương", "ĐẠT" if bad_amt == 0 else "LỖI",
        "" if bad_amt == 0 else f"{bad_amt} dòng không hợp lệ")
    dt = pd.to_datetime(gd.datetime, errors="coerce")
    bad_dt = int(dt.isna().sum())
    add("Chất lượng dữ liệu", "datetime đọc được", "ĐẠT" if bad_dt == 0 else "LỖI",
        f"Từ {dt.min():%d/%m/%Y} đến {dt.max():%d/%m/%Y}" if bad_dt == 0 else f"{bad_dt} dòng lỗi")
    fk_ok = True
    for k, t, nm in [("customer_id", kh, "customers"), ("merchant_id", mc, "merchants"), ("device_id", tb, "devices")]:
        miss = int((~gd[k].isin(t[k])).sum())
        fk_ok &= miss == 0
        add("Liên kết bảng", f"{k} có trong {nm}.csv", "ĐẠT" if miss == 0 else "LỖI",
            "100% giao dịch liên kết được" if miss == 0 else f"{miss} giao dịch không tìm thấy")
    for k, t, nm in [("customer_id", kh, "customers"), ("merchant_id", mc, "merchants"), ("device_id", tb, "devices")]:
        if not t[k].is_unique:
            fk_ok = False
            add("Liên kết bảng", f"{k} không trùng trong {nm}.csv", "LỖI", "Có mã trùng, ghép bảng sẽ nhân dòng")
    if "fraud_labels_sample.csv" in T:
        lb = T["fraud_labels_sample.csv"]
        inn = lb.transaction_id.isin(gd.transaction_id)
        add("Nhãn", "Nhãn khớp giao dịch", "ĐẠT" if inn.all() else "CẢNH BÁO",
            f"{dot(int(inn.sum()))} nhãn, {int(lb.is_fraud.sum())} gian lận ({num(lb.is_fraud.mean() * 100)}%); "
            f"{dot(len(gd) - int(inn.sum()))} giao dịch không nhãn chỉ được chấm điểm, không coi là bình thường")
    ok = uniq and bad_amt == 0 and bad_dt == 0 and fk_ok
    return T, checks, ok


# ============================================================================ 2. FEATURE + HỒ SƠ
def window_count(ts, w):
    return np.searchsorted(ts, ts + w, "right") - np.searchsorted(ts, ts - w, "left")


def build(gd, kh, mc, tb):
    gd = gd.copy(); tb = tb.copy()
    gd["datetime"] = pd.to_datetime(gd.datetime); gd["amount"] = pd.to_numeric(gd.amount)
    tb["first_seen_date"] = pd.to_datetime(tb.first_seen_date)
    df = gd.merge(kh, on="customer_id", how="left").merge(mc, on="merchant_id", how="left") \
           .merge(tb, on="device_id", how="left")
    assert len(df) == len(gd)
    df["amt_ratio"] = df.amount / df.avg_transaction_amount
    df["dev_not_registered_to_cust"] = (df.registered_customer_id != df.customer_id).astype(int)
    df["dev_age_days"] = (df.datetime.dt.normalize() - df.first_seen_date).dt.days.clip(lower=0)
    df["dev_is_new"] = (df.dev_age_days <= 30).astype(int)
    df["loc_mismatch"] = (df.location != df.home_location).astype(int)
    df["ip_foreign"] = (df.ip_country != "VN").astype(int)
    df["merchant_high"] = (df.merchant_risk_level == "HIGH").astype(int)
    df["auth_none"] = (df.authentication_method == "NONE").astype(int)
    tobool = lambda s: s.astype(str).str.upper().isin(["TRUE", "1", "YES"])
    df["dev_compromised"] = ((df.device_risk_level == "HIGH") | tobool(df.rooted_or_jailbroken)
                             | tobool(df.emulator_flag)).astype(int)
    df = df.sort_values(["customer_id", "datetime"]).reset_index(drop=True)
    df["ts"] = (df.datetime - pd.Timestamp("2000-01-01")).dt.total_seconds()
    n5, f30, s30, gap = [], [], [], []
    for _, g in df.groupby("customer_id", sort=False):
        ts = g.ts.values; f = (g.transaction_status != "SUCCESS").astype(int).values
        cf = np.concatenate([[0], np.cumsum(f)]); i = np.arange(len(ts))
        lo5 = np.searchsorted(ts, ts - 300, "left"); lo = np.searchsorted(ts, ts - CASE_GAP_S, "left")
        n5.append(i + 1 - lo5)                                  # [t-5 phút, t], gồm chính nó
        nf = cf[i + 1] - cf[lo]; f30.append(nf); s30.append((i + 1 - lo) - nf)   # [t-30 phút, t]
        gap.append(np.diff(ts, prepend=np.nan))                 # khoảng cách tới giao dịch TRƯỚC
    df["n_txn_5m"] = np.concatenate(n5); df["n_fail_30m"] = np.concatenate(f30)
    df["n_succ_30m"] = np.concatenate(s30)
    df["min_gap_s"] = np.concatenate(gap); df["log_min_gap"] = np.log1p(df.min_gap_s.fillna(1e7))
    # Ngữ cảnh của chính khách (v3.2): so với các giao dịch TRƯỚC đó của cùng khách
    a_over, n1h, gpct = [], [], []
    for _, g in df.groupby("customer_id", sort=False):
        ts = g.ts.values; am = g.amount.values.astype(float); k = len(ts)
        cmax = np.maximum.accumulate(am); lo1 = np.searchsorted(ts, ts - 3600, "left"); gp = np.diff(ts)
        a_ = np.zeros(k); c_ = np.full(k, 0.5)
        for i in range(k):
            if i >= 3:
                a_[i] = math.log(am[i] / cmax[i - 1])            # >0: vượt mức cao nhất từng chi
            if i >= 4:
                c_[i] = float(np.mean(gp[:i - 1] > gp[i - 1]))  # tỉ lệ khoảng cách cũ dài hơn khoảng cách lần này
        a_over.append(a_); n1h.append(np.arange(k) - lo1); gpct.append(c_)
    df["c_amt_over_max"] = np.concatenate(a_over); df["c_n_1h"] = np.concatenate(n1h).astype(float)
    df["c_gap_pct"] = np.concatenate(gpct)
    # Hồ sơ hành vi: chỉ từ các giao dịch trước đó của cùng khách
    for cot, ten in [("device_id", "device"), ("location", "location"), ("payment_channel", "channel"),
                     ("ip_country", "ip")]:
        df[f"prof_new_{ten}"] = (df.groupby(["customer_id", cot]).cumcount() == 0).astype(int)
    la = np.log(df.amount); g = la.groupby(df.customer_id); n = g.cumcount()
    m = g.transform(lambda x: x.shift().expanding().mean())
    sd = g.transform(lambda x: x.shift().expanding().std()).clip(lower=0.3)
    df["prof_amt_z"] = ((la - m) / sd).where(n >= 3, 0).fillna(0)
    df["prof_typical_amt"] = np.exp(m.fillna(la))
    df["prof_amt_dev"] = (df.prof_amt_z >= 3).astype(int)
    df["cust_n_txn"] = (n + 1).astype(int)
    df["n_same_ip_other"] = df.groupby(["customer_id", "ip_country"]).cumcount()
    df["profile_dev_count"] = df[["prof_amt_dev", "prof_new_device", "prof_new_location", "prof_new_channel",
                                  "prof_new_ip"]].sum(axis=1)
    # Số khách khác nhau đã dùng thiết bị tính tới thời điểm giao dịch (TT 17/2024 sửa đổi bởi TT 25/2025 – dấu hiệu 8)
    d = df.sort_values("datetime", kind="stable"); seen, sh = {}, []
    for dv, c in zip(d.device_id, d.customer_id):
        st_ = seen.setdefault(dv, set()); st_.add(c); sh.append(len(st_))
    df["dev_n_cust_sofar"] = pd.Series(sh, index=d.index).reindex(df.index)
    df["dev_shared"] = (df.dev_n_cust_sofar >= 2).astype(int)
    return df


def wilson_lb(k, n, z=1.96):
    """Cận dưới khoảng tin cậy Wilson 95% của tỉ lệ k/n."""
    if n == 0:
        return 0.0
    p = k / n
    return float((p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n))


def rule_weights(flags, y):
    return {c: round(wilson_lb(int(flags[c][y == 1].sum()), int(flags[c].sum())), 4) for c in RULE_W}


def noisy_or(flags, w):
    return 1 - np.prod([1 - w[c] * flags[c].values for c in RULE_W], axis=0)


def oof_rule(flags, y, folds):
    o = np.zeros(len(y))
    for tr, va in folds:
        o[va] = noisy_or(flags.iloc[va], rule_weights(flags.iloc[tr], y[tr]))
    return o


def rules(df, use_profile=True, w=None):
    f = pd.DataFrame(index=df.index)
    f["flag_R1_strong"] = df.amt_ratio >= 5
    f["flag_R1_mild"] = (df.amt_ratio >= 3) & (df.amt_ratio < 5)
    f["flag_R2"] = (df.dev_not_registered_to_cust == 1) & (df.dev_is_new == 1)
    f["flag_R3"] = (df.ip_foreign == 1) & (df.loc_mismatch == 1)
    f["flag_R4"] = (df.merchant_high == 1) & ((df.amt_ratio >= 2) | ((df.prof_new_location == 1) & use_profile))
    f["flag_R5"] = df.n_fail_30m >= 2
    f["flag_R6"] = df.n_txn_5m >= 2
    f["flag_R7"] = df.dev_shared == 1
    f["flag_C1"] = df.dev_compromised == 1
    f["flag_C2"] = df.auth_none == 1
    f = f.astype(int)
    return f, noisy_or(f, w or RULE_W)


def reason(r):
    p = []
    if r.flag_R5: p.append(f"{r.n_fail_30m} lần FAILED/DECLINED của cùng khách hàng trong 30 phút gần nhất")
    if r.flag_R6: p.append(f"Giao dịch thứ {r.n_txn_5m} của cùng khách hàng trong 5 phút gần nhất")
    if r.flag_R1_strong or r.flag_R1_mild:
        p.append(f"Số tiền {vnd(r.amount)} gấp {num(r.amt_ratio)} lần mức nền {vnd(r.avg_transaction_amount)}")
    if r.flag_R2: p.append(f"Thiết bị {r.device_id} chưa đăng ký cho khách hàng, mới xuất hiện {int(r.dev_age_days)} ngày")
    if r.flag_R3: p.append(f"Giao dịch tại {r.location} (nơi ở: {r.home_location}) với IP {r.ip_country}; "
                           f"trong {r.cust_n_txn - 1} giao dịch trước đó của khách, {r.n_same_ip_other} giao dịch dùng IP {r.ip_country}")
    if r.flag_R7: p.append(f"Thiết bị {r.device_id} đã được {int(r.dev_n_cust_sofar)} khách hàng khác nhau sử dụng")
    if r.flag_R4:
        if r.flag_R1_strong or r.flag_R1_mild:
            p.append(f"Merchant {r.merchant_id} ({r.merchant_category}) có mức rủi ro nền HIGH")
        else:
            x = f"số tiền gấp {num(r.amt_ratio)} lần mức nền" if r.amt_ratio >= 2 else f"khách lần đầu giao dịch tại {r.location}"
            p.append(f"Merchant {r.merchant_id} ({r.merchant_category}) rủi ro nền HIGH, {x}")
    prof = []
    if r.prof_amt_dev and not (r.flag_R1_strong or r.flag_R1_mild):
        prof.append(f"số tiền {vnd(r.amount)} so với mức thường {vnd(r.prof_typical_amt)}")
    if r.prof_new_device and not r.flag_R2: prof.append(f"lần đầu dùng thiết bị {r.device_id}")
    if r.prof_new_location and not (r.flag_R3 or r.flag_R4): prof.append(f"lần đầu giao dịch tại {r.location}")
    if r.prof_new_ip and not r.flag_R3: prof.append(f"lần đầu dùng IP {r.ip_country}")
    if prof: p.append("Lệch hồ sơ hành vi: " + ", ".join(prof))
    if r.flag_C1: p.append(f"Thiết bị {r.device_id} có cờ rủi ro (root/emulator/rủi ro nền HIGH)")
    if r.flag_C2: p.append("Giao dịch không xác thực (NONE)")
    if not p:
        return "Không có tín hiệu bất thường nổi bật so với hồ sơ nền."
    return "; ".join(p[:3]) + "."


# ============================================================================ 2b. LỚP NGỮ CẢNH (HBOS có chiều)
def ctx_matrix(df):
    """Ma trận đặc trưng ngữ cảnh đã nhân chiều rủi ro (càng lớn càng rủi ro)."""
    # làm tròn 9 chữ số: chấm luồng tính trung bình/độ lệch chuẩn cộng dồn, có thể lệch 1e-16 so với chấm lô
    return np.round(np.column_stack([sg * df[c].fillna(CTX_FILL.get(c, 0.0)).values.astype(float)
                                     for c, sg in CTX_SPEC]), 9)


def ctx_vector(f):
    """Như ctx_matrix nhưng cho một giao dịch (dict đặc trưng) – dùng khi chấm luồng."""
    out = []
    for c, sg in CTX_SPEC:
        v = f.get(c)
        v = CTX_FILL.get(c, 0.0) if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        out.append(sg * v)
    return np.round(np.array(out), 9)


def ctx_fit(V, ref_mask):
    """Bảng tham chiếu: giá trị đã sắp xếp của giai đoạn quá khứ + bảng độ hiếm −log((j+1)/(n+1)) dựng sẵn,
    để chấm theo lô và chấm theo luồng cho ra đúng cùng một con số."""
    refs = [np.sort(V[ref_mask, j]) for j in range(V.shape[1])]
    n = int(ref_mask.sum())
    table = np.array([-math.log((j + 1) / (n + 1)) for j in range(n + 1)])
    return {"refs": refs, "n": n, "table": table, "cols": [c for c, _ in CTX_SPEC]}


def ctx_parts(V, ref):
    """Độ hiếm của từng đặc trưng: −log P_quá khứ(X ≥ x) (làm trơn +1)."""
    V = np.atleast_2d(V)
    P = np.empty(V.shape)
    for j, r in enumerate(ref["refs"]):
        tail = ref["n"] - np.searchsorted(r, V[:, j], side="left")      # số giá trị quá khứ ≥ x
        P[:, j] = ref["table"][tail]
    return P


def ctx_score(V, ref):
    P = ctx_parts(V, ref)
    S = np.zeros(P.shape[0])
    for j in range(P.shape[1]):
        S = S + P[:, j]
    return np.round(S, 9)


CTX_TEXT = {"amt_ratio": lambda r: f"số tiền gấp {num(r['amt_ratio'])} lần mức nền",
            "prof_amt_z": lambda r: f"số tiền lệch {num(r['prof_amt_z'])} độ lệch chuẩn so với thói quen",
            "c_amt_over_max": lambda r: f"vượt mức cao nhất từng chi {num(math.exp(r['c_amt_over_max']))} lần",
            "n_txn_5m": lambda r: f"giao dịch thứ {int(r['n_txn_5m'])} trong 5 phút",
            "c_n_1h": lambda r: f"{int(r['c_n_1h'])} giao dịch khác trong 1 giờ trước đó",
            "c_gap_pct": lambda r: f"nhịp nhanh hơn {num(100 * r['c_gap_pct'], 0)}% các lần trước của khách",
            "log_min_gap": lambda r: f"chỉ cách giao dịch trước {_dur(math.expm1(r['log_min_gap']))}",
            "n_fail_30m": lambda r: f"{int(r['n_fail_30m'])} lần thất bại trong 30 phút",
            "merchant_high": lambda r: "merchant rủi ro nền HIGH",
            "dev_not_registered_to_cust": lambda r: "thiết bị không đăng ký cho khách",
            "dev_age_days": lambda r: "thiết bị mới xuất hiện",
            "loc_mismatch": lambda r: "giao dịch ngoài tỉnh nơi ở", "ip_foreign": lambda r: "IP nước ngoài",
            "auth_none": lambda r: "không xác thực", "dev_compromised": lambda r: "thiết bị có cờ rủi ro",
            "dev_shared": lambda r: "thiết bị dùng chung nhiều khách",
            "prof_new_device": lambda r: "lần đầu dùng thiết bị này",
            "prof_new_location": lambda r: "lần đầu giao dịch ở tỉnh này",
            "prof_new_ip": lambda r: "lần đầu dùng IP này"}


def _dur(sec):
    if sec < 120:
        return f"{int(round(sec))} giây"
    if sec < 7200:
        return f"{int(round(sec / 60))} phút"
    if sec < 172800:
        return f"{num(sec / 3600, 0)} giờ"
    return f"{num(sec / 86400, 0)} ngày"


def ctx_note(parts, row, k=3):
    """Câu giải thích: k đặc trưng hiếm nhất (so với giai đoạn quá khứ)."""
    order = np.argsort(-parts)[:k]
    bits = []
    for j in order:
        if parts[j] < 1.0:          # P ≥ ~37%: không đáng kể
            continue
        c = CTX_SPEC[j][0]
        try:
            bits.append(CTX_TEXT[c](row))
        except Exception:  # noqa
            bits.append(c)
    return "Lệch mạnh so với thói quen (lớp ngữ cảnh): " + ", ".join(bits) if bits else "Lệch mạnh so với thói quen (lớp ngữ cảnh)"


def ctx_door_thr(ctx, y, q=None):
    """Ngưỡng cửa ngữ cảnh: phân vị (1−q) của điểm ngữ cảnh trên giao dịch THẬT – chỉ dùng nhãn 'bình thường',
    không dùng nhãn gian lận, nên không 'học' kiểu gian lận nào."""
    q = CTX_BUDGET if q is None else q
    if q <= 0:
        return float("inf")
    return float(np.quantile(ctx[y == 0], 1 - q))


# ============================================================================ 3. MÔ HÌNH + ĐÁNH GIÁ
def best_thr(y, s):
    p, r, t = precision_recall_curve(y, s); f = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f[:-1])])


def evaluate(y, s, folds):
    pred = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        pred[va] = s[va] >= best_thr(y[tr], s[tr])
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "pr_auc": average_precision_score(y, s), "fpr": fp / int((y == 0).sum()),
            "tp": tp, "fp": fp, "fn": fn, "pred": pred}


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def oof(X, y, folds):
    o = np.zeros(len(y))
    for tr, va in folds:
        o[va] = lr().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return o


def iso_score(X):
    s = -IsolationForest(n_estimators=300, random_state=SEED).fit(X).score_samples(X)
    return pd.Series(s).rank(pct=True).values


def r4(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _metrics(y, p, s):
    return {"precision": round(precision_score(y, p, zero_division=0), 4), "recall": round(recall_score(y, p), 4),
            "f1": round(f1_score(y, p), 4), "pr_auc": round(average_precision_score(y, s), 4),
            "tp": int((p & (y == 1)).sum()), "fp": int((p & (y == 0)).sum()), "fn": int((~p & (y == 1)).sum())}


def cost_threshold(y, s, cost_fn=COST_FN, cost_fp=COST_FP):
    grid = np.unique(np.round(s, 4))
    costs = [cost_fn * ((s < g) & (y == 1)).sum() + cost_fp * ((s >= g) & (y == 0)).sum() for g in grid]
    return float(grid[int(np.argmin(costs))])


def ref_period(dt):
    """Giai đoạn tham chiếu (quá khứ) = 2/3 số tháng đầu của dữ liệu. Isolation Forest, bảng phân vị
    và mô hình dùng cho luồng thời gian thực chỉ học trên giai đoạn này."""
    dt = pd.to_datetime(dt)
    mth = dt.dt.to_period("M")
    months = sorted(mth.unique())
    if len(months) >= 3:
        cut = months[int(len(months) * 2 / 3) - 1]
        nxt = months[months.index(cut) + 1]
        return ((mth <= cut).values, f"{months[0]} → {cut}", f"{nxt} → {months[-1]}",
                pd.Timestamp(nxt.start_time))
    q = dt.quantile(2 / 3)
    return (dt <= q).values, f"{dt.min():%d/%m/%Y} → {q:%d/%m/%Y}", f"sau {q:%d/%m/%Y}", q


def fit_iso_ref(X, mask, n_ref=1001):
    """Isolation Forest học trên giai đoạn quá khứ; điểm quy ra phân vị theo bảng tham chiếu cố định
    (không xếp hạng trên giao dịch tương lai)."""
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X[mask])
    raw = np.round(-iso.score_samples(X), 12)          # làm tròn để chấm lô và chấm luồng không lệch do sai số máy
    ref = np.round(np.quantile(raw[mask], np.linspace(0, 1, n_ref)), 12)
    return iso, ref, np.searchsorted(ref, raw, side="right") / len(ref)


def med_threshold(y, s, thr, target=None, cap=None):
    """Ngưỡng tầng 'xác thực thêm': ngưỡng cao nhất giữ được Recall can thiệp ≥ target trên dữ liệu học,
    nhưng không để quá cap giao dịch thật phải xác thực thêm."""
    target = TARGET_INTERVENE if target is None else target
    cap = STEPUP_CAP if cap is None else cap
    fr = np.sort(s[y == 1])[::-1]
    if len(fr) == 0:
        return thr
    t = float(fr[min(len(fr), int(np.ceil(target * len(fr)))) - 1])
    leg = np.sort(s[y == 0])[::-1]
    floor_ = float(leg[min(len(leg) - 1, int(cap * len(leg)))]) if len(leg) else t
    return min(thr, max(t, floor_))


def attack_ids(lab_fraud):
    """Gom các giao dịch gian lận có nhãn thành 'vụ': cùng khách, cách nhau ≤ 30 phút."""
    a = lab_fraud.sort_values(["customer_id", "datetime"])
    new = (a.customer_id != a.customer_id.shift()) | (a.datetime.diff().dt.total_seconds() > CASE_GAP_S)
    return pd.Series(new.cumsum().values, index=a.index).reindex(lab_fraud.index)


def evaluate_tiers(y, s, folds, lab, ctx=None, q=None):
    """Đánh giá 3 tầng hành động bằng ngưỡng lồng (ngưỡng chọn trên phần huấn luyện của từng fold).
    ctx: điểm lớp ngữ cảnh – cửa vào tầng xác thực thêm, ngưỡng chọn trên giao dịch thật của phần huấn luyện."""
    hard = np.zeros(len(y), dtype=bool); interv = np.zeros(len(y), dtype=bool); door = np.zeros(len(y), dtype=bool)
    meds = []
    for tr, va in folds:
        t_thr = best_thr(y[tr], s[tr]); t_med = med_threshold(y[tr], s[tr], t_thr)
        meds.append(t_med)
        hard[va] = s[va] >= t_thr; interv[va] = s[va] >= t_med
        if ctx is not None:
            door[va] = (s[va] < t_med) & (ctx[va] > ctx_door_thr(ctx[tr], y[tr], q))
    interv |= door
    fr, lg = y == 1, y == 0
    amt = lab.amount.values
    att = attack_ids(lab[fr])
    det = pd.Series(hard[fr], index=lab.index[fr]).groupby(att).max()
    ft = lab.fraud_type.fillna("NORMAL").values
    return {"hard_recall": float(hard[fr].mean()), "hard_tp": int(hard[fr].sum()), "hard_fp": int(hard[lg].sum()),
            "intervene_recall": float(interv[fr].mean()), "intervene_tp": int(interv[fr].sum()),
            "stepup_legit": int((interv & ~hard)[lg].sum()), "stepup_legit_rate": float((interv & ~hard)[lg].mean()),
            "touched_legit_rate": float(interv[lg].mean()),
            "value_recall_hard": float(amt[fr & hard].sum() / amt[fr].sum()),
            "value_recall_intervene": float(amt[fr & interv].sum() / amt[fr].sum()),
            "n_attacks": int(det.size), "attacks_detected": int(det.sum()), "case_recall": float(det.mean()),
            "n_fraud": int(fr.sum()), "n_legit": int(lg.sum()), "med_folds": [round(m, 4) for m in meds],
            "intervene_by_type": pd.Series(interv[fr]).groupby(ft[fr]).sum().astype(int).to_dict(),
            "hard_by_type": pd.Series(hard[fr]).groupby(ft[fr]).sum().astype(int).to_dict(),
            "ctx_tp": int(door[fr].sum()), "ctx_legit": int(door[lg].sum()), "ctx_legit_rate": float(door[lg].mean()),
            "missed_value_top": sorted([round(float(v)) for v in amt[fr & ~hard]], reverse=True)[:3],
            "hard": hard, "interv": interv, "door": door}


def novelty_threshold(an, low, ref_mask, budget=None):
    """Ngưỡng hàng đợi 'nghi vấn mẫu mới': chọn trên giai đoạn quá khứ sao cho cả kỳ chỉ khoảng `budget` ca."""
    budget = NOV_BUDGET if budget is None else budget
    pool = an[low & ref_mask]
    if len(pool) == 0:
        return 1.01
    q = 1 - budget / max(1, int(low.sum()))
    return float(np.quantile(pool, min(max(q, 0), 1)))


TYPE_RULES = {"FAILED_THEN_SUCCESS": ["flag_R5"], "VELOCITY_BURST": ["flag_R6"], "NEW_DEVICE": ["flag_R2"],
              "LOCATION_ANOMALY": ["flag_R3"], "AMOUNT_SPIKE": ["flag_R1_strong", "flag_R1_mild"],
              "HIGH_RISK_MERCHANT": ["flag_R4"]}


def leave_one_type_out(idx, y, ft, flags, X, an, thr, med, nov_thr, ctx=None, door_thr=None, grid=None):
    """Thí nghiệm ẩn từng kiểu gian lận (quy trình thống nhất của nhóm):
    - bỏ toàn bộ nhãn của kiểu đó khỏi dữ liệu học và tắt luật tương ứng;
    - học lại hồi quy logistic và trọng số luật trên phần nhãn còn lại; Isolation Forest giữ nguyên (không dùng nhãn);
    - giữ nguyên ngưỡng hiện hành; đếm số ca của kiểu bị ẩn vượt HIGH, vượt MEDIUM, lọt vào hàng đợi mẫu mới;
    - lớp ngữ cảnh không dùng nhãn gian lận nên giữ nguyên; ngưỡng cửa chọn trên giao dịch thật.
    grid: {q: ngưỡng cửa} – đếm thêm 'MEDIUM trở lên' theo từng ngân sách để vẽ bảng đánh đổi."""
    out = []; gsum = {q: 0 for q in (grid or {})}
    for T in [t for t in pd.unique(ft) if t not in ("NORMAL",)]:
        tgt = ft == T
        if tgt.sum() < 5:
            continue
        keep = ~tgt
        fl = flags.copy()
        for r in TYPE_RULES.get(T, []):
            fl[r] = 0
        tr_idx = idx[keep]
        m = lr().fit(X[tr_idx], y[keep])
        ml = m.predict_proba(X)[:, 1]
        ww = rule_weights(fl.loc[tr_idx], y[keep])
        rule = noisy_or(fl, ww)
        s = W_FUSION[0] * ml + W_FUSION[1] * rule + W_FUSION[2] * an
        sT = s[idx[tgt]]
        base_mp = sT >= med
        dr = np.zeros(len(s), dtype=bool) if ctx is None or door_thr is None else (s < med) & (ctx > door_thr)
        low = (s < med) & ~dr
        nov = low & (an >= nov_thr)
        for q, dt in (grid or {}).items():
            gsum[q] += int((base_mp | (ctx[idx[tgt]] > dt)).sum())
        out.append({"type": T, "rules_off": ", ".join(r.replace("flag_", "") for r in TYPE_RULES.get(T, [])) or "–",
                    "n": int(tgt.sum()), "high": int((sT >= thr).sum()), "medium_plus": int((base_mp | dr[idx[tgt]]).sum()),
                    "medium_plus_base": int(base_mp.sum()), "ctx_only": int((dr[idx[tgt]] & ~base_mp).sum()),
                    "queue": int(nov[idx[tgt]].sum()),
                    "queue_size": int(nov.sum())})
    tot = {"type": "TỔNG", "rules_off": "", "n": sum(o["n"] for o in out), "high": sum(o["high"] for o in out),
           "medium_plus": sum(o["medium_plus"] for o in out), "medium_plus_base": sum(o["medium_plus_base"] for o in out),
           "ctx_only": sum(o["ctx_only"] for o in out), "queue": sum(o["queue"] for o in out),
           "queue_size": None, "grid": gsum}
    return out + [tot]


def _pipe_from_pack(pack):
    """Dựng lại pipeline scikit-learn từ tham số đã lưu (để dùng chung FrozenModel)."""
    sc = StandardScaler(); sc.mean_ = np.array(pack["mean"]); sc.scale_ = np.array(pack["scale"])
    sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(sc.mean_); sc.n_samples_seen_ = pack.get("n_labels", 1)
    clf = LogisticRegression(); clf.coef_ = np.array([pack["coef"]]); clf.intercept_ = np.array([pack["intercept"]])
    clf.classes_ = np.array([0, 1]); clf.n_features_in_ = len(sc.mean_)
    return make_pipeline(sc, clf)


def run(files, tables=None, progress=lambda msg: None, pretrained_path=PRETRAINED, extra_labels=None):
    """Chạy toàn bộ Fraud Shield. Trả về dict kết quả (DataFrame + báo cáo + mô hình đóng băng).
    extra_labels: DataFrame(transaction_id, is_fraud) – nhãn do chuyên viên kết luận (học lại có kiểm soát)."""
    import fs_realtime as RT
    if tables is None:
        tables, checks, ok = validate(files)
        if not ok:
            raise ValueError("Dữ liệu chưa đạt kiểm tra bắt buộc")
    gd, kh, mc, tb = (tables[n] for n in ["transactions.csv", "customers.csv", "merchants.csv", "devices.csv"])
    nhan = tables.get("fraud_labels_sample.csv")
    t0 = time.perf_counter()
    progress("Đang tạo feature và hồ sơ hành vi (chỉ dùng dữ liệu quá khứ)…")
    df = build(gd, kh, mc, tb)
    df["build_order"] = np.arange(len(df))
    flags_B, _ = rules(df, True)
    flags_A, _ = rules(df, False)
    df = df.join(flags_B)
    X_all = df[FEATURES].fillna(0).values
    progress("Đang chấm điểm bất thường (Isolation Forest học trên giai đoạn quá khứ)…")
    ref_mask, ref_txt, live_txt, live_start = ref_period(df.datetime)
    iso, ref, an_all = fit_iso_ref(X_all, ref_mask)
    df["anomaly_score"] = an_all
    progress("Đang chấm lớp ngữ cảnh (so với thói quen của chính khách, giai đoạn quá khứ)…")
    V_all = ctx_matrix(df)
    ctx_ref = ctx_fit(V_all, ref_mask)
    ctx_all = ctx_score(V_all, ctx_ref)
    df["ctx_score"] = ctx_all
    ctx_info = None
    feat_ms = (time.perf_counter() - t0) * 1000
    w = W_FUSION
    lab_view = None
    model_live = None
    n_extra = 0

    labeled = nhan is not None
    if labeled:
        nhan = nhan[nhan.transaction_id.isin(df.transaction_id)].drop_duplicates("transaction_id")
        if extra_labels is not None and len(extra_labels):
            ex = extra_labels[~extra_labels.transaction_id.isin(nhan.transaction_id)
                              & extra_labels.transaction_id.isin(df.transaction_id)].drop_duplicates("transaction_id")
            ex = ex.assign(fraud_type=np.where(ex.is_fraud.astype(int) == 1, "ANALYST_FRAUD", "NORMAL"))
            n_extra = int(len(ex))
            nhan = pd.concat([nhan[["transaction_id", "is_fraud", "fraud_type"]],
                              ex[["transaction_id", "is_fraud", "fraud_type"]]], ignore_index=True)
        yy = nhan.is_fraud.astype(int)
        labeled = len(nhan) >= 50 and yy.sum() >= 10 and (yy == 0).sum() >= 10
    rep = {"version": VERSION, "seed": SEED, "fusion_weights": W_FUSION, "features": FEATURES,
           "n_transactions": int(len(df)), "reference_period": ref_txt, "live_period": live_txt,
           "n_extra_labels": n_extra}
    if labeled:
        mode = "train"
        progress("Đang huấn luyện và đánh giá (5-fold, ngưỡng lồng)…")
        _, _, anom_A = fit_iso_ref(df[F_NOPROF].fillna(0).values, ref_mask)
        idx = df.index[df.transaction_id.isin(nhan.transaction_id)].values
        lab = df.loc[idx].merge(nhan[["transaction_id", "is_fraud", "fraud_type"]], on="transaction_id",
                                how="left").set_index(idx)
        y = lab.is_fraud.values.astype(int); ft = lab.fraud_type.fillna("NORMAL").values
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
        FlB, FlA = flags_B.loc[idx], flags_A.loc[idx]
        w_rules = rule_weights(FlB, y)
        ml_B = oof(lab[FEATURES].fillna(0).values, y, folds)
        ml_A = oof(lab[F_NOPROF].fillna(0).values, y, folds)
        rB, rA = oof_rule(FlB, y, folds), oof_rule(FlA, y, folds)
        aB, aA = df.loc[idx, "anomaly_score"].values, anom_A[idx]
        cands = {"Chỉ Rule": rB, "Chỉ Isolation Forest": aB, "Chỉ Logistic Regression": ml_B,
                 "Gộp 3 tầng – không hồ sơ (A)": w[0] * ml_A + w[1] * rA + w[2] * aA,
                 "Gộp 3 tầng – có hồ sơ (B, chính thức)": w[0] * ml_B + w[1] * rB + w[2] * aB}
        comp, per_type = {}, {}
        for k, s in cands.items():
            e = evaluate(y, s, folds); per_type[k] = pd.Series(e.pop("pred")).groupby(ft).sum().astype(int).to_dict()
            comp[k] = r4(e)
        fused = cands["Gộp 3 tầng – có hồ sơ (B, chính thức)"]
        counts = pd.Series(ft).value_counts().to_dict()
        order = np.argsort(-fused)
        topk = [{"k": k, "fraud": int(y[order[:k]].sum()), "precision": round(y[order[:k]].sum() / k, 4),
                 "recall": round(y[order[:k]].sum() / y.sum(), 4)} for k in (25, 50, 100, 150, 200) if k <= len(y)]
        cthr = cost_threshold(y, fused); cp = fused >= cthr
        cost = {"threshold": round(cthr, 4), "recall": round(recall_score(y, cp), 4),
                "precision": round(precision_score(y, cp), 4), "alerts_labeled": int(cp.sum()),
                "fn": int(((~cp) & (y == 1)).sum()), "fp": int((cp & (y == 0)).sum()),
                "detected_by_type": pd.Series(cp.astype(int)).groupby(ft).sum().astype(int).to_dict()}
        # 3 tầng hành động – đánh giá bằng ngưỡng lồng
        ctx_l = ctx_all[idx]
        tiers = evaluate_tiers(y, fused, folds, lab.assign(fraud_type=ft), ctx=ctx_l)
        hard_pred, interv_pred, door_pred = tiers.pop("hard"), tiers.pop("interv"), tiers.pop("door")
        # Bảng đánh đổi: ngân sách cửa ngữ cảnh → bắt thêm bao nhiêu, bao nhiêu khách thật phải xác thực thêm
        trade = []
        for q in CTX_GRID:
            tq = evaluate_tiers(y, fused, folds, lab.assign(fraud_type=ft), ctx=ctx_l, q=q)
            trade.append({"budget": q, "intervene_tp": tq["intervene_tp"], "intervene_recall": round(tq["intervene_recall"], 4),
                          "stepup_legit_rate": round(tq["stepup_legit_rate"], 4), "ctx_tp": tq["ctx_tp"]})
        # Hiệu quả từng luật (trên nhãn) + mức giảm PR-AUC khi bỏ luật
        per_rule = {}
        for c in RULE_W:
            keep = {k: (0.0 if k == c else 1.0) for k in RULE_W}
            r_ = np.zeros(len(y))
            for tr, va in folds:
                ww = rule_weights(FlB.iloc[tr], y[tr]); ww = {k: v * keep[k] for k, v in ww.items()}
                r_[va] = noisy_or(FlB.iloc[va], ww)
            ap = average_precision_score(y, w[0] * ml_B + w[1] * r_ + w[2] * aB)
            per_rule[c] = {"fired_labeled": int(FlB[c].sum()), "tp": int(FlB[c][y == 1].sum()),
                           "precision": round(float(FlB[c][y == 1].sum() / max(1, FlB[c].sum())), 6),
                           "weight": w_rules[c], "fired_all": int(flags_B[c].sum()),
                           "pr_auc_drop": round(float(average_precision_score(y, fused) - ap), 6)}
        Xl = lab[FEATURES].fillna(0).values
        # Chia theo thời gian: học trên giai đoạn quá khứ, kiểm tra trên giai đoạn sau – cũng là mô hình chạy luồng
        tsplit = None
        trm, tem = ref_mask[idx], ~ref_mask[idx]
        if y[trm].sum() >= 10 and y[tem].sum() >= 5 and (y[trm] == 0).sum() >= 10:
            f2 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(trm.sum()), y[trm]))
            s_tr = w[0] * oof(Xl[trm], y[trm], f2) + w[1] * oof_rule(FlB[trm], y[trm], f2) + w[2] * aB[trm]
            thr_t = best_thr(y[trm], s_tr); med_t = med_threshold(y[trm], s_tr, thr_t)
            lr_t = lr().fit(Xl[trm], y[trm]); rw_t = rule_weights(FlB[trm], y[trm])
            s_te = w[0] * lr_t.predict_proba(Xl[tem])[:, 1] + w[1] * noisy_or(FlB[tem], rw_t) + w[2] * aB[tem]
            pt = s_te >= thr_t
            dt_t = ctx_door_thr(ctx_l[trm], y[trm])
            pm = (s_te >= med_t) | (ctx_l[tem] > dt_t)
            ye = y[tem]
            tsplit = {"train_months": ref_txt, "test_months": live_txt,
                      "n_train": int(trm.sum()), "fraud_train": int(y[trm].sum()), "n_test": int(tem.sum()),
                      "fraud_test": int(ye.sum()), "precision": round(precision_score(ye, pt, zero_division=0), 6),
                      "recall": round(recall_score(ye, pt), 6), "f1": round(f1_score(ye, pt), 6),
                      "pr_auc": round(average_precision_score(ye, s_te), 6),
                      "fp": int((pt & (ye == 0)).sum()), "fn": int((~pt & (ye == 1)).sum()),
                      "threshold": round(thr_t, 4), "medium_threshold": round(med_t, 4),
                      "intervene_recall": round(float(pm[ye == 1].mean()), 6),
                      "stepup_legit_rate": round(float((pm & ~pt)[ye == 0].mean()), 6)}
            low_t = ((w[0] * lr_t.predict_proba(X_all)[:, 1] + w[1] * noisy_or(flags_B, rw_t)
                      + w[2] * df.anomaly_score.values) < med_t) & ~(ctx_all > dt_t)
            model_live = RT.FrozenModel(lr_t, rw_t, iso, ref, thr_t, med_t,
                                        novelty_threshold(df.anomaly_score.values, low_t, ref_mask),
                                        f"{VERSION} · học {ref_txt}",
                                        {"train": ref_txt, "test": live_txt, "start": live_start},
                                        ctx_ref=ctx_ref, ctx_thr=dt_t)
        THR = best_thr(y, fused); MED = med_threshold(y, fused, THR)
        CTX_THR = ctx_door_thr(ctx_l, y)
        ctx_grid = {q: ctx_door_thr(ctx_l, y, q) for q in CTX_GRID}
        ctx_info = {"budget": CTX_BUDGET, "threshold": round(CTX_THR, 4),
                    "ref_q": round(float((ctx_all[ref_mask] > CTX_THR).mean()), 6),
                    "tradeoff": trade, "ctx_tp_nested": tiers["ctx_tp"], "ctx_legit_nested": tiers["ctx_legit"],
                    "ctx_legit_rate_nested": round(tiers["ctx_legit_rate"], 4)}
        crit_l = ((FlB["flag_R5"] == 1) | (FlB["flag_R6"] == 1)).values
        L = np.where((fused >= THR) & crit_l, "CRITICAL", np.where(fused >= THR, "HIGH",
                                                                   np.where(fused >= MED, "MEDIUM", "LOW")))
        levels_labeled = {l: {"n": int((L == l).sum()), "fraud": int(y[L == l].sum())}
                          for l in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]}
        top = max(v["pr_auc_drop"] for v in per_rule.values())
        for c, v in per_rule.items():
            dr, pr_, fpn = v["pr_auc_drop"], v["precision"], v["fired_labeled"] - v["tp"]
            if dr == top and dr > 0.001:
                note = "Quan trọng nhất"
            elif dr >= 0.005:
                note = "Quan trọng" + (f"; {fpn} báo nhầm" if fpn else "")
            elif dr > 0.001 and pr_ < 0.5:
                note = f"Báo nhầm nhiều: {fpn}/{v['fired_labeled']} lần bật là khách thật"
            elif dr > 0.001:
                note = "Ít ca, đóng góp nhỏ"
            elif pr_ >= 0.9:
                note = "Đúng nhưng ca nào cũng có luật khác cùng bật"
            elif pr_ < 0.2:
                note = "Không giúp phát hiện; giữ để ghi lý do, tuân thủ"
            else:
                note = "Chỉ là tín hiệu phụ"
            v["note"] = note; v["basis"] = RULE_BASIS[c]
        model = lr().fit(Xl, y)
        ml_full = model.predict_proba(X_all)[:, 1]
        ml_all = ml_full.copy(); ml_all[idx] = ml_B
        rule_all = noisy_or(flags_B, w_rules); rule_all[idx] = rB
        ev = evaluate(y, fused, folds)
        labf = lab.assign(pred=ev["pred"], risk_oof=fused, fraud_type=ft)
        fpr_seg = labf[labf.is_fraud == 0].groupby("customer_segment").pred.mean().round(4).to_dict()
        err_cols = ["transaction_id", "fraud_type", "amount", "avg_transaction_amount", "amt_ratio", "location",
                    "home_location", "ip_country", "merchant_risk_level", "profile_dev_count", "risk_oof"]
        kpi = {"intervene_recall": round(tiers["intervene_recall"], 4), "intervene_tp": tiers["intervene_tp"],
               "stepup_legit": tiers["stepup_legit"], "stepup_legit_rate": round(tiers["stepup_legit_rate"], 4),
               "hard_recall": round(float(ev["recall"]), 4), "hard_tp": ev["tp"], "hard_fp": ev["fp"],
               "value_recall_hard": round(tiers["value_recall_hard"], 4),
               "value_recall_intervene": round(tiers["value_recall_intervene"], 4),
               "case_recall": round(tiers["case_recall"], 4), "attacks_detected": tiers["attacks_detected"],
               "n_attacks": tiers["n_attacks"], "n_fraud": tiers["n_fraud"], "n_legit": tiers["n_legit"],
               "med_folds": tiers["med_folds"], "missed_value_top": tiers["missed_value_top"],
               "intervene_by_type": tiers["intervene_by_type"], "hard_by_type": tiers["hard_by_type"],
               "target_intervene": TARGET_INTERVENE, "stepup_cap": STEPUP_CAP,
               "ctx_tp": tiers["ctx_tp"], "ctx_legit": tiers["ctx_legit"]}
        rep.update({"labeled_counts": {k: int(v) for k, v in counts.items()}, "comparison_nested_cv": comp,
                    "detected_by_type": per_type, "topk": topk, "cost_threshold": cost, "time_split": tsplit,
                    "per_rule": per_rule, "levels_labeled": levels_labeled, "fpr_by_segment_labeled": fpr_seg,
                    "kpi": kpi,
                    "false_negatives": labf[(labf.is_fraud == 1) & (labf.pred == 0)][err_cols].round(3).to_dict("records"),
                    "false_positives": labf[(labf.is_fraud == 0) & (labf.pred == 1)][err_cols].round(3).to_dict("records")})
        lab_view = pd.DataFrame({"transaction_id": lab.transaction_id.values, "y": y, "fraud_type": ft,
                                 "score": np.round(fused, 5), "hard": hard_pred, "intervene": interv_pred,
                                 "ctx_door": door_pred})
        model_info = {"source": "Huấn luyện trên nhãn trong Data Pack vừa tải lên"
                                + (f" + {n_extra} nhãn chuyên viên" if n_extra else ""),
                      "n_labels": int(len(y)), "n_fraud": int(y.sum())}
        pipe_deploy = model
    else:
        mode = "score"
        progress("Không có nhãn – chấm điểm bằng mô hình đã huấn luyện sẵn…")
        pack = json.loads(Path(pretrained_path).read_text(encoding="utf-8"))
        THR = pack["threshold"]; MED = pack.get("medium_threshold", 0.6 * THR)
        CTX_THR = float(np.quantile(ctx_all[ref_mask], 1 - pack.get("context_ref_q", 0.06)))
        ctx_info = {"budget": CTX_BUDGET, "threshold": round(CTX_THR, 4), "ref_q": pack.get("context_ref_q", 0.06)}
        w_rules = pack.get("rule_weights", RULE_W)
        pipe_deploy = _pipe_from_pack(pack)
        ml_all = ml_full = pipe_deploy.predict_proba(X_all)[:, 1]
        rule_all = noisy_or(flags_B, w_rules)
        model_info = {"source": pack.get("source", "Mô hình huấn luyện sẵn"), "n_labels": pack.get("n_labels"),
                      "n_fraud": pack.get("n_fraud")}

    progress("Đang xếp hạng, viết lý do và gom case…")
    t1 = time.perf_counter()
    df["rule_score"] = rule_all
    df["ml_score"] = ml_all
    df["risk_score"] = (w[0] * df.ml_score + w[1] * df.rule_score + w[2] * df.anomaly_score).clip(0, 1)
    df["predicted_fraud"] = (df.risk_score >= THR).astype(int)
    crit = (df.flag_R5 == 1) | (df.flag_R6 == 1)
    lvl = np.where(df.risk_score >= THR, "HIGH", np.where(df.risk_score >= MED, "MEDIUM", "LOW"))
    df["risk_level"] = np.where((df.predicted_fraud == 1) & crit, "CRITICAL", lvl)
    door = (df.risk_level == "LOW").values & (ctx_all > CTX_THR)
    df["ctx_door"] = door.astype(int)
    df.loc[door, "risk_level"] = "MEDIUM"
    df["action"] = df.risk_level.map(ACTIONS)
    low = (df.risk_level == "LOW").values
    nov_thr = novelty_threshold(df.anomaly_score.values, low, ref_mask)
    df["novelty"] = (low & (df.anomaly_score.values >= nov_thr)).astype(int)
    d = np.where(df.predicted_fraud == 1, (df.risk_score - THR) / (1 - THR), (THR - df.risk_score) / THR)
    df["confidence"] = (0.5 + 0.5 * d).clip(0, 1).round(4)
    df["alert_reason"] = [reason(r) for r in df.itertuples()]
    if door.any():
        P = ctx_parts(V_all[door], ctx_ref)
        rows = df.loc[door, [c for c, _ in CTX_SPEC]].to_dict("records")
        notes = [ctx_note(p_, r_) for p_, r_ in zip(P, rows)]
        df.loc[door, "alert_reason"] = [n_ + ". " + a_ if not a_.startswith("Không có") else n_ + "."
                                        for n_, a_ in zip(notes, df.loc[door, "alert_reason"])]
    df = df.sort_values(["risk_score", "amount"], ascending=[False, False]).reset_index(drop=True)
    df["priority_rank"] = np.arange(1, len(df) + 1)
    infer_ms = (time.perf_counter() - t1) * 1000 + feat_ms
    df["processing_time_ms"] = round(infer_ms / len(df), 3)
    df["model_name"] = VERSION
    df["risk_score"] = df.risk_score.round(4)

    cases = make_cases(df)
    df["case_id"] = df.transaction_id.map(cases.attrs["map"])
    model_deploy = RT.FrozenModel(pipe_deploy, w_rules, iso, ref, THR, MED, nov_thr, VERSION,
                                  {"train": "toàn bộ nhãn", "test": None, "start": None},
                                  ctx_ref=ctx_ref, ctx_thr=CTX_THR)

    if labeled:
        progress("Đang chạy thí nghiệm ẩn từng kiểu gian lận…")
        rep["loto"] = leave_one_type_out(idx, y, ft, flags_B, X_all, an_all, THR, MED, nov_thr,
                                         ctx=ctx_all, door_thr=CTX_THR, grid=ctx_grid)
    rep.update({"mode": mode, "model_info": model_info, "threshold": round(THR, 4),
                "medium_threshold": round(MED, 4), "rule_weights": w_rules, "novelty_threshold": round(nov_thr, 4),
                "n_novelty": int(df.novelty.sum()),
                "n_alerts": int(df.predicted_fraud.sum()), "n_cases": int(len(cases)),
                "n_stepup": int((df.risk_level == "MEDIUM").sum()), "n_ctx_door": int(door.sum()),
                "context_layer": ctx_info,
                "case_reduction": round(1 - len(cases) / max(1, df.predicted_fraud.sum()), 4),
                "risk_level_counts": {k: int(v) for k, v in df.risk_level.value_counts().to_dict().items()},
                "alert_rate_by_segment": df.groupby("customer_segment").predicted_fraud.mean().round(4).to_dict(),
                "alert_rate_by_home_location": df.groupby("home_location").predicted_fraud.mean().round(4).to_dict(),
                "rule_hits_on_alerts": {RULE_NAMES[c]: int(df.loc[df.predicted_fraud == 1, c].sum()) for c in RULE_W},
                "latency_ms_per_txn_batch": round(infer_ms / len(df), 3),
                "period": [f"{df.datetime.min():%d/%m/%Y}", f"{df.datetime.max():%d/%m/%Y}"]})
    # File nộp theo template
    if "submission_output_template.csv" in tables:
        tmpl = tables["submission_output_template.csv"]
        cols = [c for c in tmpl.columns if c in df.columns] or OUT_COLS
        sub = tmpl[["transaction_id"]].merge(df[cols], on="transaction_id", how="left")
    else:
        order = {t: i for i, t in enumerate(gd.transaction_id)}
        sub = df[OUT_COLS].sort_values("transaction_id", key=lambda s: s.map(order)).reset_index(drop=True)
    return {"df": df, "cases": cases, "report": rep, "submission": sub, "lab_view": lab_view, "mode": mode,
            "model": model_deploy, "model_live": model_live, "tables": tables}


def make_cases(df):
    a = df[df.predicted_fraud == 1].sort_values(["customer_id", "datetime"]).copy()
    rank_lvl = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    if a.empty:
        c = pd.DataFrame(columns=["case_id", "case_rank", "customer_id", "n_transactions", "start_time", "end_time",
                                  "total_amount", "max_risk_score", "case_level", "devices", "transaction_ids",
                                  "top_reason"])
        c.attrs["map"] = {}
        return c
    new = (a.customer_id != a.customer_id.shift()) | (a.datetime.diff().dt.total_seconds() > CASE_GAP_S)
    a["seq"] = new.cumsum()
    top = a.sort_values("risk_score", ascending=False).drop_duplicates("seq").set_index("seq").alert_reason
    cases = a.groupby("seq").agg(customer_id=("customer_id", "first"), n_transactions=("transaction_id", "size"),
                                 start_time=("datetime", "min"), end_time=("datetime", "max"),
                                 total_amount=("amount", "sum"), max_risk_score=("risk_score", "max"),
                                 case_level=("risk_level", lambda s: max(s, key=rank_lvl.get)),
                                 devices=("device_id", lambda s: ";".join(sorted(set(s)))),
                                 transaction_ids=("transaction_id", lambda s: ";".join(s))).reset_index()
    cases["top_reason"] = cases.seq.map(top)
    cases = cases.sort_values(["max_risk_score", "n_transactions"], ascending=[False, False]).reset_index(drop=True)
    cases.insert(0, "case_rank", np.arange(1, len(cases) + 1))
    cases.insert(0, "case_id", [f"CASE{i:04d}" for i in cases.case_rank])
    m = dict(zip(a.transaction_id, a.seq.map(dict(zip(cases.seq, cases.case_id)))))
    cases = cases.drop(columns="seq")
    cases.attrs["map"] = m
    return cases


def retrain_compare(res, extra):
    """Học lại có kiểm soát: so sánh mô hình TRƯỚC và SAU khi thêm nhãn chuyên viên.
    Cả hai mô hình học trên giai đoạn quá khứ và được kiểm tra trên nhãn BTC của giai đoạn sau (không dùng để học).
    Nhãn chuyên viên chỉ được dùng nếu giao dịch thuộc giai đoạn học, để không rò rỉ vào tập kiểm tra."""
    df = res["df"].sort_values("build_order").reset_index(drop=True)
    btc = res["tables"].get("fraud_labels_sample.csv")
    if btc is None:
        return None
    ref_mask, ref_txt, live_txt, _ = ref_period(df.datetime)
    lab = df.merge(btc[["transaction_id", "is_fraud"]].drop_duplicates("transaction_id"), on="transaction_id",
                   how="left")
    tr_btc = lab.is_fraud.notna().values & ref_mask
    te = lab.is_fraud.notna().values & ~ref_mask
    ex = extra[~extra.transaction_id.isin(btc.transaction_id)] if extra is not None else extra
    exm = dict(zip(ex.transaction_id, ex.is_fraud.astype(int))) if ex is not None and len(ex) else {}
    y_ex = lab.transaction_id.map(exm)
    tr_ex = y_ex.notna().values & ref_mask
    X = lab[FEATURES].fillna(0).values
    Fl = lab[list(RULE_W)].astype(int)
    an = lab.anomaly_score.values
    cx = lab.ctx_score.values if "ctx_score" in lab else np.zeros(len(lab))
    yte = lab.is_fraud.values[te].astype(int)
    out = {"train_period": ref_txt, "test_period": live_txt, "n_test": int(te.sum()), "fraud_test": int(yte.sum()),
           "n_new_labels": int(tr_ex.sum()), "n_new_fraud": int(np.nansum(y_ex.values[tr_ex])),
           "n_ignored_future": int((y_ex.notna().values & ~ref_mask).sum())}
    for tag, mask, yv in [("before", tr_btc, lab.is_fraud.values),
                          ("after", tr_btc | tr_ex, np.where(tr_ex, y_ex.values, lab.is_fraud.values))]:
        y = yv[mask].astype(int)
        f2 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
        s_tr = W_FUSION[0] * oof(X[mask], y, f2) + W_FUSION[1] * oof_rule(Fl[mask], y, f2) + W_FUSION[2] * an[mask]
        thr = best_thr(y, s_tr); med = med_threshold(y, s_tr, thr)
        m = lr().fit(X[mask], y); rw = rule_weights(Fl[mask], y)
        s = W_FUSION[0] * m.predict_proba(X[te])[:, 1] + W_FUSION[1] * noisy_or(Fl[te], rw) + W_FUSION[2] * an[te]
        p, pm = s >= thr, (s >= med) | (cx[te] > ctx_door_thr(cx[mask], y))
        out[tag] = {"n_train": int(mask.sum()), "fraud_train": int(y.sum()),
                    "pr_auc": round(float(average_precision_score(yte, s)), 4),
                    "recall": round(float(p[yte == 1].mean()), 4), "fp": int((p & (yte == 0)).sum()),
                    "f1": round(float(f1_score(yte, p)), 4),
                    "intervene_recall": round(float(pm[yte == 1].mean()), 4),
                    "stepup_rate": round(float((pm & ~p)[yte == 0].mean()), 4),
                    "threshold": round(thr, 4), "medium_threshold": round(med, 4)}
    out["recommend"] = bool(out["after"]["pr_auc"] >= out["before"]["pr_auc"] - 0.005
                            and out["after"]["fp"] <= out["before"]["fp"] + 2)
    return out


def train_pretrained(files, out_path=PRETRAINED):
    """Huấn luyện mô hình dùng cho chế độ chỉ chấm điểm (chạy một lần trên Data Pack có nhãn)."""
    res = run(files)
    assert res["mode"] == "train"
    df, lv = res["df"], res["lab_view"]
    lab = df[df.transaction_id.isin(lv.transaction_id)].merge(lv[["transaction_id", "y"]], on="transaction_id")
    model = lr().fit(lab[FEATURES].fillna(0).values, lab.y.values)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sc, clf = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    pack = {"version": VERSION, "features": FEATURES, "threshold": res["report"]["threshold"],
            "medium_threshold": res["report"]["medium_threshold"], "rule_weights": res["report"]["rule_weights"],
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]), "n_labels": int(len(lab)), "n_fraud": int(lab.y.sum()),
            "context_ref_q": res["report"]["context_layer"]["ref_q"], "context_budget": CTX_BUDGET,
            "source": "Học trên 2.000 nhãn công khai của Data Pack BTC (snapshot 26/10/2026, chỉ dùng dữ liệu quá khứ)"}
    Path(out_path).write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    import sys
    res = run(collect_dir(sys.argv[1]))
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    res["submission"].to_csv(out / "submission_output.csv", index=False, encoding="utf-8-sig")
    res["cases"].to_csv(out / "cases.csv", index=False, encoding="utf-8-sig")
    (out / "kpi_report.json").write_text(json.dumps(res["report"], ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    print(json.dumps({k: res["report"].get(k) for k in ["mode", "threshold", "n_alerts", "n_cases", "risk_level_counts"]},
                     ensure_ascii=False))
