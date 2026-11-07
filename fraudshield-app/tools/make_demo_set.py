"""Tạo bộ 1.000 giao dịch mẫu cho ứng dụng khách FS Pay (dùng để trình diễn luồng thời gian thực).

- Khách hàng, thói quen (máy, tỉnh, kênh, mức chi) lấy từ lịch sử Data Pack BTC để giao dịch "giống thật".
- Giao dịch là HƯ CẤU, do nhóm tạo theo kịch bản; diễn ra sau thời điểm cuối của dữ liệu BTC.
- Cột `scenario` và `is_fraud_sim` là kịch bản nhóm cài vào – dùng để đối chiếu khi demo, KHÔNG dùng để đánh giá mô hình.
Chạy: python tools/make_demo_set.py  →  data_demo/fs_pay_demo_1000.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(2026)
gd = pd.read_csv(ROOT / "data/transactions.csv", parse_dates=["datetime"])
kh = pd.read_csv(ROOT / "data/customers.csv").set_index("customer_id")
mc = pd.read_csv(ROOT / "data/merchants.csv")
PROV = sorted(gd.location.unique())
FOREIGN = ["SG", "TH", "US", "KR", "JP", "MY"]
cnt = gd.customer_id.value_counts()
CUSTS = sorted(cnt[cnt >= 10].index.tolist(), key=lambda c: -cnt[c])[:30]
START = (gd.datetime.max() + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=7)   # 01/07/2026 07:00
DAYS = 4

tb = pd.read_csv(ROOT / "data/devices.csv")
prof = {}
for c in CUSTS:
    h = gd[gd.customer_id == c]
    own = tb[(tb.registered_customer_id == c) & (tb.device_risk_level != "HIGH")].device_id.tolist()
    used_own = [d for d in h.device_id.value_counts().index if d in own] or own or [h.device_id.mode().iloc[0]]
    # Giao dịch bình thường chỉ dùng thiết bị đăng ký cho chính khách (không dùng lại máy của kẻ gian trong lịch sử)
    prof[c] = {"dev": used_own[0], "devs": used_own,
               "home": kh.loc[c, "home_location"], "locs": h.location.value_counts().index.tolist(),
               "chan": h.payment_channel.mode().iloc[0], "chans": h.payment_channel.unique().tolist(),
               "avg": float(kh.loc[c, "avg_transaction_amount"]), "mers": h.merchant_id.unique().tolist()}
LOW_M = mc[mc.merchant_risk_level != "HIGH"].merchant_id.tolist()
HIGH_M = mc[mc.merchant_risk_level == "HIGH"].merchant_id.tolist()
CAT = mc.set_index("merchant_id").merchant_category
TRANSFER_M = mc[mc.merchant_category.isin(["MONEY_TRANSFER"])].merchant_id.tolist()
ELEC_M = mc[mc.merchant_category.isin(["ELECTRONICS", "ONLINE_MARKETPLACE"]) & (mc.merchant_risk_level != "HIGH")].merchant_id.tolist()
rows = []
dev_seq = [0]
LAST = {}


def amt(x):
    return float(max(10_000, round(x / 1000) * 1000))


def new_dev(kind="LA"):
    dev_seq[0] += 1
    return f"DEV-{kind}-{dev_seq[0]:03d}"


def add(c, t, mer, a, dev, loc, ip="VN", chan=None, status="SUCCESS", scen="Bình thường", fraud=0,
        new_device=False, rooted=False):
    p = prof[c]
    if fraud:
        LAST.setdefault(c, []).append(t)
    rows.append({"customer_id": c, "merchant_id": mer, "datetime": t, "amount": amt(a), "currency": "VND",
                 "payment_channel": chan or p["chan"], "device_id": dev, "location": loc, "transaction_status": status,
                 "ip_country": ip, "is_card_present": False,
                 "authentication_method": "BIOMETRIC" if dev == p["dev"] else "OTP",
                 "device_new_record": bool(new_device), "device_rooted": bool(rooted), "scenario": scen,
                 "is_fraud_sim": int(fraud)})


def rand_time(day_hours=(7, 23)):
    d = int(rng.integers(0, DAYS))
    hh = rng.uniform(*day_hours)
    return START.normalize() + pd.Timedelta(days=d, hours=hh)


# ---------------- giao dịch bình thường: mỗi khách cách nhau ≥ 40 phút
N_NORMAL_PLAIN, N_TRAVEL, N_BIG, N_NEWCHAN = 853, 30, 15, 20
last = LAST
for c in CUSTS:
    last.setdefault(c, [])


def free_time(c, gap_min=60):
    for _ in range(200):
        t = rand_time()
        if all(abs((t - x).total_seconds()) > gap_min * 60 for x in last[c]):
            last[c].append(t)
            return t
    t = rand_time(); last[c].append(t); return t


# ---------------- kịch bản gian lận
def pick():
    return CUSTS[int(rng.integers(len(CUSTS)))]


for _ in range(7):   # chiếm tài khoản: máy lạ, IP nước ngoài, tỉnh khác
    c = pick(); p = prof[c]; t = rand_time((0, 24)); d = new_dev()
    loc = rng.choice([x for x in PROV if x != p["home"]]); ip = rng.choice(FOREIGN)
    for k in range(int(rng.integers(1, 3))):
        add(c, t + pd.Timedelta(minutes=12 * k), rng.choice(ELEC_M + TRANSFER_M), p["avg"] * rng.uniform(3, 8), d, loc, ip=ip,
            scen="Gian lận – chiếm tài khoản, máy lạ ở nước ngoài", fraud=1, new_device=True)
for _ in range(6):   # thử sai OTP rồi thành công
    c = pick(); p = prof[c]; t = rand_time((0, 24)); d = new_dev()
    n_fail = int(rng.integers(2, 4)); a = p["avg"] * rng.uniform(2, 5)
    for k in range(n_fail + 1):
        add(c, t + pd.Timedelta(seconds=25 * k), rng.choice(ELEC_M), a, d, p["home"],
            status="FAILED" if k < n_fail else "SUCCESS", scen="Gian lận – thử sai OTP rồi thành công", fraud=1,
            new_device=True)
for _ in range(6):   # dồn dập: điện thoại bị lấy cắp, nhiều giao dịch trong 5 phút
    c = pick(); p = prof[c]; t = rand_time((0, 24))
    for k in range(int(rng.integers(3, 6))):
        add(c, t + pd.Timedelta(seconds=55 * k), rng.choice(LOW_M), p["avg"] * rng.uniform(0.8, 2.0), p["dev"], p["home"],
            scen="Gian lận – giao dịch dồn dập", fraud=1)
for _ in range(6):   # số tiền bất thường từ máy quen
    c = pick(); p = prof[c]
    add(c, rand_time(), rng.choice(TRANSFER_M + ELEC_M), p["avg"] * rng.uniform(6, 12), p["dev"], p["home"],
        scen="Gian lận – số tiền bất thường", fraud=1)
for _ in range(5):   # merchant rủi ro cao, tỉnh khác
    c = pick(); p = prof[c]; t = rand_time((0, 24)); mer = rng.choice(HIGH_M)
    loc = rng.choice([x for x in PROV if x != p["home"]])
    for k in range(int(rng.integers(1, 3))):
        add(c, t + pd.Timedelta(minutes=20 * k), mer, p["avg"] * rng.uniform(2, 4), p["dev"], loc,
            scen="Gian lận – merchant rủi ro cao", fraud=1)
for _ in range(4):   # máy đã root / giả lập
    c = pick(); p = prof[c]; t = rand_time((0, 24)); d = new_dev("ROOT")
    for k in range(int(rng.integers(1, 3))):
        add(c, t + pd.Timedelta(minutes=15 * k), rng.choice(ELEC_M), p["avg"] * rng.uniform(2, 5), d, p["home"],
            scen="Gian lận – máy đã root/giả lập", fraud=1, new_device=True, rooted=True)
for _ in range(2):   # một thiết bị dùng cho nhiều tài khoản (dấu hiệu số 8)
    t = rand_time((0, 24)); d = new_dev("RING"); loc = rng.choice(PROV)
    for k, c in enumerate(rng.choice(CUSTS, 3, replace=False)):
        add(c, t + pd.Timedelta(minutes=18 * k), rng.choice(TRANSFER_M), prof[c]["avg"] * rng.uniform(2, 4), d, loc,
            scen="Gian lận – một thiết bị dùng cho nhiều tài khoản", fraud=1, new_device=(k == 0))
for _ in range(4):   # lừa đảo dụ chính chủ tự chuyển tiền – rất khó bắt
    c = pick(); p = prof[c]
    add(c, rand_time(), rng.choice(TRANSFER_M), p["avg"] * rng.uniform(1.5, 3), p["dev"], p["home"],
        scen="Gian lận – lừa chính chủ tự chuyển tiền (khó bắt)", fraud=1)

for i in range(N_NORMAL_PLAIN):
    c = CUSTS[i % len(CUSTS)]
    p = prof[c]
    mer = rng.choice(p["mers"]) if rng.random() < 0.6 else rng.choice(LOW_M)
    loc = p["home"] if rng.random() < 0.85 else rng.choice(p["locs"])
    chan = p["chan"] if rng.random() < 0.85 else rng.choice(p["chans"])
    dev = p["dev"] if rng.random() < 0.9 else rng.choice(p["devs"])
    a = p["avg"] * rng.lognormal(0, 0.45)
    add(c, free_time(c), mer, a, dev, loc, chan=chan)
for i in range(N_TRAVEL):      # đi du lịch: máy quen, tỉnh mới, IP Việt Nam
    c = CUSTS[int(rng.integers(len(CUSTS)))]
    p = prof[c]
    loc = rng.choice([x for x in PROV if x not in p["locs"]] or PROV)
    add(c, free_time(c), rng.choice(mc[mc.merchant_category.isin(["HOTEL", "DINING", "TRAVEL", "FUEL"])
                                        & (mc.merchant_risk_level != "HIGH")].merchant_id.tolist()),
        p["avg"] * rng.uniform(0.8, 2.0), p["dev"], loc, scen="Bình thường – đi du lịch tỉnh khác")
for i in range(N_BIG):         # mua món lớn hợp lệ: máy quen, ở nhà, gấp 3–4 lần
    c = CUSTS[int(rng.integers(len(CUSTS)))]
    p = prof[c]
    add(c, free_time(c), rng.choice(ELEC_M), p["avg"] * rng.uniform(3, 4), p["dev"], p["home"],
        scen="Bình thường – mua món lớn")
for i in range(N_NEWCHAN):     # đổi kênh thanh toán
    c = CUSTS[int(rng.integers(len(CUSTS)))]
    p = prof[c]
    other = [x for x in ["CARD", "QR", "E_WALLET", "MOBILE_BANKING", "BANK_TRANSFER"] if x not in p["chans"]] or ["QR"]
    add(c, free_time(c), rng.choice(LOW_M), p["avg"] * rng.lognormal(0, 0.3), p["dev"], p["home"],
        chan=rng.choice(other), scen="Bình thường – thử kênh mới")


df = pd.DataFrame(rows).sort_values(["datetime", "customer_id"]).reset_index(drop=True)
# cắt/bù đúng 1.000 giao dịch bằng giao dịch bình thường
plain = df.index[df.scenario == "Bình thường"]
if len(df) > 1000:
    df = df.drop(rng.choice(plain, len(df) - 1000, replace=False)).reset_index(drop=True)
df.insert(0, "transaction_id", [f"FSP-{i + 1:05d}" for i in range(len(df))])
df["datetime"] = df.datetime.dt.strftime("%Y-%m-%d %H:%M:%S")
out = ROOT / "data_demo/fs_pay_demo_1000.csv"
df.to_csv(out, index=False)
print(len(df), int(df.is_fraud_sim.sum()), df.scenario.value_counts().to_dict())
