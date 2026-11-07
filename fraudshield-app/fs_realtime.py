"""
FRAUD SHIELD v3.1 – CHẤM ĐIỂM THỜI GIAN THỰC

- FlatIForest: Isolation Forest (300 cây) đã huấn luyện, chuyển sang mảng numpy để chấm từng giao dịch
  trong ~0,2 ms; kết quả trùng với scikit-learn.
- FrozenModel: mô hình đã "đóng băng" (hồi quy logistic + trọng số luật + Isolation Forest + bảng phân vị
  tham chiếu + ngưỡng) – dùng chung cho chấm theo lô và chấm theo luồng.
- OnlineScorer: bảng trạng thái trong bộ nhớ cho từng khách hàng / thiết bị; nhận từng giao dịch theo thứ tự
  thời gian, chỉ dùng thông tin trước hoặc tại thời điểm giao dịch.
- parity_check: cho toàn bộ giao dịch đi qua OnlineScorer và đối chiếu với chấm theo lô.
"""
import math
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pandas as pd
import fraud_core as F


def _average_path_length(n):
    """Độ dài đường đi trung bình c(n) của Isolation Forest (cùng công thức với scikit-learn)."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    m = n > 2
    out[m] = 2.0 * (np.log(n[m] - 1.0) + np.euler_gamma) - 2.0 * (n[m] - 1.0) / n[m]
    return out

ACTIONS = {"CRITICAL": "Tạm dừng giao dịch, gọi xác minh ngay",
           "HIGH": "Tạm dừng, chuyên viên kiểm tra trong ca",
           "MEDIUM": "Yêu cầu xác thực thêm (OTP/sinh trắc học)",
           "LOW": "Cho qua"}


# ============================================================================ Isolation Forest dạng mảng
class FlatIForest:
    """Chuyển IsolationForest (scikit-learn) thành mảng numpy; tính đồng loạt trên mọi cây."""

    def __init__(self, iso):
        ests = iso.estimators_
        T = len(ests)
        M = max(e.tree_.node_count for e in ests)
        self.CL = np.full((T, M), -1, dtype=np.int64)
        self.CR = np.full((T, M), -1, dtype=np.int64)
        self.FE = np.zeros((T, M), dtype=np.int64)
        self.TH = np.zeros((T, M))
        self.LD = np.zeros((T, M))
        depth = 0
        for i, (est, feats) in enumerate(zip(ests, iso.estimators_features_)):
            t = est.tree_
            n = t.node_count
            self.CL[i, :n] = t.children_left
            self.CR[i, :n] = t.children_right
            self.FE[i, :n] = np.where(t.feature >= 0, np.asarray(feats)[np.maximum(t.feature, 0)], 0)
            self.TH[i, :n] = t.threshold
            self.LD[i, :n] = _average_path_length(t.n_node_samples)
            depth = max(depth, t.max_depth)
        self.T = T
        self.R = np.arange(T)
        self.max_depth = depth
        self.cN = float(_average_path_length(np.array([iso.max_samples_]))[0])

    def raw_one(self, x):
        """Điểm dị biệt thô của 1 giao dịch (càng cao càng bất thường) = -score_samples."""
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        R, node = self.R, np.zeros(self.T, dtype=np.int64)
        d = np.zeros(self.T)
        for _ in range(self.max_depth + 1):
            cl = self.CL[R, node]
            leaf = cl == -1
            if leaf.all():
                break
            go = np.where(x[self.FE[R, node]] <= self.TH[R, node], cl, self.CR[R, node])
            node = np.where(leaf, node, go)
            d += ~leaf
        return float(2.0 ** (-(d + self.LD[R, node]).mean() / self.cN))

    def raw(self, X):
        """Chấm cả lô (dùng cho kiểm thử)."""
        return np.array([self.raw_one(x) for x in np.asarray(X)])


def to_percentile(raw, ref):
    """Quy điểm thô sang thang 0–1 theo bảng phân vị tham chiếu của giai đoạn quá khứ."""
    return np.searchsorted(ref, np.round(raw, 12), side="right") / len(ref)


# ============================================================================ Mô hình đóng băng
class FrozenModel:
    def __init__(self, lr_pipe, rule_w, iso, ref, thr, med, nov_thr, name, info=None):
        sc, clf = lr_pipe.named_steps["standardscaler"], lr_pipe.named_steps["logisticregression"]
        self.lr = lr_pipe
        self.mean, self.scale = sc.mean_.copy(), sc.scale_.copy()
        self.coef, self.b = clf.coef_[0].copy(), float(clf.intercept_[0])
        self.rule_w = dict(rule_w)
        self.iso = iso
        self.flat = FlatIForest(iso)
        self.ref = np.asarray(ref)
        self.thr, self.med, self.nov_thr = float(thr), float(med), float(nov_thr)
        self.w = F.W_FUSION
        self.name = name
        self.info = info or {}

    # ---------- chấm theo lô (dùng scikit-learn) – để đối chiếu
    def score_frame(self, df):
        X = df[F.FEATURES].fillna(0).values
        ml = self.lr.predict_proba(X)[:, 1]
        rule = F.noisy_or(df[list(F.RULE_W)].astype(int), self.rule_w)
        an = to_percentile(-self.iso.score_samples(X), self.ref)
        s = np.clip(self.w[0] * ml + self.w[1] * rule + self.w[2] * an, 0, 1)
        return pd.DataFrame({"transaction_id": df.transaction_id.values, "ml_score": ml, "rule_score": rule,
                             "anomaly_score": an, "risk_score": s,
                             "risk_level": level(s, df.flag_R5.values, df.flag_R6.values, self.thr, self.med)})

    # ---------- chấm 1 giao dịch (numpy thuần)
    def score_vec(self, x, flags):
        z = ((x - self.mean) / self.scale) @ self.coef + self.b
        ml = 1.0 / (1.0 + math.exp(-z))
        p = 1.0
        for c, wv in self.rule_w.items():
            if flags[c]:
                p *= 1.0 - wv
        rule = 1.0 - p
        an = float(np.searchsorted(self.ref, round(self.flat.raw_one(x), 12), side="right") / len(self.ref))
        s = min(1.0, max(0.0, self.w[0] * ml + self.w[1] * rule + self.w[2] * an))
        return ml, rule, an, s


def level(s, r5, r6, thr, med):
    s = np.asarray(s)
    crit = (np.asarray(r5) == 1) | (np.asarray(r6) == 1)
    lv = np.where(s >= thr, "HIGH", np.where(s >= med, "MEDIUM", "LOW"))
    return np.where((s >= thr) & crit, "CRITICAL", lv)


# ============================================================================ Bảng trạng thái trong bộ nhớ
def _tobool(v):
    return str(v).upper() in ("TRUE", "1", "YES")


class OnlineScorer:
    """Chấm từng giao dịch theo thứ tự thời gian. Trạng thái của mỗi khách hàng / thiết bị nằm trong RAM;
    mỗi giao dịch chỉ đọc và cập nhật bản ghi của đúng khách hàng, thiết bị đó."""

    def __init__(self, model, kh, mc, tb):
        self.m = model
        self.cust = kh.set_index("customer_id").to_dict("index")
        self.merch = mc.set_index("merchant_id").to_dict("index")
        tb = tb.copy()
        tb["first_seen_date"] = pd.to_datetime(tb.first_seen_date)
        self.dev = tb.set_index("device_id").to_dict("index")
        self.C, self.D = {}, {}
        self.n_scored = 0

    def _state(self, cid):
        st = self.C.get(cid)
        if st is None:
            st = {"win": deque(), "last": None, "n": 0, "s": 0.0, "ss": 0.0, "vals": [],
                  "seen": {"device": set(), "location": set(), "channel": set(), "ip": set()}, "ipcnt": {}}
            self.C[cid] = st
        return st

    def features(self, t, update=True):
        """t: dict một giao dịch (cột như transactions.csv). Trả về dict đặc trưng + thông tin hiển thị."""
        cid, did = t["customer_id"], t["device_id"]
        k = self.cust.get(cid, {}); mm = self.merch.get(t["merchant_id"], {}); dv = self.dev.get(did, {})
        dt = pd.Timestamp(t["datetime"])
        ts = (dt - pd.Timestamp("2000-01-01")).total_seconds()
        amount = float(t["amount"])
        avg = k.get("avg_transaction_amount", np.nan)
        f = {"amount": amount, "avg_transaction_amount": avg, "amt_ratio": amount / avg if avg else np.nan}
        reg = dv.get("registered_customer_id")
        f["dev_not_registered_to_cust"] = int(reg != cid)
        fs = dv.get("first_seen_date")
        age = (dt.normalize() - fs).days if fs is not None and not pd.isna(fs) else np.nan
        f["dev_age_days"] = max(age, 0) if not pd.isna(age) else np.nan
        f["dev_is_new"] = int((not pd.isna(f["dev_age_days"])) and f["dev_age_days"] <= 30)
        f["loc_mismatch"] = int(t["location"] != k.get("home_location"))
        f["ip_foreign"] = int(t["ip_country"] != "VN")
        f["merchant_high"] = int(mm.get("merchant_risk_level") == "HIGH")
        f["auth_none"] = int(t["authentication_method"] == "NONE")
        f["dev_compromised"] = int(dv.get("device_risk_level") == "HIGH" or _tobool(dv.get("rooted_or_jailbroken"))
                                   or _tobool(dv.get("emulator_flag")))
        st = self._state(cid)
        fail = int(t["transaction_status"] != "SUCCESS")
        win = st["win"]
        while win and win[0][0] < ts - F.CASE_GAP_S:
            win.popleft()
        n5 = 1 + sum(1 for x in win if x[0] >= ts - 300)
        nf = fail + sum(x[1] for x in win)
        f["n_txn_5m"] = n5
        f["n_fail_30m"] = nf
        f["n_succ_30m"] = (len(win) + 1) - nf
        gap = max(0.0, ts - st["last"]) if st["last"] is not None else np.nan
        f["min_gap_s"] = gap
        f["log_min_gap"] = math.log1p(gap) if not pd.isna(gap) else math.log1p(1e7)
        keys = {"device": did, "location": t["location"], "channel": t["payment_channel"], "ip": t["ip_country"]}
        for nm, v in keys.items():
            f[f"prof_new_{nm}"] = int(v not in st["seen"][nm])
        la = math.log(amount)
        n = st["n"]
        if n >= 1:
            m = st["s"] / n
            f["prof_typical_amt"] = math.exp(m)
        else:
            m = None
            f["prof_typical_amt"] = amount
        if n >= 3:
            sd = max(float(np.std(st["vals"], ddof=1)), 0.3)
            f["prof_amt_z"] = (la - m) / sd
        else:
            f["prof_amt_z"] = 0.0
        f["prof_amt_dev"] = int(f["prof_amt_z"] >= 3)
        f["cust_n_txn"] = n + 1
        f["n_same_ip_other"] = st["ipcnt"].get(t["ip_country"], 0)
        f["profile_dev_count"] = (f["prof_amt_dev"] + f["prof_new_device"] + f["prof_new_location"]
                                  + f["prof_new_channel"] + f["prof_new_ip"])
        cs = self.D.get(did, set())
        f["dev_n_cust_sofar"] = len(cs | {cid})
        f["dev_shared"] = int(f["dev_n_cust_sofar"] >= 2)
        if update:
            win.append((ts, fail))
            st["last"] = ts
            for nm, v in keys.items():
                st["seen"][nm].add(v)
            st["n"] = n + 1; st["s"] += la; st["ss"] += la * la; st["vals"].append(la)
            st["ipcnt"][t["ip_country"]] = st["ipcnt"].get(t["ip_country"], 0) + 1
            self.D.setdefault(did, set()).add(cid)
        return f

    @staticmethod
    def flags(f):
        ar = f["amt_ratio"]
        return {"flag_R1_strong": int(ar >= 5), "flag_R1_mild": int(3 <= ar < 5),
                "flag_R2": int(f["dev_not_registered_to_cust"] == 1 and f["dev_is_new"] == 1),
                "flag_R3": int(f["ip_foreign"] == 1 and f["loc_mismatch"] == 1),
                "flag_R4": int(f["merchant_high"] == 1 and (ar >= 2 or f["prof_new_location"] == 1)),
                "flag_R5": int(f["n_fail_30m"] >= 2), "flag_R6": int(f["n_txn_5m"] >= 2),
                "flag_R7": int(f["dev_shared"] == 1), "flag_C1": int(f["dev_compromised"] == 1),
                "flag_C2": int(f["auth_none"] == 1)}

    def warm(self, rows):
        """Nạp lịch sử (không chấm điểm) để dựng trạng thái ban đầu."""
        for t in rows:
            self.features(t, update=True)

    def score(self, t, with_reason=True):
        t0 = time.perf_counter()
        f = self.features(t, update=True)
        fl = self.flags(f)
        x = np.array([0.0 if pd.isna(f[c]) else float(f[c]) for c in F.FEATURES])
        ml, rule, an, s = self.m.score_vec(x, fl)
        lv = level([s], [fl["flag_R5"]], [fl["flag_R6"]], self.m.thr, self.m.med)[0]
        nov = bool(lv == "LOW" and an >= self.m.nov_thr)
        ms = (time.perf_counter() - t0) * 1000
        out = {"transaction_id": t["transaction_id"], "risk_score": s, "ml_score": ml, "rule_score": rule,
               "anomaly_score": an, "risk_level": lv, "action": ACTIONS[lv], "novelty": nov, "latency_ms": ms}
        if with_reason:
            r = SimpleNamespace(**{**t, **f, **fl})
            k = self.cust.get(t["customer_id"], {})
            r.home_location = k.get("home_location")
            mm = self.merch.get(t["merchant_id"], {})
            r.merchant_category = mm.get("merchant_category")
            out["alert_reason"] = F.reason(r)
        out["flags"] = fl
        out["features"] = f
        self.n_scored += 1
        return out


def stream_order(gd):
    """Thứ tự luồng: theo thời gian; cùng thời điểm thì theo khách hàng rồi thứ tự trong tệp gốc
    (trùng với cách chấm theo lô)."""
    g = gd.copy()
    g["_dt"] = pd.to_datetime(g.datetime)
    g["_i"] = np.arange(len(g))
    return g.sort_values(["_dt", "customer_id", "_i"], kind="stable").drop(columns=["_dt", "_i"])


def parity_check(model, tables, df_batch=None, progress=None):
    """Cho mọi giao dịch đi qua OnlineScorer (từng giao dịch, đúng thứ tự thời gian) và đối chiếu với chấm theo lô
    bằng cùng một mô hình đóng băng."""
    gd, kh, mc, tb = (tables[n] for n in ["transactions.csv", "customers.csv", "merchants.csv", "devices.csv"])
    if df_batch is None:
        df_batch = F.build(gd, kh, mc, tb)
        df_batch = df_batch.join(F.rules(df_batch, True)[0])
    batch = model.score_frame(df_batch).set_index("transaction_id")
    feats_b = df_batch.set_index("transaction_id")[F.FEATURES]
    sc = OnlineScorer(model, kh, mc, tb)
    rows = stream_order(gd).to_dict("records")
    out, lat, fdiff = [], [], 0.0
    t0 = time.perf_counter()
    for i, t in enumerate(rows):
        r = sc.score(t, with_reason=False)
        out.append((r["transaction_id"], r["risk_score"], r["risk_level"]))
        lat.append(r["latency_ms"])
        fb = feats_b.loc[r["transaction_id"]].fillna(0).values.astype(float)
        fs = np.array([0.0 if pd.isna(r["features"][c]) else float(r["features"][c]) for c in F.FEATURES])
        fdiff = max(fdiff, float(np.max(np.abs(fb - fs))))
        if progress and i % 500 == 0:
            progress(i / len(rows))
    total = time.perf_counter() - t0
    s = pd.DataFrame(out, columns=["transaction_id", "risk_stream", "level_stream"]).set_index("transaction_id")
    j = batch.join(s)
    diff = (j.risk_score - j.risk_stream).abs()
    lat = np.array(lat)
    return {"n": int(len(j)), "max_abs_diff": float(diff.max()), "n_score_mismatch": int((diff > 1e-9).sum()),
            "n_level_mismatch": int((j.risk_level != j.level_stream).sum()), "max_feature_diff": fdiff,
            "latency_mean_ms": float(lat.mean()), "latency_p95_ms": float(np.percentile(lat, 95)),
            "latency_p99_ms": float(np.percentile(lat, 99)), "latency_max_ms": float(lat.max()),
            "total_s": float(total), "model": model.name}
