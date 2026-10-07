# -*- coding: utf-8 -*-
"""
FDTD 濾光片穿透率：特徵擷取 + 視覺化檢查 + 400–1000 nm 覆蓋篩選
資料格式：T.shape = (n_samples, n_wavelengths)，預設 350–1100 nm、1 nm 間隔（751 點），T 為 0–1

主要函式
  make_synthetic   → 合成測試光譜（含 ground truth），可大量產生
  from_array / load_csv / to_uniform_grid / save_dataset / load_dataset
  extract_batch    → summary、peaks、YN、TP（支援多核心平行 + 進度條）
  add_sampling_info→ 原始取樣密度檢查
  evaluate         → PASS/FAIL 與失敗原因（含 peak_ranges、主峰/第二峰範圍、峰形條件）
  select_coverage  → 貪婪挑選覆蓋 400–1000 nm 的組合
  recompute_arrays → 大量資料時只為子集合重算 YN/TP（省記憶體）
  synthetic_check  → 合成資料的偵測正確率
  self_check       → 快速確認本檔所有功能完整（更新後建議先執行）
  plot_profile / plot_gallery / plot_overview / plot_selection（可用 labels 加上設計參數說明）

※ 平行處理（n_jobs != 1）在 Windows / macOS 或 Jupyter 中，請把本檔存成模組
  （例如 spectra_features.py）再 import 使用，並把主程式放在 if __name__ == "__main__": 內。
"""
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, peak_prominences, savgol_filter

__version__ = "2026-10-07.r3"

WL = np.arange(350, 1101, 1.0)

CFG = dict(
    baseline_mode="none",   # 穿透率建議 "none"；可選 "percentile" / "min"
    baseline_pct=1.0,
    sg_window=0,            # FDTD 無隨機雜訊，預設不平滑；有數值震盪可設 5–9
    sg_order=2,
    d1_window=7,            # 一階導數（偵測 shoulder）視窗
    shoulder_prom=0.1,      # shoulder 斜率凹陷門檻（相對最大斜率）
    prom_frac=0.05,         # 峰 prominence 門檻（相對最大值）
    height_frac=0.03,
    min_dist_nm=3,
    merge_frac=0.6,         # 相鄰峰間谷底 >= merge_frac × 較高峰 → 視為同一通帶 ripple 並合併
    sig_area_frac=0.05,     # 有效峰：面積佔比 >= 5%
    sig_height_frac=0.2,    #         且相對高度 >= 0.2（否則為 sidelobe）
    width_ref="prominence", # FWHM / 10% 寬 / 邊緣寬的高度基準：
                            #   "prominence"：局部背景（scipy prominence，扣除漏光與相鄰峰）
                            #   "absolute"  ：固定 T 值 width_ref_value（例如 0 → 半高 = T_peak / 2）
    width_ref_value=0.0,    # width_ref="absolute" 時的基準（絕對 T 單位）
    leak_guard_fwhm=1.0,    # 計算漏光時，每個有效峰排除 [hm_left − g×FWHM, hm_right + g×FWHM]
                            # （且至少涵蓋 10% 高度寬），避免把峰自身的裙擺當成漏光
    band=(400, 1000),
)

CRIT = dict(
    max_sig_peaks=3,        # None → 不限制全域有效峰數
    max_fwhm=60.0,
    min_core=0.5,
    min_in_band=0.85,
    min_peak_T=0.3,
    min_rejection_db=10.0,
    max_ripple=0.2,
    allow_shoulders=False,
    min_raw_pts_fwhm=5,     # 呼叫 add_sampling_info 後才會檢查
    peak_ranges=[],         # 指定波長範圍內的峰數條件（以峰值波長是否落在範圍內計數），例如：
                            #   [dict(range=(800, 900), min=1, max=1)]           800–900 nm 內恰 1 個有效峰
                            #   [dict(range=(700, 900), min=1)]                  700–900 nm 內至少 1 個（不設上限）
                            #   [dict(range=(400, 500), max=0, name="no_blue")]  400–500 nm 內不可有有效峰
                            # 欄位：range=(lo, hi) 必填；min / max 可省略；
                            #       kind="sig"（有效峰，預設）| "all"（含 sidelobe，需傳入 peaks 表）；
                            #       name：失敗原因名稱（預設 "pk800-900"）
    main_peak_range=None,   # 主峰（最高峰）波長範圍，例如 (600, 700)；None 不檢查
    second_peak_range=None, # 第二高有效峰的波長範圍，例如 (800, 900)；None 不檢查
    second_peak_required=True,  # True：沒有第二峰視為 FAIL；False：有第二峰才檢查範圍
    # 以下峰形條件為選填（設定數值才檢查）：
    #   max_shape_factor  ：有效峰 FW10/FWHM 上限（尾巴長度；Gaussian 1.82、Lorentzian 3.0）
    #   max_flat_factor   ：有效峰 FW90/FWHM 上限（峰頂平坦度；Gaussian 0.39、flat-top ≈ 0.8）
    #   max_gauss_nrmse   ：有效峰與 Gaussian 的偏差上限（Gaussian ≈ 0、Lorentzian ≈ 0.09）
    #   min_main_area_frac：主峰面積佔比下限
)


# ============================================================================
# 進度條（有 tqdm 用 tqdm，否則用簡易文字進度）
# ============================================================================
def _fmt_sec(s):
    if not np.isfinite(s):
        return "--:--"
    s = int(s)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class _SimpleBar:
    def __init__(self, total, desc, every=2.0):
        self.total, self.desc, self.every = total, desc, every
        self.n, self.t0, self.last = 0, time.time(), 0.0

    def update(self, k=1):
        self.n += k
        now = time.time()
        if now - self.last >= self.every or self.n >= self.total:
            self.last = now
            el = now - self.t0
            rate = self.n / el if el > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else float("nan")
            print(f"\r{self.desc}: {self.n:,}/{self.total:,} ({100 * self.n / max(self.total, 1):5.1f}%)"
                  f" | {rate:,.0f}/s | elapsed {_fmt_sec(el)} | ETA {_fmt_sec(eta)}",
                  end="", flush=True)

    def close(self):
        print()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _NullBar:
    def update(self, k=1):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _progress(total, desc, enable=True):
    if not enable:
        return _NullBar()
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit="spec", smoothing=0.05)
    except ImportError:
        return _SimpleBar(total, desc)


# ============================================================================
# 工具函式
# ============================================================================
def _idx2wl(x, wl):
    return np.interp(x, np.arange(len(wl)), wl)


def _runs(mask, wl):
    """mask 為 True 的連續區段 → [(start_wl, end_wl), ...]"""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(float(wl[s]), float(wl[e])) for s, e in zip(starts, ends)]


def preprocess(t, cfg=CFG):
    cfg = {**CFG, **cfg}            # 舊版 cfg 缺少的參數以目前預設值補齊
    t = np.asarray(t, float)
    w = cfg["sg_window"]
    ts = savgol_filter(t, w | 1, cfg["sg_order"]) if w and w >= 3 else t.copy()
    if cfg["baseline_mode"] == "percentile":
        base = float(np.percentile(ts, cfg["baseline_pct"]))
    elif cfg["baseline_mode"] == "min":
        base = float(ts.min())
    else:
        base = 0.0
    y = np.clip(ts - base, 0, None)
    return y, base


def _merge_ripples(yn, pk, merge_frac):
    """把同一通帶內被 ripple 切開的峰合併，回傳代表峰與各組成員"""
    groups = [[pk[0]]]
    for p in pk[1:]:
        q = groups[-1][-1]
        if yn[q:p + 1].min() >= merge_frac * max(yn[q], yn[p]):
            groups[-1].append(p)
        else:
            groups.append([p])
    reps = np.array([g[int(np.argmax(yn[g]))] for g in groups])
    return reps, groups


def _fmt_val(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def _label_text(sid, srow, labels):
    """
    依 labels 產生圖標題附加說明：
      None              → 不加
      pd.DataFrame      → 該 id 那一列，格式 "col=val, col=val"（例如設計參數表）
      pd.Series / dict  → id → 文字
      callable          → f(id, summary_row) → 文字
    """
    if labels is None:
        return ""
    try:
        if callable(labels) and not isinstance(labels, (pd.DataFrame, pd.Series)):
            t = labels(sid, srow)
        elif isinstance(labels, pd.DataFrame):
            key = sid if sid in labels.index else str(sid)
            if key not in labels.index:
                return ""
            r = labels.loc[key]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            t = ", ".join(f"{k}={_fmt_val(v)}" for k, v in r.items())
        elif isinstance(labels, (pd.Series, dict)):
            t = labels.get(sid, labels.get(str(sid), ""))
        else:
            t = ""
    except Exception:
        t = ""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    return str(t)


# ============================================================================
# 合成測試資料
# ============================================================================
SYN_MIX = dict(
    single_gauss=0.30, single_lorentz=0.15, flattop_ripple=0.10, double=0.12,
    triple=0.08, shoulder=0.08, sidelobe=0.07, broad=0.05, ringing=0.03, T_gt1=0.02,
)
_SYN_TRUE_N = dict(double=2, triple=3, shoulder=2)


def _g(c, fw, W):
    return np.exp(-4 * np.log(2) * (W - c) ** 2 / fw ** 2)


def _lor(c, fw, W):
    return 1 / (1 + 4 * (W - c) ** 2 / fw ** 2)


def _flat(c, w, W, n=6):
    return 1 / (1 + np.abs(2 * (W - c) / w) ** (2 * n))


def _syn_type(name, m, rng, W, floor_level):
    """產生 m 條指定類型的光譜（向量化），回傳 Y (m, n_wl)、真實中心、真實 FWHM"""
    def U(a, b):
        return rng.uniform(a, b, (m, 1))

    floor = floor_level * U(0.5, 1.5) + 0.01 * np.sin(W / 7 + U(0, 2 * np.pi))

    if name == "single_gauss":
        c, fw, A = U(390, 1010), U(10, 60), U(0.5, 0.95)
        Y, C, F = A * _g(c, fw, W), [c], [fw]
    elif name == "single_lorentz":
        c, fw, A = U(400, 1000), U(8, 40), U(0.5, 0.95)
        Y, C, F = A * _lor(c, fw, W), [c], [fw]
    elif name == "flattop_ripple":
        c, w, A = U(420, 980), U(30, 80), U(0.6, 0.9)
        amp, per = U(0.02, 0.1), U(8, 20)
        Y = A * _flat(c, w, W) * (1 + amp * np.sin(2 * np.pi * (W - c) / per))
        C, F = [c], [w]
    elif name == "double":
        c1 = U(400, 800)
        c2 = np.minimum(c1 + U(80, 250), 1000)
        f1, f2 = U(15, 50), U(15, 50)
        A1 = U(0.5, 0.9)
        A2 = A1 * U(0.5, 1.0)
        Y, C, F = A1 * _g(c1, f1, W) + A2 * _g(c2, f2, W), [c1, c2], [f1, f2]
    elif name == "triple":
        c1 = U(400, 560)
        c2 = c1 + U(130, 220)
        c3 = np.minimum(c2 + U(130, 220), 1000)
        fs = [U(15, 45) for _ in range(3)]
        As = [U(0.4, 0.85) for _ in range(3)]
        cs = [c1, c2, c3]
        Y = sum(a * _g(c, f, W) for a, c, f in zip(As, cs, fs))
        C, F = cs, fs
    elif name == "shoulder":
        c, fw, A = U(420, 950), U(20, 40), U(0.6, 0.9)
        c2 = c + fw * U(0.6, 1.0) * rng.choice([-1.0, 1.0], (m, 1))
        f2 = fw * U(0.7, 1.0)
        Y = A * _g(c, fw, W) + A * U(0.3, 0.5) * _g(c2, f2, W)
        C, F = [c, c2], [fw, f2]
    elif name == "sidelobe":
        c, fw, A = U(450, 950), U(15, 40), U(0.6, 0.9)
        cs = np.where(c > 700, c - U(150, 300), c + U(150, 300))
        Y = A * _g(c, fw, W) + A * U(0.08, 0.18) * _g(cs, U(30, 60), W)
        C, F = [c], [fw]
    elif name == "broad":
        c, fw, A = U(500, 850), U(150, 350), U(0.4, 0.8)
        Y, C, F = A * _g(c, fw, W), [c], [fw]
    elif name == "ringing":   # 模擬時間不足造成的振盪尾巴，可能出現 T < 0
        c, fw, A = U(450, 950), U(15, 40), U(0.6, 0.9)
        Y = A * _g(c, fw, W) + A * U(0.05, 0.12) * np.cos(2 * np.pi * (W - c) / U(8, 20)) \
            * np.exp(-np.abs(W - c) / U(60, 150))
        C, F = [c], [fw]
    elif name == "T_gt1":     # 數值問題：T > 1
        c, fw, A = U(420, 980), U(15, 40), U(1.01, 1.08)
        Y, C, F = A * _g(c, fw, W), [c], [fw]
    else:
        raise ValueError(f"未知類型: {name}")

    Y = Y + floor
    return Y, np.round(np.hstack(C), 1).tolist(), np.round(np.hstack(F), 1).tolist()


def make_synthetic(n=1000, wl=WL, mix=None, seed=0, floor_level=0.02,
                   dtype=np.float32, gen_chunk=20000, progress=True):
    """
    產生 n 條合成穿透率光譜。
    mix：各類型比例，預設 SYN_MIX；可傳 {"single_gauss": 1} 只產生單一類型
    回傳 T (n, len(wl))、ids、meta（type / true_n_peaks / true_centers / true_fwhms，index = ids）
    記憶體估計：float32 時約 n × len(wl) × 4 bytes（1M × 751 ≈ 3 GB）
    """
    rng = np.random.default_rng(seed)
    mix = dict(mix or SYN_MIX)
    names = list(mix)
    p = np.array([mix[k] for k in names], float)
    p /= p.sum()
    lab = rng.choice(len(names), size=n, p=p)
    W = np.asarray(wl, float)[None, :]
    T = np.empty((n, W.shape[1]), dtype)
    centers, fwhms = [None] * n, [None] * n

    with _progress(n, "synthesize", progress) as bar:
        for k, name in enumerate(names):
            idx = np.flatnonzero(lab == k)
            for a in range(0, len(idx), gen_chunk):
                sub = idx[a:a + gen_chunk]
                Y, C, F = _syn_type(name, len(sub), rng, W, floor_level)
                T[sub] = Y
                for j, i in enumerate(sub):
                    centers[i], fwhms[i] = C[j], F[j]
                bar.update(len(sub))

    ids = [f"syn_{i:07d}" for i in range(n)]
    type_arr = np.array(names)[lab]
    meta = pd.DataFrame({
        "type": type_arr,
        "true_n_peaks": [_SYN_TRUE_N.get(t, 1) for t in type_arr],
        "true_centers": centers,
        "true_fwhms": fwhms,
    }, index=pd.Index(ids, name="id"))
    return T, ids, meta


def synthetic_check(summary, meta):
    """比較擷取結果與 ground truth"""
    df = summary.join(meta, how="inner")
    out = {
        "n_sig_peaks": pd.crosstab(df["type"], df["n_sig_peaks"].fillna(0).astype(int),
                                   margins=True),
        "shoulder": pd.crosstab(df["type"], df["n_shoulders"].fillna(0) > 0,
                                normalize="index").round(3),
    }
    single = df[df["type"].isin(["single_gauss", "single_lorentz", "T_gt1", "flattop_ripple"])]
    tc = single["true_centers"].str[0]
    tf = single["true_fwhms"].str[0]
    out["accuracy"] = pd.DataFrame({
        "center_abs_err_nm": (single["main_peak_wl"] - tc).abs().groupby(single["type"]).median(),
        "fwhm_rel_err": ((single["main_fwhm"] - tf) / tf).abs().groupby(single["type"]).median(),
    }).round(3)
    return out


# ============================================================================
# 資料讀取 / 格點統一 / 儲存
# ============================================================================
_C = 299792458.0


def _to_nm(x, unit="auto"):
    """把 FDTD 輸出的波長/頻率軸轉成 nm。unit: 'auto' | 'm' | 'um' | 'nm' | 'Hz'"""
    x = np.asarray(x, float)
    if unit == "auto":
        mx = np.nanmax(x)
        unit = "Hz" if mx > 1e12 else "m" if mx < 1e-3 else "um" if mx < 50 else "nm"
    if unit == "Hz":
        return _C / x * 1e9
    return x * {"m": 1e9, "um": 1e3, "nm": 1.0}[unit]


def check_grid(wl, T=None, tol=1e-6):
    """檢查波長格點與 T 的 shape / 單位（不產生大型暫存陣列）"""
    wl = np.asarray(wl, float)
    d = np.diff(wl)
    if not (np.all(d > 0) and np.allclose(d, d[0], atol=tol)):
        raise ValueError("波長需為等間隔且遞增；請先用 to_uniform_grid() 內插")
    if T is not None:
        if T.ndim != 2 or T.shape[1] != len(wl):
            raise ValueError(f"T.shape 應為 (n_samples, {len(wl)})，目前為 {T.shape}")
        if np.isnan(T.sum(dtype=np.float64)):
            warnings.warn(f"T 含 NaN（{int(np.isnan(T).any(1).sum())} 條光譜），結果可能異常")
            mx = np.nanmax(T)
        else:
            mx = T.max()
        if mx > 1.5:
            warnings.warn("T 最大值 > 1.5，可能是百分比單位，請先 /100")


def to_uniform_grid(wl_raw, T_raw, wl_new=WL, wl_unit="auto", max_gap_warn=2.0):
    """
    FDTD 常在頻率上等間隔取樣 → 波長不等間隔且遞減。
    排序後線性內插到 wl_new，回傳 (n_samples, len(wl_new)) 與排序後原始波長。
    """
    wl_raw = _to_nm(wl_raw, wl_unit)
    T_raw = np.atleast_2d(np.asarray(T_raw, float))
    if T_raw.shape[1] != len(wl_raw):
        if T_raw.shape[0] == len(wl_raw):
            T_raw = T_raw.T
        else:
            raise ValueError(f"T_raw.shape {T_raw.shape} 與波長點數 {len(wl_raw)} 不符")
    order = np.argsort(wl_raw)
    wl_s, T_s = wl_raw[order], T_raw[:, order]

    if wl_new[0] < wl_s[0] - 1e-9 or wl_new[-1] > wl_s[-1] + 1e-9:
        warnings.warn(f"目標範圍 {wl_new[0]:.0f}–{wl_new[-1]:.0f} nm 超出原始資料 "
                      f"{wl_s[0]:.1f}–{wl_s[-1]:.1f} nm，範圍外以端點值填補")
    inr = (wl_s >= wl_new[0]) & (wl_s <= wl_new[-1])
    if inr.sum() > 1:
        gaps = np.diff(wl_s[inr])
        if gaps.max() > max_gap_warn:
            k = int(np.argmax(gaps))
            warnings.warn(f"原始取樣在 {wl_s[inr][k]:.0f} nm 附近間隔 {gaps.max():.1f} nm，"
                          f"窄峰的峰高/FWHM 可能失真（可用 add_sampling_info 檢查）")
    return np.vstack([np.interp(wl_new, wl_s, t) for t in T_s]), wl_s


def load_csv(path, layout="wl_rows", wl_new=WL, wl_unit="auto"):
    """
    layout="wl_rows"    ：第一欄為波長/頻率，其餘每欄一條光譜（常見 FDTD 匯出）
    layout="sample_rows"：第一欄為 id，header 為波長/頻率，每列一條光譜
    """
    df = pd.read_csv(path)
    if layout == "wl_rows":
        x = df.iloc[:, 0].to_numpy(float)
        data = df.iloc[:, 1:]
        T_raw, ids = data.T.to_numpy(float), [str(c) for c in data.columns]
    elif layout == "sample_rows":
        ids = df.iloc[:, 0].astype(str).tolist()
        data = df.iloc[:, 1:]
        x = data.columns.astype(float).to_numpy()
        T_raw = data.to_numpy(float)
    else:
        raise ValueError("layout 需為 'wl_rows' 或 'sample_rows'")
    T, wl_raw = to_uniform_grid(x, T_raw, wl_new, wl_unit)
    return T, ids, wl_raw


def from_array(T_raw, wl_raw=None, ids=None, wl_new=WL, wl_unit="auto", percent="auto"):
    """
    numpy array → 標準格式 (n_samples, len(wl_new))
      T_raw  : (n_samples, n_wl) 或 (n_wl, n_samples)；1D 視為單條；可為 memmap
      wl_raw : None → 假設 T_raw 已在 wl_new 格點上（不複製資料）
               否則為原始波長/頻率軸（m / µm / nm / Hz 自動判斷），會排序並內插
      ids    : None → "0", "1", ...（統一轉成 str，方便與參數表對齊）
      percent: "auto"（最大值 > 1.5 視為 %）| True | False
    回傳 T, ids(list[str]), wl_s（排序後原始波長 nm；wl_raw=None 時為 wl_new）
    """
    T_raw = np.asarray(T_raw)
    if T_raw.ndim == 1:
        T_raw = T_raw[None, :]
    if wl_raw is None:
        n_wl = len(wl_new)
        if T_raw.shape[1] != n_wl:
            if T_raw.shape[0] == n_wl:
                T_raw = T_raw.T
                print(f"[from_array] 偵測到 (n_wl, n_samples)，已轉置為 {T_raw.shape}")
            else:
                raise ValueError(f"T_raw.shape {T_raw.shape} 與 wl_new 點數 {n_wl} 不符；"
                                 f"若格點不同請提供 wl_raw")
        T, wl_s = T_raw, np.asarray(wl_new, float)
    else:
        T, wl_s = to_uniform_grid(wl_raw, T_raw, wl_new, wl_unit)

    if percent is True or (percent == "auto" and np.nanmax(T) > 1.5):
        T = np.asarray(T, np.float32) / 100.0
        print("[from_array] 視為百分比單位，已 /100")

    ids = [str(i) for i in ids] if ids is not None else [str(i) for i in range(len(T))]
    if len(ids) != len(T):
        raise ValueError(f"ids 數量 {len(ids)} 與光譜數 {len(T)} 不符")
    return T, ids, wl_s


def save_dataset(path, T, ids, wl=WL, wl_raw=None, params=None):
    """光譜存 npz；設計參數表（index = ids）另存 <name>_params.parquet（需 pyarrow）"""
    path = Path(path)
    check_grid(wl, T)
    np.savez_compressed(path, T=np.asarray(T, np.float32), ids=np.asarray(ids, dtype=str),
                        wl=np.asarray(wl, float),
                        wl_raw=np.asarray(wl_raw if wl_raw is not None else [], float))
    if params is not None:
        params.to_parquet(path.with_name(path.stem + "_params.parquet"))


def load_dataset(path):
    """回傳 T (float32), ids, wl, wl_raw, params"""
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    T, ids, wl = d["T"], d["ids"].tolist(), d["wl"]
    wl_raw = d["wl_raw"] if d["wl_raw"].size else None
    pq = path.with_name(path.stem + "_params.parquet")
    params = pd.read_parquet(pq) if pq.exists() else None
    check_grid(wl, T)
    return T, ids, wl, wl_raw, params


def add_sampling_info(summary, peaks, wl_raw):
    """以原始（內插前）波長點檢查每個峰 FWHM 內的取樣點數（假設所有樣本共用同一組 wl_raw）"""
    wl_s = np.sort(np.asarray(wl_raw, float))
    if len(peaks):
        lo = np.searchsorted(wl_s, peaks["hm_left"].to_numpy(), side="left")
        hi = np.searchsorted(wl_s, peaks["hm_right"].to_numpy(), side="right")
        peaks["n_raw_pts_fwhm"] = hi - lo
        mn = peaks[peaks["is_sig"]].groupby("id")["n_raw_pts_fwhm"].min()
        summary["min_raw_pts_fwhm"] = mn.reindex(summary.index)
    return summary, peaks


# ============================================================================
# 單條光譜特徵擷取
# ============================================================================
def extract_features(t_raw, wl=WL, cfg=CFG):
    cfg = {**CFG, **cfg}            # 舊版 cfg 缺少的參數以目前預設值補齊
    t_raw = np.asarray(t_raw, float)
    y, base = preprocess(t_raw, cfg)
    dx = float(wl[1] - wl[0])
    ymax, ysum = y.max(), y.sum()
    flags = dict(n_T_gt1=int((t_raw > 1.001).sum()), n_T_neg=int((t_raw < -1e-3).sum()))
    tp = y + base
    if ymax <= 0:
        return {"valid": False, **flags}, pd.DataFrame(), np.zeros_like(y), tp
    yn = y / ymax

    # ---------------- 整體特徵 ----------------
    lo, hi = cfg["band"]
    inb = (wl >= lo) & (wl <= hi)
    cdf = np.cumsum(y) / ysum
    wl05, wl50, wl95 = np.interp([0.05, 0.5, 0.95], cdf, wl)
    centroid = (wl * y).sum() / ysum
    rms_width = max(np.sqrt((((wl - centroid) ** 2) * y).sum() / ysum), 1e-9)
    hm_runs = _runs(yn >= 0.5, wl)

    s = dict(
        valid=True, **flags, baseline=base,
        T_peak=float(tp.max()),
        peak_wl_max=float(wl[np.argmax(tp)]),
        T_mean_band=float(tp[inb].mean()),
        total_area=ysum * dx,
        centroid=centroid, median_wl=wl50, rms_width=rms_width,
        skewness=((((wl - centroid) ** 3) * y).sum() / ysum) / rms_width ** 3,
        wl05=wl05, wl95=wl95, span90=wl95 - wl05,
        eq_width=ysum * dx / ymax,
        in_band_frac=y[inb].sum() / ysum,
        below_band_frac=y[wl < lo].sum() / ysum,
        above_band_frac=y[wl > hi].sum() / ysum,
        n_hm_lobes=len(hm_runs),
        hm_cover_nm=(yn[inb] >= 0.5).sum() * dx,
        hm_intervals=hm_runs,
    )

    # ---------------- 找峰 + 合併 ripple ----------------
    pk_all, _ = find_peaks(yn, prominence=cfg["prom_frac"], height=cfg["height_frac"],
                           distance=max(1, int(round(cfg["min_dist_nm"] / dx))))
    if len(pk_all) == 0:
        s.update(n_peaks=0, n_sig_peaks=0)
        return s, pd.DataFrame(), yn, tp
    pk, groups = _merge_ripples(yn, pk_all, cfg["merge_frac"])

    bounds = [0] + [a + int(np.argmin(yn[a:b + 1])) for a, b in zip(pk[:-1], pk[1:])] + [len(y)]
    prom = peak_prominences(yn, pk)[0]

    # ---- 寬度的高度基準 ----
    # prominence：level = 峰高 − rel × (峰高 − 局部背景)，局部背景由 scipy prominence 決定
    # absolute  ：level = 峰高 − rel × (峰高 − 固定值)，搜尋範圍限制在該峰的 valley 區段內
    if cfg.get("width_ref", "prominence") == "absolute":
        r0 = (cfg.get("width_ref_value", 0.0) - base) / ymax           # 換成 yn 單位
        lb = np.array(bounds[:-1], dtype=np.intp)
        rb = np.minimum(np.array(bounds[1:], dtype=np.intp), len(y) - 1)
        pdata = (np.maximum(yn[pk] - r0, 1e-12).astype(float), lb, rb)
    else:
        pdata = None
    w50 = peak_widths(yn, pk, rel_height=0.5, prominence_data=pdata)
    w_hi = peak_widths(yn, pk, rel_height=0.1, prominence_data=pdata)   # 90% 高度
    w_lo = peak_widths(yn, pk, rel_height=0.9, prominence_data=pdata)   # 10% 高度
    if pdata is not None:
        # 在區段邊界仍高於半高 → 與相鄰峰重疊，半高交點不存在，寬度被截在 valley
        unres = (yn[pdata[1]] > w50[1]) | (yn[pdata[2]] > w50[1])
    else:
        unres = np.zeros(len(pk), bool)

    def to_wl(x):
        return float(_idx2wl(x, wl))

    def to_T(v):
        return float(v * ymax + base)

    rows = []
    for i, p in enumerate(pk):
        seg = slice(bounds[i], bounds[i + 1])
        li, ri = int(np.ceil(w50[2][i])), int(np.floor(w50[3][i]))
        fwhm = w50[0][i] * dx
        win = yn[li:ri + 1]
        mins = find_peaks(-win)[0] if len(win) > 2 else np.array([], int)
        ripple = float((yn[p] - win[mins].min()) / yn[p]) if len(mins) else 0.0
        hm_l, hm_r = to_wl(w50[2][i]), to_wl(w50[3][i])
        # 與同中心 / 峰高 / FWHM 的 Gaussian 比較（不做擬合，大量資料也快）
        # 視窗 = 中心 ± 1.5 FWHM（Gaussian 在此已 < 0.3%，Lorentzian 仍約 10%）
        ref = yn[p] - (prom[i] if pdata is None else pdata[0][i])     # 局部背景（yn 單位）
        amp = max(yn[p] - ref, 1e-12)
        c_mid = (hm_l + hm_r) / 2
        m_g = (wl >= c_mid - 1.5 * fwhm) & (wl <= c_mid + 1.5 * fwhm)
        g_model = ref + amp * np.exp(-4 * np.log(2) * (wl[m_g] - c_mid) ** 2 / max(fwhm, 1e-9) ** 2)
        gauss_nrmse = float(np.sqrt(np.mean((yn[m_g] - g_model) ** 2)) / amp) if m_g.any() else np.nan
        rows.append(dict(
            gauss_nrmse=gauss_nrmse,        # 越小越像 Gaussian；理想 ≈ 0，同 FWHM 的 Lorentzian ≈ 0.09
            peak_wl=float(wl[p]), T_peak=float(tp[p]),
            rel_height=float(yn[p]), prominence=float(prom[i]),
            fwhm=fwhm, hm_left=hm_l, hm_right=hm_r, hm_level=to_T(w50[1][i]),
            fw10=w_lo[0][i] * dx, fw10_left=to_wl(w_lo[2][i]), fw10_right=to_wl(w_lo[3][i]),
            fw10_level=to_T(w_lo[1][i]),
            Q=wl[p] / max(fwhm, 1e-9),
            shape_factor=w_lo[0][i] / max(w50[0][i], 1e-9),   # FW10/FWHM：尾巴長度（Gaussian 1.82、Lorentzian 3.0）
            flat_factor=w_hi[0][i] / max(w50[0][i], 1e-9),    # FW90/FWHM：峰頂平坦度（Gaussian 0.39、flat-top ≈ 0.8）
            asymmetry=(hm_r - wl[p]) / max(wl[p] - hm_l, 1e-9),
            edge_l=(w_hi[2][i] - w_lo[2][i]) * dx,
            edge_r=(w_lo[3][i] - w_hi[3][i]) * dx,
            hm_unresolved=bool(unres[i]),   # True：absolute 基準下半高交點落在相鄰峰區段外
            ripple=ripple,
            n_ripple_peaks=len(groups[i]) - 1,
            ripple_wls=[float(wl[q]) for q in groups[i] if q != p],
            area_frac=y[seg].sum() / ysum,
            fwhm_area_frac=y[li:ri + 1].sum() / ysum,
            seg_lo=float(wl[bounds[i]]), seg_hi=float(wl[bounds[i + 1] - 1]),
        ))
    pdf = pd.DataFrame(rows)
    pdf["is_sig"] = (pdf.area_frac >= cfg["sig_area_frac"]) & (pdf.rel_height >= cfg["sig_height_frac"])

    sig = pdf[pdf.is_sig] if pdf.is_sig.any() else pdf.loc[[pdf.rel_height.idxmax()]]
    sig = sig.sort_values("peak_wl")

    # ---------------- 通帶 / 漏光 ----------------
    core = np.zeros(len(y), bool)
    passband = np.zeros(len(y), bool)
    g = cfg.get("leak_guard_fwhm", 1.0)
    for hl, hr, fw, fl, fr in sig[["hm_left", "hm_right", "fwhm",
                                   "fw10_left", "fw10_right"]].to_numpy():
        core |= (wl >= hl) & (wl <= hr)
        passband |= (wl >= min(fl, hl - g * fw)) & (wl <= max(fr, hr + g * fw))
    s["leak_excl"] = _runs(passband, wl)          # 漏光計算時被排除的範圍（作圖用）
    stop = inb & ~passband
    if stop.any():
        k = np.where(stop)[0][np.argmax(tp[stop])]
        leak_max, leak_wl, leak_mean = float(tp[k]), float(wl[k]), float(tp[stop].mean())
    else:
        leak_max = leak_wl = leak_mean = np.nan
    rejection_db = 10 * np.log10(s["T_peak"] / max(leak_max, 1e-6)) if np.isfinite(leak_max) else np.nan

    # ---------------- shoulder：斜率在單調邊緣上出現凹陷 ----------------
    d1 = savgol_filter(yn, cfg["d1_window"] | 1, 2, deriv=1)
    a = np.abs(d1).max()
    m_up = find_peaks(-d1, prominence=cfg["shoulder_prom"] * a)[0]
    m_dn = find_peaks(d1, prominence=cfg["shoulder_prom"] * a)[0]
    sh = np.concatenate([m_up[d1[m_up] > 0.02 * a], m_dn[d1[m_dn] < -0.02 * a]]).astype(int)
    sh = sh[yn[sh] > 0.2]
    sh = np.array([c for c in sh if np.min(np.abs(c - pk_all)) > 2], dtype=int)

    # ---------------- 峰彙總 ----------------
    main = pdf.loc[pdf.rel_height.idxmax()]
    # 第二高峰：主峰以外、有效峰中 rel_height 最高者
    others = pdf[pdf.is_sig].drop(index=main.name, errors="ignore").sort_values("rel_height",
                                                                                ascending=False)
    if len(others):
        sec = others.iloc[0]
        second = dict(second_peak_wl=sec.peak_wl, second_rel_height=sec.rel_height,
                      second_fwhm=sec.fwhm, second_area_frac=sec.area_frac)
    else:
        second = dict(second_peak_wl=np.nan, second_rel_height=np.nan,
                      second_fwhm=np.nan, second_area_frac=np.nan)
    if len(sig) >= 2:
        sep = np.diff(sig.peak_wl.to_numpy())
        mean_fw = (sig.fwhm.to_numpy()[:-1] + sig.fwhm.to_numpy()[1:]) / 2
        min_sep, min_res = sep.min(), (sep / mean_fw).min()
    else:
        min_sep = min_res = np.nan

    s.update(
        n_peaks=len(pdf),
        n_sig_peaks=int(pdf.is_sig.sum()),
        n_sidelobes=int((~pdf.is_sig).sum()),
        n_shoulders=len(sh),
        shoulder_wls=[float(wl[c]) for c in sh],
        sig_peak_wls=sig.peak_wl.tolist(),
        sig_fwhms=sig.fwhm.round(1).tolist(),
        sig_area_fracs=sig.area_frac.round(3).tolist(),
        sig_T_peaks=sig.T_peak.round(3).tolist(),
        main_peak_wl=main.peak_wl, main_fwhm=main.fwhm,
        main_area_frac=main.area_frac, main_Q=main.Q,
        **second,
        max_fwhm_sig=sig.fwhm.max(),
        max_shape_factor=sig.shape_factor.max(),
        max_edge=float(sig[["edge_l", "edge_r"]].to_numpy().max()),
        max_ripple=sig.ripple.max(),
        main_shape_factor=main.shape_factor,
        main_flat_factor=main.flat_factor,
        max_flat_factor=sig.flat_factor.max(),
        main_gauss_nrmse=main.gauss_nrmse,
        max_gauss_nrmse=sig.gauss_nrmse.max(),
        n_hm_unresolved=int(pdf.loc[pdf.is_sig, "hm_unresolved"].sum()),
        core_frac=y[core].sum() / ysum,
        leak_max_band=leak_max, leak_wl=leak_wl, leak_mean_band=leak_mean,
        leak_max_oob=float(tp[~inb].max()),
        rejection_db=rejection_db,
        min_peak_sep=min_sep, min_resolution=min_res,
    )
    return s, pdf, yn, tp


# ============================================================================
# 批次處理（分塊 + 多核心 + 進度條）
# ============================================================================
def _extract_chunk(args):
    """worker：處理一塊光譜（需為模組層級函式才能被多行程 pickle）"""
    T_chunk, ids_chunk, wl, cfg, keep_arrays = args
    S, P = [], []
    shape = (len(T_chunk), len(wl))
    YN = np.zeros(shape, np.float32) if keep_arrays else None
    TP = np.zeros(shape, np.float32) if keep_arrays else None
    for k, (sid, t) in enumerate(zip(ids_chunk, T_chunk)):
        s, p, yn, tp = extract_features(t, wl, cfg)
        s["id"] = sid
        S.append(s)
        if keep_arrays:
            YN[k], TP[k] = yn, tp
        if len(p):
            p.insert(0, "id", sid)
            P.append(p)
    P = pd.concat(P, ignore_index=True) if P else None
    return S, P, YN, TP


def extract_batch(T, ids=None, wl=WL, cfg=CFG, n_jobs=1, chunk_size=2000,
                  progress=True, keep_arrays=True):
    """
    T          : (n_samples, n_wavelengths)，float32 / float64 皆可（不會整批複製）
    ids        : None → 0..n-1
    n_jobs     : 1 = 單核心；-1 = 全部核心；>1 = 指定核心數
    chunk_size : 每個工作單位的光譜數（1k–5k 通常合適）
    keep_arrays: False 時不回傳 YN / TP（1M 條可省約 6 GB），之後用 recompute_arrays 補算子集合
    """
    T = np.asarray(T)
    if T.ndim == 1:
        T = T[None, :]
    check_grid(wl, T)
    n = len(T)
    ids = list(ids) if ids is not None else list(range(n))
    if len(ids) != n:
        raise ValueError(f"ids 數量 {len(ids)} 與光譜數 {n} 不符")
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1

    starts = list(range(0, n, chunk_size))
    tasks = ((T[a:a + chunk_size], ids[a:a + chunk_size], wl, cfg, keep_arrays) for a in starts)
    results = [None] * len(starts)
    YN = np.empty((n, len(wl)), np.float32) if keep_arrays else None
    TP = np.empty((n, len(wl)), np.float32) if keep_arrays else None

    def collect(j, r, bar):
        S, P, yn, tp = r
        if keep_arrays:
            a = starts[j]
            YN[a:a + len(S)], TP[a:a + len(S)] = yn, tp
        results[j] = (S, P)
        bar.update(len(S))

    t0 = time.time()
    with _progress(n, f"extract (n_jobs={n_jobs})", progress) as bar:
        if n_jobs == 1:
            for j, task in enumerate(tasks):
                collect(j, _extract_chunk(task), bar)
        else:
            from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
            it = enumerate(tasks)
            pending = {}
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                def submit_next():
                    try:
                        j, task = next(it)
                    except StopIteration:
                        return False
                    pending[ex.submit(_extract_chunk, task)] = j
                    return True

                for _ in range(2 * n_jobs):          # 限制同時在途的工作量，避免記憶體暴增
                    if not submit_next():
                        break
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        collect(pending.pop(f), f.result(), bar)
                        submit_next()

    summary = pd.DataFrame([s for S, _ in results for s in S]).set_index("id")
    P = [p for _, p in results if p is not None]
    peaks = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    if progress:
        el = time.time() - t0
        print(f"完成 {n:,} 條，耗時 {_fmt_sec(el)}（{n / max(el, 1e-9):,.0f} 條/秒），"
              f"峰總數 {len(peaks):,}")
    return summary, peaks, YN, TP


def recompute_arrays(T, idx, wl=WL, cfg=CFG):
    """只為子集合（例如 PASS 候選）重算 YN / TP，不做找峰，速度快"""
    idx = np.asarray(idx, int)
    YN = np.zeros((len(idx), len(wl)), np.float32)
    TP = np.zeros((len(idx), len(wl)), np.float32)
    for k, i in enumerate(idx):
        y, base = preprocess(T[i], cfg)
        m = y.max()
        if m > 0:
            YN[k] = y / m
        TP[k] = y + base
    return YN, TP


# ============================================================================
# 條件判定
# ============================================================================
def count_peaks_in_range(summary, lo, hi, peaks=None, kind="sig"):
    """
    各光譜峰值波長落在 [lo, hi] nm 內的峰數（回傳 Series，index 同 summary）
      kind="sig"：只算有效峰（取自 summary["sig_peak_wls"]）
      kind="all"：含 sidelobe（需提供 peaks 表）
    """
    if kind == "sig":
        if "sig_peak_wls" not in summary:
            return pd.Series(0, index=summary.index)
        w = pd.to_numeric(summary["sig_peak_wls"].explode(), errors="coerce")
        cnt = ((w >= lo) & (w <= hi)).groupby(level=0, sort=False).sum()
    elif kind == "all":
        if peaks is None or not len(peaks):
            raise ValueError('kind="all" 需提供 peaks 表：evaluate(summary, crit, peaks)')
        cnt = peaks.loc[peaks["peak_wl"].between(lo, hi)].groupby("id", sort=False).size()
    else:
        raise ValueError('kind 需為 "sig" 或 "all"')
    return cnt.reindex(summary.index, fill_value=0).astype(int)


# 選填的峰形條件：(結果欄名, summary 欄位, crit 鍵, 比較方向)
_SHAPE_CRITERIA = [
    ("shape", "max_shape_factor", "max_shape_factor", "<="),
    ("flat", "max_flat_factor", "max_flat_factor", "<="),
    ("gauss", "max_gauss_nrmse", "max_gauss_nrmse", "<="),
    ("dominance", "main_area_frac", "min_main_area_frac", ">="),
]


def _need(s, colname, key):
    if colname not in s:
        raise KeyError(f"summary 缺少欄位 '{colname}'（可能是舊版的擷取結果），"
                       f"無法檢查 {key}；請以目前版本重新執行 extract_batch / run_fdtd")


def evaluate(summary, crit=CRIT, peaks=None):
    crit = {**CRIT, **crit}         # 舊版 crit 缺少的條件以目前預設值補齊
    s = summary

    def col(name):
        return s[name] if name in s else pd.Series(np.nan, index=s.index)

    chk = pd.DataFrame({
        "peaks": (col("n_sig_peaks").between(1, crit["max_sig_peaks"])
                  if crit.get("max_sig_peaks") is not None
                  else pd.Series(True, index=s.index)),          # None → 不限制全域峰數
        "fwhm": col("max_fwhm_sig") <= crit["max_fwhm"],
        "core": col("core_frac") >= crit["min_core"],
        "in_band": col("in_band_frac") >= crit["min_in_band"],
        "peak_T": col("T_peak") >= crit["min_peak_T"],
        "rejection": col("rejection_db") >= crit["min_rejection_db"],
        "ripple": col("max_ripple") <= crit["max_ripple"],
        "shoulder": (col("n_shoulders") == 0) | bool(crit["allow_shoulders"]),
        "T_range": (col("n_T_gt1") == 0) & (col("n_T_neg") == 0),
    }, index=s.index).astype(bool)

    if "min_raw_pts_fwhm" in s:     # 有呼叫 add_sampling_info 時才檢查
        chk["sampling"] = (s["min_raw_pts_fwhm"] >= crit.get("min_raw_pts_fwhm", 5)).to_numpy()

    # ---- 指定波長範圍內的峰數 ----
    for r in crit.get("peak_ranges") or []:
        lo, hi = r["range"]
        cnt = count_peaks_in_range(s, lo, hi, peaks, r.get("kind", "sig"))
        ok = pd.Series(True, index=s.index)
        if r.get("min") is not None:
            ok &= cnt >= r["min"]
        if r.get("max") is not None:
            ok &= cnt <= r["max"]
        chk[r.get("name") or f"pk{lo:.0f}-{hi:.0f}"] = ok.to_numpy()

    # ---- 主峰 / 第二高峰所在波長範圍 ----
    rng = crit.get("main_peak_range")
    if rng is not None:
        lo, hi = rng
        chk[f"main{lo:.0f}-{hi:.0f}"] = col("main_peak_wl").between(lo, hi).to_numpy()
    rng = crit.get("second_peak_range")
    if rng is not None:
        _need(s, "second_peak_wl", "second_peak_range")
        lo, hi = rng
        v = col("second_peak_wl")
        ok = v.between(lo, hi)
        if not crit.get("second_peak_required", True):
            ok = ok | v.isna()                                   # 沒有第二峰也算通過
        chk[f"second{lo:.0f}-{hi:.0f}"] = ok.to_numpy()

    # ---- 峰形條件（crit 中有給值才檢查）----
    for name, colname, key, op in _SHAPE_CRITERIA:
        if crit.get(key) is not None:
            _need(s, colname, key)
            v = col(colname)
            chk[name] = ((v <= crit[key]) if op == "<=" else (v >= crit[key])).to_numpy()

    names = list(chk.columns)
    valid = s["valid"].fillna(False).astype(bool).to_numpy()
    chk.loc[~valid, names] = False
    chk["pass"] = chk[names].all(axis=1)

    fr = np.full(len(chk), "", dtype=object)          # 向量化組合失敗原因（大量資料時快很多）
    for nm in names:
        fr = fr + np.where(chk[nm].to_numpy(), "", nm + ",")
    chk["fail_reasons"] = pd.Series(fr, index=chk.index).str.rstrip(",")
    chk.loc[~valid, "fail_reasons"] = "invalid"
    return chk


# ============================================================================
# 覆蓋篩選
# ============================================================================
def select_coverage(summary, YN, TP, evals, wl=WL, band=(400, 1000), level=0.5,
                    min_abs_T=0.0, overlap_penalty=0.5, min_gain_nm=5, max_n=None,
                    quality=None):
    """
    覆蓋定義：在 λ 處 T/T_peak >= level 且 T >= min_abs_T。
    只從 evaluate() 判定 PASS 的候選中挑；每輪選 score 最大者：
        score = 新覆蓋nm × quality − overlap_penalty × 重疊nm
    quality：品質權重 0–1（pd.Series 以 id 對齊，或與 summary 同長度的 array）；None = 全部 1
             例：偏好接近 Gaussian → quality = (1 − summary["max_gauss_nrmse"] / 0.05).clip(0, 1)
    summary / YN / TP / evals 的列順序需一致；chosen 為這組輸入內的位置索引。
    """
    lo, hi = band
    inb = (wl >= lo) & (wl <= hi)
    cand = np.flatnonzero(evals["pass"].to_numpy())
    Cc = (YN[cand][:, inb] >= level) & (TP[cand][:, inb] >= min_abs_T)
    ewc = summary["eq_width"].to_numpy(dtype=float)[cand]
    if quality is None:
        qc = np.ones(len(cand))
    else:
        q = (quality.reindex(summary.index).to_numpy(float) if isinstance(quality, pd.Series)
             else np.asarray(quality, float))
        qc = np.clip(np.nan_to_num(q[cand], nan=0.0), 0.0, 1.0)
    alive = np.ones(len(cand), bool)
    covered = np.zeros(inb.sum(), bool)
    chosen, log = [], []

    while alive.any():
        gain = Cc[:, ~covered].sum(1)
        over = Cc[:, covered].sum(1)
        score = gain * qc - overlap_penalty * over - 1e-3 * ewc
        score[(gain < min_gain_nm) | ~alive] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break
        i = int(cand[j])
        alive[j] = False
        chosen.append(i)
        covered |= Cc[j]
        log.append(dict(
            id=summary.index[i],
            peaks=summary["sig_peak_wls"].iloc[i],
            fwhms=summary["sig_fwhms"].iloc[i],
            T_peaks=summary["sig_T_peaks"].iloc[i],
            new_nm=int(gain[j]), overlap_nm=int(over[j]),
            cum_coverage=round(float(covered.mean()), 3),
        ))
        if covered.all() or (max_n and len(chosen) >= max_n):
            break

    combo = TP[chosen][:, inb].sum(0) if chosen else np.zeros(inb.sum())
    report = dict(
        n_candidates=len(cand),
        n_selected=len(chosen),
        coverage=float(covered.mean()),
        gaps=_runs(~covered, wl[inb]),
        flatness=float(combo.min() / combo.max()) if chosen and combo.max() > 0 else 0.0,
    )
    return pd.DataFrame(log), report, chosen


# ============================================================================
# 視覺化
# ============================================================================
def _legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [
        Line2D([], [], color="C0", lw=1.3, marker="<", ms=4, label="FWHM"),
        Line2D([], [], color="C0", ls=":", lw=1.2, label="width @10%"),
        Line2D([], [], color="C0", marker="v", ls="", label="peak"),
        Line2D([], [], color="C0", marker="^", mfc="none", ls="", label="merged ripple"),
        Line2D([], [], color="red", marker="x", mew=2, ls="", label="shoulder"),
        Line2D([], [], color="tab:red", ls="--", marker="o", ms=4, label="max in-band leak"),
        Line2D([], [], color="tab:purple", ls="-.", label="centroid"),
        Line2D([], [], color="0.4", ls=":", label="5%/95% energy"),
        Patch(fc="C0", alpha=0.25, label="peak segment (area %)"),
        Patch(fc="0.7", alpha=0.35, label="sidelobe segment"),
        Patch(fc="0.93", label="out of band"),
        Patch(fc="tab:red", alpha=0.35, label="leak-excluded window (bottom bar)"),
    ]


def plot_profile(k, summary, peaks, TP=None, T_raw=None, evals=None, wl=WL, cfg=CFG,
                 ax=None, compact=False, labels=None):
    """
    k：位置索引。TP 為 None 時（keep_arrays=False）會由 T_raw 即時重算。
    labels：標題附加說明（設計參數 DataFrame / id→文字 的 Series 或 dict / 函式 f(id, summary_row)）
    所有特徵標在圖上，右側文字框為整體特徵與 PASS/FAIL。
    """
    import matplotlib.pyplot as plt
    single = ax is None
    if single:
        fig, ax = plt.subplots(figsize=(12, 4.4))
    sid = summary.index[k]
    s = summary.iloc[k]
    if TP is not None:
        t = np.asarray(TP[k], float)
    elif T_raw is not None:
        y_, b_ = preprocess(T_raw[k], cfg)
        t = y_ + b_
    else:
        raise ValueError("TP 與 T_raw 至少需提供一個")
    fs = 7 if compact else 8
    lo, hi = cfg["band"]

    ax.axvspan(wl[0], lo, color="0.93", zorder=0)
    ax.axvspan(hi, wl[-1], color="0.93", zorder=0)
    if T_raw is not None and TP is not None:
        ax.plot(wl, T_raw[k], color="0.6", lw=0.8, zorder=2)
    ax.plot(wl, t, color="k", lw=1.1, zorder=3)
    ax.set_xlim(wl[0], wl[-1])
    extra = _label_text(sid, s, labels)
    ax.set_title(f"{sid}  |  {extra}" if extra else str(sid), fontsize=fs + 2, loc="left")
    if compact:
        ax.tick_params(labelsize=fs)
    else:
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance")

    if not bool(s.get("valid", False)) or not (s.get("n_peaks", 0) > 0):
        ax.text(0.5, 0.5, "no valid peak", transform=ax.transAxes, ha="center")
        return ax

    p = peaks[peaks["id"] == sid].sort_values("peak_wl")
    base = s["baseline"]
    cols = plt.cm.tab10.colors
    ci = 0
    for _, r in p.iterrows():
        m = (wl >= r.seg_lo) & (wl <= r.seg_hi)
        if not r.is_sig:
            ax.fill_between(wl[m], base, t[m], color="0.7", alpha=0.35, lw=0, zorder=1)
            continue
        c = cols[ci % 10]
        ax.fill_between(wl[m], base, t[m], color=c, alpha=0.22, lw=0, zorder=1)
        ax.annotate("", xy=(r.hm_left, r.hm_level), xytext=(r.hm_right, r.hm_level),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=1.3, shrinkA=0, shrinkB=0),
                    zorder=4)
        ax.plot([r.fw10_left, r.fw10_right], [r.fw10_level] * 2, ":", color=c, lw=1.2, zorder=4)
        ax.plot(r.peak_wl, r.T_peak, "v", color=c, ms=6, zorder=5)
        for q in r.ripple_wls:
            ax.plot(q, np.interp(q, wl, t), "^", color=c, ms=4, mfc="none", zorder=5)
        lab = f"{r.peak_wl:.0f} nm\nFWHM {r.fwhm:.1f}\nA {100 * r.area_frac:.0f}%  Q {r.Q:.0f}"
        if not compact:
            lab += f"\nedge {r.edge_l:.0f}/{r.edge_r:.0f}  SF {r.shape_factor:.2f}  FF {r.flat_factor:.2f}"
            if r.ripple > 0:
                lab += f"\nripple {r.ripple:.2f}"
        ax.annotate(lab, (r.peak_wl, r.T_peak), xytext=(0, 6 + 28 * (ci % 2)),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fs, color=c,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.85), zorder=6)
        ci += 1

    for q in s["shoulder_wls"]:
        ax.plot(q, np.interp(q, wl, t), "x", color="red", ms=8, mew=2, zorder=6)

    excl = s.get("leak_excl")
    for a, b in (excl if isinstance(excl, list) else []):
        ax.axvspan(a, b, ymin=0, ymax=0.025, color="tab:red", alpha=0.35, lw=0, zorder=2)
    if np.isfinite(s["leak_max_band"]):
        ax.axhline(s["leak_max_band"], ls="--", color="tab:red", lw=0.8, zorder=2)
        ax.plot(s["leak_wl"], s["leak_max_band"], "o", color="tab:red", ms=4, zorder=5)
        ax.text(wl[0] + 3, s["leak_max_band"], f"leak {s['leak_max_band']:.3f}",
                color="tab:red", fontsize=fs, va="bottom")
    for x in (s["wl05"], s["wl95"]):
        ax.axvline(x, ls=":", color="0.4", lw=0.8, zorder=2)
    ax.axvline(s["centroid"], ls="-.", color="tab:purple", lw=0.8, zorder=2)

    ax.set_ylim(min(0.0, t.min()), max(t.max(), 0.05) * (1.45 if compact else 1.7))

    lines = [
        f"sig/side/shoulder {int(s.n_sig_peaks)}/{int(s.n_sidelobes)}/{int(s.n_shoulders)}",
        f"T_peak {s.T_peak:.2f}  rej {s.rejection_db:.1f} dB",
        f"core {s.core_frac:.2f}  in-band {s.in_band_frac:.2f}",
        f"span90 {s.span90:.0f} nm",
        f"max FWHM {s.max_fwhm_sig:.1f}  edge {s.max_edge:.0f}",
        f"oob max T {s.leak_max_oob:.3f}",
    ]
    if "min_raw_pts_fwhm" in s and pd.notna(s["min_raw_pts_fwhm"]):
        lines.append(f"raw pts in FWHM {int(s['min_raw_pts_fwhm'])}")
    if s.n_T_gt1 or s.n_T_neg:
        lines.append("!! T>1 or T<0 (check FDTD)")
    face = "white"
    if evals is not None:
        ev = evals.iloc[k]
        if ev["pass"]:
            lines.append("PASS")
            face = "#e8f5e9"
        else:
            lines.append("FAIL:\n  " + ev["fail_reasons"].replace(",", "\n  "))
            face = "#ffebee"
    ax.text(1.01, 1.0, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=fs, family="monospace", bbox=dict(boxstyle="round", fc=face, ec="0.6"))

    if single:
        ax.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=6, fontsize=7, frameon=False)
        plt.tight_layout()
        plt.show()
    return ax


def plot_gallery(idx, summary, peaks, TP=None, T_raw=None, evals=None, wl=WL, cfg=CFG,
                 ncols=2, nrows=4, pdf_path=None, progress=True, labels=None):
    """多條光譜分頁檢查；給 pdf_path 會輸出多頁 PDF。labels 見 plot_profile"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    idx = list(idx)
    per = ncols * nrows
    pdf = PdfPages(pdf_path) if pdf_path else None
    with _progress(len(idx), "plot", progress and pdf is not None) as bar:
        for start in range(0, len(idx), per):
            chunk = idx[start:start + per]
            fig, axes = plt.subplots(nrows, ncols, figsize=(9.5 * ncols, 3.3 * nrows), squeeze=False)
            for ax, k in zip(axes.flat, chunk):
                plot_profile(k, summary, peaks, TP, T_raw, evals, wl, cfg, ax=ax, compact=True,
                             labels=labels)
            for ax in axes.flat[len(chunk):]:
                ax.axis("off")
            fig.legend(handles=_legend_handles(), loc="lower center", ncol=11, fontsize=7,
                       frameon=False)
            fig.tight_layout(rect=(0, 0.03, 1, 1))
            if pdf:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.show()
            bar.update(len(chunk))
    if pdf:
        pdf.close()


def plot_overview(summary, evals, chosen=(), wl=WL, cfg=CFG, crit=CRIT,
                  max_rows=150, max_points=20000, seed=0):
    """左：主峰波長 vs 最大 FWHM（大量資料時隨機抽樣顯示）；右：候選的半高區段（紅 = 選中）"""
    import matplotlib.pyplot as plt
    crit = {**CRIT, **crit}
    rng = np.random.default_rng(seed)
    n = len(summary)
    ok = evals["pass"].to_numpy()
    has_pk = summary["n_peaks"].fillna(0).to_numpy() > 0
    x = summary["main_peak_wl"].to_numpy(float)
    yv = summary["max_fwhm_sig"].to_numpy(float)
    ch = np.asarray(list(chosen), int)
    chs = set(ch.tolist())
    vmax = crit.get("max_sig_peaks") or max(3, int(np.nanmax(summary["n_sig_peaks"].to_numpy(float))))

    show = np.zeros(n, bool)
    show[rng.choice(n, min(n, max_points), replace=False)] = True
    show[ch] = True

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw=dict(width_ratios=[1, 1.2]))
    f = has_pk & ~ok & show
    okp = ok & show
    a1.scatter(x[f], yv[f], marker="x", color="0.6", s=12, label="fail", rasterized=True)
    sc = a1.scatter(x[okp], yv[okp], c=summary["n_sig_peaks"].to_numpy(float)[okp], cmap="viridis",
                    vmin=1, vmax=vmax, s=24, edgecolors="k", linewidths=0.3,
                    label="pass", rasterized=True)
    if len(ch):
        a1.scatter(x[ch], yv[ch], s=130, facecolors="none", edgecolors="red", linewidths=1.5,
                   label="selected")
    a1.axhline(crit["max_fwhm"], ls="--", color="tab:red", lw=0.8)
    a1.axvspan(*cfg["band"], color="gold", alpha=0.08)
    fig.colorbar(sc, ax=a1, label="n sig peaks")
    a1.set(xlabel="main peak wavelength (nm)", ylabel="max FWHM of sig peaks (nm)",
           xlim=(wl[0], wl[-1]),
           title=f"showing {show.sum():,}/{n:,} | pass {ok.sum():,}")
    a1.legend(fontsize=8)

    pass_idx = np.array([i for i in np.flatnonzero(ok) if i not in chs], int)
    n_extra = max(0, max_rows - len(ch))
    if len(pass_idx) > n_extra:
        pass_idx = rng.choice(pass_idx, n_extra, replace=False)
    rows = sorted(list(ch) + list(pass_idx), key=lambda i: x[i])
    for r_i, i in enumerate(rows):
        sel = i in chs
        for a, b in summary["hm_intervals"].iloc[i]:
            a2.plot([a, b + 1], [r_i, r_i], color="red" if sel else "0.55",
                    lw=3 if sel else 1.5, solid_capstyle="butt")
    a2.axvspan(*cfg["band"], color="gold", alpha=0.08)
    a2.set(xlim=(wl[0], wl[-1]), xlabel="wavelength (nm)",
           ylabel="PASS candidates (sorted by main peak)",
           title=f"half-max intervals (red = selected, {len(rows)} rows)")
    if 0 < len(rows) <= 40:
        a2.set_yticks(range(len(rows)))
        a2.set_yticklabels([str(summary.index[i]) for i in rows], fontsize=6)
    fig.tight_layout()
    plt.show()


def plot_selection(chosen, summary, TP, report, wl=WL, cfg=CFG):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axvspan(*cfg["band"], color="gold", alpha=0.08)
    for a, b in report["gaps"]:
        ax.axvspan(a - 0.5, b + 0.5, color="red", alpha=0.15)
    for n, i in enumerate(chosen):
        c = f"C{n % 10}"
        ax.plot(wl, TP[i], color=c, lw=1.2, label=str(summary.index[i]))
        for pw in summary["sig_peak_wls"].iloc[i]:
            ax.text(pw, np.interp(pw, wl, TP[i]), f"{pw:.0f}", color=c, fontsize=7,
                    ha="center", va="bottom")
    if chosen:
        ax.plot(wl, TP[chosen].max(0), "k--", lw=0.8, label="envelope")
    ax.set(xlim=(wl[0], wl[-1]), xlabel="Wavelength (nm)", ylabel="Transmittance",
           title=f"coverage {report['coverage']:.1%} | gaps (red) {len(report['gaps'])} | "
                 f"flatness {report['flatness']:.2f}")
    ax.legend(ncol=6, fontsize=7)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 自我檢查
# ============================================================================
def self_check(verbose=True):
    """
    以少量合成資料實際執行一次，確認欄位、判定條件與標題說明功能都存在（約 1–2 秒）。
    更新程式後建議先執行：self_check()  → 應顯示 OK
    """
    T, ids, _ = make_synthetic(n=200, seed=1, progress=False)
    s, p, YN, TP = extract_batch(T, ids, n_jobs=1, progress=False)
    missing = []
    for c in ["gauss_nrmse", "shape_factor", "flat_factor", "hm_unresolved", "edge_l", "edge_r"]:
        if c not in p:
            missing.append(f"peaks.{c}")
    for c in ["max_shape_factor", "max_flat_factor", "max_gauss_nrmse", "main_flat_factor",
              "second_peak_wl", "second_rel_height", "leak_excl", "rejection_db", "n_hm_unresolved"]:
        if c not in s:
            missing.append(f"summary.{c}")
    crit = dict(CRIT, max_sig_peaks=None, peak_ranges=[dict(range=(400, 600), max=5)],
                main_peak_range=(400, 1000), second_peak_range=(400, 1000),
                second_peak_required=False,
                max_shape_factor=2.2, max_flat_factor=0.5, max_gauss_nrmse=0.05,
                min_main_area_frac=0.5)
    ev = evaluate(s, crit, p)
    for c in ["pk400-600", "main400-1000", "second400-1000", "shape", "flat", "gauss",
              "dominance", "pass", "fail_reasons"]:
        if c not in ev:
            missing.append(f"evals.{c}")
    lab = _label_text(ids[0], s.iloc[0], pd.DataFrame({"period": [400.0]}, index=[ids[0]]))
    if lab != "period=400":
        missing.append("labels")
    select_coverage(s, YN, TP, evaluate(s), min_abs_T=0.0,
                    quality=(1 - s["max_gauss_nrmse"] / 0.05).clip(0, 1))
    ok = not missing
    if verbose:
        print(f"spectra_features {__version__}: " + ("OK" if ok else f"缺少 {missing}"))
    return ok


# ============================================================================
# 測試流程（合成資料 / FDTD 資料）
# ============================================================================
def _print_stats(summary, ev):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    n = len(summary)
    print(f"\n[PASS] {int(ev['pass'].sum()):,}/{n:,} ({ev['pass'].mean():.1%})")
    print("\n[各條件通過率]\n", ev.drop(columns=["pass", "fail_reasons"]).mean().round(3))
    if "n_sig_peaks" in summary:
        print("\n[有效峰數分布]\n",
              summary["n_sig_peaks"].fillna(0).astype(int).value_counts().sort_index())
    print("\n[主要失敗原因]\n", ev.loc[~ev["pass"], "fail_reasons"].value_counts().head(10))
    cols = [c for c in ["n_sig_peaks", "main_peak_wl", "second_peak_wl", "max_fwhm_sig", "core_frac",
                        "rejection_db", "in_band_frac", "T_peak", "max_ripple", "max_shape_factor",
                        "max_flat_factor", "max_gauss_nrmse", "min_raw_pts_fwhm"] if c in summary]
    print("\n[特徵統計]\n", summary[cols].describe(percentiles=[0.05, 0.5, 0.95]).T.round(3))


def run_synthetic(n=20000, seed=0, n_jobs=-1, chunk_size=1000, crit=CRIT, coverage_kw=None,
                  plots=True):
    """合成資料測試：產生 → 擷取 → 與 ground truth 比對 → 覆蓋篩選 → 作圖（標題顯示類型）"""
    T, ids, meta = make_synthetic(n=n, seed=seed)
    summary, peaks, YN, TP = extract_batch(T, ids, n_jobs=n_jobs, chunk_size=chunk_size)
    ev = evaluate(summary, crit, peaks)
    _print_stats(summary, ev)

    chk = synthetic_check(summary, meta)
    print("\n[偵測到的有效峰數 vs 類型]\n", chk["n_sig_peaks"])
    print("\n[shoulder 偵測比例]\n", chk["shoulder"])
    print("\n[單峰類型的中心/FWHM 誤差（中位數）]\n", chk["accuracy"])
    print("\n[各類型 PASS 比例]\n", ev["pass"].groupby(meta["type"]).mean().round(3))

    kw = dict(level=0.5, min_abs_T=0.2)
    kw.update(coverage_kw or {})
    log, rep, chosen = select_coverage(summary, YN, TP, ev, **kw)
    print("\n[覆蓋篩選]\n", log)
    print(rep)

    if plots:
        types = meta["type"].to_numpy()
        ex = [int(np.flatnonzero(types == t)[0]) for t in pd.unique(types)]   # 每類型各一條
        plot_profile(ex[0], summary, peaks, TP, evals=ev, labels=meta["type"])
        plot_gallery(ex, summary, peaks, TP, evals=ev, labels=meta["type"])
        plot_overview(summary, ev, chosen, crit=crit)
        if chosen:
            plot_selection(chosen, summary, TP, rep)
    return dict(T=T, ids=ids, meta=meta, summary=summary, peaks=peaks, evals=ev,
                YN=YN, TP=TP, log=log, report=rep, chosen=chosen)


def run_fdtd(T_raw, wl_raw=None, ids=None, params=None, wl=WL, cfg=CFG, crit=CRIT,
             n_jobs=-1, chunk_size=2000, keep_arrays=None, coverage_kw=None, plots=True,
             fail_pdf="fdtd_fail_check.pdf", max_pdf=400, seed=0, quality=None, labels=None):
    """
    FDTD 資料測試（輸入 numpy array）：整理格式 → 擷取 → 取樣檢查 → 判定 → 覆蓋篩選 → 作圖
      T_raw / wl_raw / ids：見 from_array
      params     : 設計參數 DataFrame（index 為 ids 字串），會列出被選中設計的參數
      keep_arrays: None → 20 萬條以下保留 YN/TP，以上改為只對 PASS 候選重算
      fail_pdf   : FAIL 光譜的檢查圖輸出路徑（最多 max_pdf 條，隨機抽樣）；None 不輸出
      quality    : 覆蓋篩選的品質權重；"gauss" → 依 max_gauss_nrmse 自動計算
      labels     : 圖標題附加說明；"params" → 使用 params 表；或 DataFrame / Series / dict / 函式
    """
    T, ids, wl_s = from_array(T_raw, wl_raw, ids, wl)
    n = len(T)
    if keep_arrays is None:
        keep_arrays = n <= 200_000
    if isinstance(labels, str) and labels == "params":
        labels = params
    print(f"[FDTD] spectra_features {__version__} | {n:,} 條光譜 | "
          f"格點 {wl[0]:.0f}–{wl[-1]:.0f} nm, {len(wl)} 點 | keep_arrays={keep_arrays}")

    summary, peaks, YN, TP = extract_batch(T, ids, wl=wl, cfg=cfg, n_jobs=n_jobs,
                                           chunk_size=chunk_size, keep_arrays=keep_arrays)
    if wl_raw is not None:
        summary, peaks = add_sampling_info(summary, peaks, wl_s)
    ev = evaluate(summary, crit, peaks)
    _print_stats(summary, ev)

    if isinstance(quality, str) and quality == "gauss":
        quality = (1 - summary["max_gauss_nrmse"] / 0.05).clip(0, 1)

    # ---- 覆蓋篩選 ----
    kw = dict(level=0.5, min_abs_T=0.2)
    kw.update(coverage_kw or {})
    if keep_arrays:
        log, rep, chosen = select_coverage(summary, YN, TP, ev, wl=wl, band=cfg["band"],
                                           quality=quality, **kw)
        sel_summary, sel_TP, sel_chosen = summary, TP, chosen
    else:
        idx = np.flatnonzero(ev["pass"].to_numpy())
        YNs, TPs = recompute_arrays(T, idx, wl, cfg)
        log, rep, ch = select_coverage(summary.iloc[idx], YNs, TPs, ev.iloc[idx],
                                       wl=wl, band=cfg["band"], quality=quality, **kw)
        chosen = [int(idx[c]) for c in ch]
        sel_summary, sel_TP, sel_chosen = summary.iloc[idx], TPs, ch
    print("\n[覆蓋篩選]\n", log)
    print(rep)
    if params is not None and chosen:
        print("\n[選中設計的參數]\n", params.reindex([summary.index[i] for i in chosen]))

    # ---- 作圖 ----
    if plots:
        TPp, Traw = (TP, None) if keep_arrays else (None, T)
        plot_overview(summary, ev, chosen, wl=wl, cfg=cfg, crit=crit)
        if chosen:
            plot_selection(sel_chosen, sel_summary, sel_TP, rep, wl=wl, cfg=cfg)
            plot_gallery(chosen, summary, peaks, TPp, Traw, ev, wl, cfg, labels=labels)
        fails = np.flatnonzero(~ev["pass"].to_numpy())
        if fail_pdf and len(fails):
            if len(fails) > max_pdf:
                fails = np.sort(np.random.default_rng(seed).choice(fails, max_pdf, replace=False))
            plot_gallery(fails, summary, peaks, TPp, Traw, ev, wl, cfg, pdf_path=fail_pdf,
                         labels=labels)
            print(f"FAIL 檢查圖已輸出：{fail_pdf}（{len(fails)} 條）")

    return dict(T=T, ids=ids, wl_raw=wl_s, summary=summary, peaks=peaks, evals=ev,
                YN=YN, TP=TP, log=log, report=rep, chosen=chosen, labels=labels)


# ============================================================================
# 範例
# ============================================================================
if __name__ == "__main__":
    # ---------------- 從 numpy array 匯入 ----------------
    # (A) 已是 (n_samples, 751)、350–1100 nm、1 nm 間隔 → 直接使用
    # T = np.load("T.npy")                                            # 或任何來源的 ndarray
    # ids = [f"design_{i}" for i in range(len(T))]                    # 可省略，預設 0..n-1（建議用 str）
    # summary, peaks, YN, TP = extract_batch(T, ids, n_jobs=-1)
    #
    # (B) 方向相反 (751, n_samples) → 先轉置
    # summary, peaks, YN, TP = extract_batch(T.T, ids)
    #
    # (C) 百分比單位 → 先 /100
    # summary, peaks, YN, TP = extract_batch(T / 100, ids)
    #
    # (D) FDTD 原始格點（頻率等間隔 / 非等間隔 / 單位 m 或 Hz / 範圍不同）
    #     wl_or_f: (n_raw,)；T_raw: (n_samples, n_raw) 或 (n_raw, n_samples) 皆可
    # T, wl_raw = to_uniform_grid(wl_or_f, T_raw)                     # 注意回傳兩個值
    # summary, peaks, YN, TP = extract_batch(T, ids)
    # summary, peaks = add_sampling_info(summary, peaks, wl_raw)      # 窄峰取樣檢查
    #
    # (E) 格點為 1 nm 但範圍不是 350–1100 → 傳入對應 wl，後續函式也要用同一個 wl
    # wl = np.arange(400, 1001, 1.0)
    # summary, peaks, YN, TP = extract_batch(T, ids, wl=wl)
    # ev = evaluate(summary)
    # log, rep, chosen = select_coverage(summary, YN, TP, ev, wl=wl)
    # plot_profile(0, summary, peaks, TP, evals=ev, wl=wl)
    #
    # (F) 單條光譜 (751,)
    # s, p, yn, tp = extract_features(t)                              # 回傳 dict、DataFrame、兩個 array
    #
    # (G) 大量資料：存成 .npy 後以 memory-map 讀取，不會一次載入記憶體
    # np.save("T.npy", T.astype(np.float32))
    # T = np.load("T.npy", mmap_mode="r")
    # summary, peaks, _, _ = extract_batch(T, ids, n_jobs=-1, chunk_size=2000, keep_arrays=False)

    # ---------------- 從 CSV / npz 匯入 ----------------
    # T, ids, wl_raw = load_csv("fdtd_T.csv", layout="wl_rows")       # → (n_samples, 751)
    # save_dataset("fdtd_T.npz", T, ids, wl_raw=wl_raw, params=params_df)
    # T, ids, wl, wl_raw, params = load_dataset("fdtd_T.npz")
    # summary, peaks, YN, TP = extract_batch(T, ids, wl=wl, n_jobs=-1)
    # summary, peaks = add_sampling_info(summary, peaks, wl_raw)
    # ev = evaluate(summary)

    # ---------------- 大量資料（0.1–1M 條）建議流程 ----------------
    # summary, peaks, _, _ = extract_batch(T, ids, n_jobs=-1, chunk_size=2000, keep_arrays=False)
    # summary.to_parquet("summary.parquet"); peaks.to_parquet("peaks.parquet")
    # ev = evaluate(summary)
    # idx = np.flatnonzero(ev["pass"].to_numpy())                     # 只對 PASS 候選重算陣列
    # YNs, TPs = recompute_arrays(T, idx)
    # log, rep, ch = select_coverage(summary.iloc[idx], YNs, TPs, ev.iloc[idx], min_abs_T=0.2)
    # chosen = idx[ch]                                                 # 轉回全體的位置索引
    # plot_selection(ch, summary.iloc[idx], TPs, rep)
    # plot_gallery(np.flatnonzero(~ev["pass"].to_numpy())[:400], summary, peaks,
    #              T_raw=T, evals=ev, pdf_path="check_fail.pdf")

    self_check()                    # 確認程式完整（應顯示 OK）

    MODE = "synthetic"              # "synthetic" | "fdtd"

    if MODE == "synthetic":
        # ---------------- 合成資料測試 ----------------
        # 先用 1–2 萬條測速度，再估算 1M 所需時間（圖標題會顯示合成類型）
        res = run_synthetic(n=20000, seed=0, n_jobs=-1, chunk_size=1000)

    elif MODE == "fdtd":
        # ---------------- FDTD資料測試 ----------------  << input from np array
        # 直接把記憶體中的 ndarray 指定給下列變數
        T_fdtd = None       # 必填：穿透率 ndarray，(n_samples, n_wl) 或 (n_wl, n_samples) 皆可
        wl_fdtd = None      # 已在 350–1100 nm / 1 nm 格點 → None
                            # 否則給原始波長/頻率軸 ndarray（m / µm / nm / Hz 自動判斷）
        ids_fdtd = None     # 可選：每條光譜的 id（list / ndarray），預設 "0", "1", ...
        params_fdtd = None  # 可選：設計參數 DataFrame（index = ids）

        # 圖標題附加說明（可選），例如：
        #   LABELS = "params"                                       → 使用 params_fdtd 的所有欄位
        #   LABELS = params_fdtd[["period", "height"]]              → 只顯示指定欄位
        #   LABELS = {"0": "baseline design", "15": "best NIR"}     → 指定文字
        #   LABELS = lambda sid, s: f"main {s.main_peak_wl:.0f} nm" → 自訂函式
        LABELS = None

        if T_fdtd is None:
            raise ValueError("請將 FDTD 穿透率 ndarray 指定給 T_fdtd")
        if params_fdtd is not None:
            params_fdtd.index = params_fdtd.index.astype(str)

        crit = dict(CRIT)   # 依需求調整，例如：
        # crit = dict(CRIT, max_sig_peaks=2, min_peak_T=0.08, min_rejection_db=10,
        #             main_peak_range=(600, 700), second_peak_range=(800, 900),
        #             second_peak_required=False, max_shape_factor=2.1, max_flat_factor=0.5)
        res = run_fdtd(
            T_fdtd, wl_raw=wl_fdtd, ids=ids_fdtd, params=params_fdtd, crit=crit,
            n_jobs=-1, chunk_size=2000,
            coverage_kw=dict(level=0.5, min_abs_T=0.05),
            fail_pdf="fdtd_fail_check.pdf", max_pdf=400, labels=LABELS,
        )

        # ---- 匯出 PASS profiles 到 PDF ----
        EXPORT_PASS_PDF = True
        PASS_PDF = "fdtd_pass.pdf"
        PASS_MAX = 400              # None = 全部；給整數則依主峰波長均勻抽樣到此數量
        SELECTED_ONLY = False       # True：只匯出覆蓋篩選選中的組合（res["chosen"]）

        if EXPORT_PASS_PDF:
            s, p, ev = res["summary"], res["peaks"], res["evals"]
            TP = res["TP"]                              # keep_arrays=False 時為 None
            Traw = res["T"] if TP is None else None     # 沒有 TP 時由原始 T 即時重算

            if SELECTED_ONLY:
                idx = np.asarray(res["chosen"], int)
            else:
                idx = np.flatnonzero(ev["pass"].to_numpy())
            idx = idx[np.argsort(s["main_peak_wl"].to_numpy()[idx])]   # 依主峰波長排序
            if PASS_MAX is not None and len(idx) > PASS_MAX:
                idx = idx[np.linspace(0, len(idx) - 1, PASS_MAX).astype(int)]

            if len(idx):
                print(f"匯出 {len(idx):,} 條 PASS profiles，約 {int(np.ceil(len(idx) / 8)):,} 頁 → {PASS_PDF}")
                plot_gallery(idx, s, p, TP=TP, T_raw=Traw, evals=ev, pdf_path=PASS_PDF,
                             labels=res["labels"])
            else:
                print("沒有 PASS 的光譜，未輸出 PDF")

    # 結果都在 res 中：res["summary"], res["peaks"], res["evals"], res["chosen"] ...
    # 存檔：res["summary"].to_parquet("summary.parquet"); res["peaks"].to_parquet("peaks.parquet")
