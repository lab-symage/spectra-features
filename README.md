# FDTD 濾光片穿透率：特徵擷取、判定與覆蓋篩選

本工具從 FDTD 模擬的濾光片穿透率光譜中擷取峰形、漏光、能量分布等特徵，依條件判定 PASS / FAIL，並從候選中挑選一組以單純、窄峰為主、可涵蓋 400–1000 nm 的光譜組合。支援 0.1–1M 條光譜的平行處理，並可把特徵直接標示在 profile 圖上檢查。

---

## 目錄

1. [快速開始](#1-快速開始)
2. [資料格式](#2-資料格式)
3. [處理流程總覽](#3-處理流程總覽)
4. [前處理](#4-前處理)
5. [峰偵測、ripple 合併與有效峰](#5-峰偵測ripple-合併與有效峰)
6. [逐峰特徵（peaks 表）](#6-逐峰特徵peaks-表)
7. [整體特徵（summary 表）](#7-整體特徵summary-表)
8. [判定條件（evaluate）](#8-判定條件evaluate)
9. [覆蓋篩選（select_coverage）](#9-覆蓋篩選select_coverage)
10. [視覺化標示說明](#10-視覺化標示說明)
11. [參數一覽（CFG / CRIT）](#11-參數一覽cfg--crit)
12. [合成測試資料](#12-合成測試資料)
13. [函式一覽](#13-函式一覽)
14. [大量資料與效能](#14-大量資料與效能)
15. [常見問題與注意事項](#15-常見問題與注意事項)

---

## 1. 快速開始

將 script 存成 `spectra_features.py`。若要使用多核心平行處理（`n_jobs != 1`），在 Windows、macOS 或 Jupyter 中必須以 import 的方式使用。

```python
import numpy as np
from spectra_features import run_fdtd, evaluate, select_coverage, plot_gallery, CRIT, CFG

# T: (n_samples, 751)，350–1100 nm、1 nm 間隔，數值 0–1
res = run_fdtd(T)

# 原始格點不同（頻率等間隔、單位 m 或 Hz 等）時，提供原始波長或頻率軸
res = run_fdtd(T_raw, wl_raw=wl_or_freq)

res["summary"]   # 每條光譜一列的整體特徵
res["peaks"]     # 每個峰一列的逐峰特徵
res["evals"]     # 各條件的判定結果、pass、fail_reasons
res["chosen"]    # 覆蓋篩選選中的光譜（位置索引）
```

調整判定門檻時不需要重新擷取特徵：

```python
crit = dict(CRIT, min_peak_T=0.08, min_rejection_db=8)
ev = evaluate(res["summary"], crit, res["peaks"])
```

---

## 2. 資料格式

| 項目 | 規格 |
|---|---|
| 穿透率 `T` | `np.ndarray`，shape `(n_samples, n_wavelengths)`，每列一條光譜 |
| 數值範圍 | 0–1。若為百分比，`from_array` 在最大值 > 1.5 時會自動除以 100 |
| 波長軸 `wl` | 等間隔、遞增，預設 `WL = np.arange(350, 1101, 1.0)`（751 點） |
| ids | 每條光譜的識別碼，建議用 `str`，以便與設計參數表對齊 |

不同來源的資料可用以下函式整理成標準格式：

| 函式 | 用途 |
|---|---|
| `from_array(T_raw, wl_raw=None, ids=None)` | numpy array → 標準格式。會自動轉置、處理百分比、內插 |
| `to_uniform_grid(wl_raw, T_raw)` | 不等間隔或頻率取樣 → 1 nm 格點，回傳 `(T, wl_sorted)` |
| `load_csv(path, layout)` | CSV → 標準格式。`layout="wl_rows"` 或 `"sample_rows"` |
| `save_dataset` / `load_dataset` | 以 npz 存取光譜，設計參數另存 parquet |

`to_uniform_grid` 會自動判斷單位：最大值 > 10¹² 視為 Hz；< 10⁻³ 視為 m；< 50 視為 µm；其他視為 nm。

---

## 3. 處理流程總覽

```
T_raw ─► from_array / to_uniform_grid ─► T (n_samples, 751)
                                           │
                                  extract_batch（多核心）
                                           │
          ┌────────────────────────────────┼─────────────────────────┐
     summary（整體特徵）            peaks（逐峰特徵）            YN / TP（正規化 / 處理後 T）
          │                                │
          └──────────► evaluate(crit) ◄─────┘
                             │
                   evals（PASS / FAIL + 原因）
                             │
                   select_coverage ─► chosen、report（覆蓋率、缺口）
                             │
          plot_profile / plot_gallery / plot_overview / plot_selection
```

單條光譜在 `extract_features` 中的處理順序：

1. 前處理：平滑（可選）、扣基線（可選）、正規化
2. 計算整體能量分布特徵
3. 找候選峰 → 合併 ripple → 以 valley 切分區段 → 判定有效峰
4. 計算逐峰寬度、邊緣、ripple、Gaussian 相似度
5. 計算通帶、保護帶與漏光、rejection
6. 偵測 shoulder
7. 彙總峰相關特徵

---

## 4. 前處理

### 4.1 平滑與基線

$$
T_s(\lambda) = \begin{cases} \mathrm{SG}(T_\text{raw};\ w, k) & \text{if } w \ge 3 \\ T_\text{raw} & \text{otherwise} \end{cases}
$$

- SG 為 Savitzky–Golay 濾波，視窗 `sg_window = w`、階數 `sg_order = k`。預設 `w = 0`，即不平滑，因為 FDTD 沒有隨機雜訊。

$$
b = \begin{cases} 0 & \texttt{baseline\_mode="none"}\ \text{（預設）} \\ \mathrm{percentile}(T_s, p) & \texttt{"percentile"} \\ \min T_s & \texttt{"min"} \end{cases}
$$

$$
y(\lambda) = \max\left(T_s(\lambda) - b,\ 0\right), \qquad y_\text{max} = \max_\lambda y(\lambda)
$$

$$
y_n(\lambda) = \frac{y(\lambda)}{y_\text{max}} \quad\text{（正規化，YN）}, \qquad T_p(\lambda) = y(\lambda) + b \quad\text{（處理後絕對穿透率，TP）}
$$

穿透率建議使用 `baseline_mode="none"`：背景漏光本身就是要評估的對象，扣掉會掩蓋問題。

### 4.2 FDTD 數值檢查

| 特徵 | 定義 | 用途 |
|---|---|---|
| `n_T_gt1` | 原始 T 中 $T > 1.001$ 的點數 | 能量不守恆，通常是監視器位置或正規化設定有問題 |
| `n_T_neg` | 原始 T 中 $T < -0.001$ 的點數 | 常見於模擬時間不足造成的振盪 |

---

## 5. 峰偵測、ripple 合併與有效峰

### 5.1 候選峰

以 `scipy.signal.find_peaks` 在 $y_n$ 上找局部極大，條件：

- prominence ≥ `prom_frac`（預設 0.05）
- 高度 ≥ `height_frac`（預設 0.03）
- 相鄰峰間距 ≥ `min_dist_nm`（預設 3 nm）

### 5.2 prominence（局部背景）

採用 scipy 的定義：從峰頂向左右延伸水平線，直到碰到更高的訊號或光譜邊界；在左右兩段範圍內各取最低點，以兩者中**較高**的作為參考高度 $B$（局部背景）。

$$
P = y_{n,p} - B
$$

### 5.3 ripple 合併

依序檢查相鄰的候選峰 $q, p$，若兩峰之間的谷底滿足

$$
\min_{q \le j \le p} y_n(j) \ \ge\ \texttt{merge\_frac} \times \max\left(y_{n,q},\ y_{n,p}\right)
$$

就合併為同一個通帶，由群組中最高的峰代表（預設 `merge_frac = 0.6`）。被合併的子峰位置記錄在 `ripple_wls`。此步驟避免 flat-top 通帶上的 ripple 被誤判為多個峰。

### 5.4 valley 區段切分

設合併後的代表峰依波長排序為 $p_1 < p_2 < \dots < p_m$，相鄰峰之間的谷底為

$$
v_i = \arg\min_{p_i \le j \le p_{i+1}} y_n(j)
$$

第 $i$ 個峰的區段為 $[v_{i-1},\ v_i)$，其中 $v_0 = 0$，$v_m = $ 光譜末端。所有區段剛好不重疊地涵蓋整條光譜。

### 5.5 有效峰（significant peak）

同時滿足下列兩項才是有效峰（`is_sig = True`）：

$$
\texttt{area\_frac} \ge \texttt{sig\_area\_frac}\ (0.05) \quad\text{且}\quad \texttt{rel\_height} \ge \texttt{sig\_height\_frac}\ (0.2)
$$

不符合的峰稱為 **sidelobe**：次要的穿透帶、窄而尖的突起、振盪造成的小峰都屬於此類。prominence 未達 `prom_frac` 的細小起伏不會被偵測為峰，直接視為背景。

若沒有任何峰符合條件，彙總特徵會改用最高的峰計算，避免全部變成 NaN；但 `n_sig_peaks` 仍為 0。

---

## 6. 逐峰特徵（peaks 表）

每個合併後的峰一列，`id` 欄位對應 summary 的 index。

### 6.1 寬度的高度基準

寬度在高度水平 $L_r$ 處量測，$r$ 為 `rel_height`：

| `width_ref` | 高度水平 | 搜尋範圍 |
|---|---|---|
| `"prominence"`（預設） | $L_r = y_{n,p} - r \cdot P$ | prominence 的左右 base |
| `"absolute"` | $L_r = y_{n,p} - r \cdot (y_{n,p} - r_0)$，$r_0 = (\texttt{width\_ref\_value} - b)/y_\text{max}$ | 該峰的 valley 區段 |

從峰頂往左右搜尋，找到 $y_n$ 降到 $L_r$ 的位置，以線性內插求得分數索引 $x_L(r)$、$x_R(r)$。

- `prominence` 模式量的是「峰在局部背景之上的寬度」，不受漏光背景和相鄰峰影響。
- `absolute` 模式（例如基準 0，半高 = T_peak / 2）量的是實際被看到的頻寬。重疊峰可能找不到交點，此時寬度會被截在 valley，並標記 `hm_unresolved = True`。

### 6.2 特徵定義

| 特徵 | 公式 / 定義 | 用途 |
|---|---|---|
| `peak_wl` | 峰頂波長 $\lambda_p$ | 峰位置 |
| `T_peak` | $T_p(\lambda_p)$ | 峰的絕對穿透率 |
| `rel_height` | $y_{n,p}$（＝該峰 T ÷ 整條光譜最大 T，基線為 0 時） | 相對強度，判定有效峰 |
| `prominence` | $P$（正規化單位） | 峰相對局部背景的高度 |
| `fwhm` | $\left[x_R(0.5) - x_L(0.5)\right]\Delta\lambda$ | 半高全寬 (nm) |
| `hm_left` / `hm_right` | $\lambda\left(x_L(0.5)\right)$、$\lambda\left(x_R(0.5)\right)$ | 半高交點波長 |
| `hm_level` | $L_{0.5} \cdot y_\text{max} + b$ | 半高水平（絕對 T），作圖用 |
| `fw10` | $\left[x_R(0.9) - x_L(0.9)\right]\Delta\lambda$ | 10% 高度處全寬，反映裙擺 |
| `fw10_left` / `fw10_right` / `fw10_level` | 10% 高度的交點與水平 | 作圖、漏光排除範圍 |
| `Q` | $\lambda_p / \text{FWHM}$ | 品質因子，相對頻寬的倒數，跨波長比較窄度 |
| `shape_factor` | $\text{FW10} / \text{FWHM}$ | 尾巴長度。Gaussian $=\sqrt{\ln 10/\ln 2} \approx 1.82$；Lorentzian $= 3.0$ |
| `asymmetry` | $\dfrac{\text{hm\_right} - \lambda_p}{\lambda_p - \text{hm\_left}}$ | > 1 右側拖尾，< 1 左側拖尾 |
| `edge_l` | $\left[x_L(0.1) - x_L(0.9)\right]\Delta\lambda$ | 左緣 10%→90% 過渡寬，越小越陡 |
| `edge_r` | $\left[x_R(0.9) - x_R(0.1)\right]\Delta\lambda$ | 右緣 90%→10% 過渡寬 |
| `hm_unresolved` | absolute 模式下，區段邊界仍高於半高水平 | 重疊峰在固定基準下無法分開 |
| `ripple` | $\dfrac{y_{n,p} - \min_{j \in \text{FWHM},\ j\ \text{局部極小}} y_n(j)}{y_{n,p}}$，無局部極小時為 0 | 通帶內凹陷深度 |
| `n_ripple_peaks` | 被合併的子峰數 | 通帶平整度 |
| `ripple_wls` | 被合併子峰的波長 list | 作圖 |
| `area_frac` | $\dfrac{\sum_{j \in [v_{i-1}, v_i)} y_j}{\sum_j y_j}$ | 峰下面積佔比（圖上的 **A**），含區段內的背景 |
| `fwhm_area_frac` | $\dfrac{\sum_{j=\lceil x_L(0.5)\rceil}^{\lfloor x_R(0.5)\rfloor} y_j}{\sum_j y_j}$ | 半高寬內的能量佔比 |
| `seg_lo` / `seg_hi` | 區段兩端波長 | 作圖填色 |
| `gauss_nrmse` | 見 6.3 | 與 Gaussian 的偏差 |
| `is_sig` | 見 5.5 | 是否為有效峰 |
| `n_raw_pts_fwhm` | 原始（內插前）波長點落在 [hm_left, hm_right] 的數量 | 由 `add_sampling_info` 加入；窄峰取樣是否足夠 |

### 6.3 Gaussian 相似度 `gauss_nrmse`

不做擬合，直接與「同中心、同峰高、同 FWHM」的 Gaussian 比較：

$$
c = \frac{\text{hm\_left} + \text{hm\_right}}{2}, \qquad B' = y_{n,p} - P, \qquad A = y_{n,p} - B'
$$

$$
g(\lambda) = B' + A \exp\left[-4\ln 2 \, \frac{(\lambda - c)^2}{\text{FWHM}^2}\right]
$$

$$
\texttt{gauss\_nrmse} = \frac{1}{A}\sqrt{\frac{1}{N}\sum_{\lambda \in W}\left[y_n(\lambda) - g(\lambda)\right]^2}, \qquad W = [c - 1.5\,\text{FWHM},\ c + 1.5\,\text{FWHM}]
$$

absolute 模式時，$P$ 改用 $y_{n,p} - r_0$。

參考值：理想 Gaussian ≈ 0；同 FWHM 的 Lorentzian 約 0.09（在 ±1.5 FWHM 處 Lorentzian 仍有約 10%，Gaussian 已低於 0.3%）；flat-top、不對稱或有 shoulder 的峰會更高。比較視窗內若有相鄰峰，也會使數值上升。

---

## 7. 整體特徵（summary 表）

index 為 ids，每條光譜一列。以下 $\Delta\lambda$ 為波長間隔，band 為 `CFG["band"]`（預設 400–1000 nm）。

### 7.1 能量分布

| 特徵 | 公式 / 定義 | 用途 |
|---|---|---|
| `valid` | $y_\text{max} > 0$ | 有效光譜 |
| `baseline` | $b$ | 扣除的基線 |
| `T_peak` | $\max_\lambda T_p$ | 最大絕對穿透率 |
| `peak_wl_max` | $\arg\max_\lambda T_p$ | 最高點波長 |
| `T_mean_band` | band 內 $T_p$ 平均 | 整體穿透水準 |
| `total_area` | $\sum y \cdot \Delta\lambda$ | 總穿透能量 |
| `centroid` | $\dfrac{\sum \lambda\, y}{\sum y}$ | 能量重心。不對稱、多峰、背景都會使它偏離峰值波長 |
| `median_wl` | 累積能量 $\mathrm{CDF}(\lambda) = 0.5$ 的波長，$\mathrm{CDF}(\lambda) = \sum_{\lambda' \le \lambda} y / \sum y$ | 不受極端值影響的中心 |
| `wl05` / `wl95` | CDF = 0.05 / 0.95 的波長 | 能量分布範圍 |
| `span90` | $\text{wl95} - \text{wl05}$ | 涵蓋 90% 能量的寬度 |
| `rms_width` | $\sqrt{\dfrac{\sum (\lambda - \text{centroid})^2 y}{\sum y}}$ | 二階矩寬度，對尾巴與背景敏感 |
| `skewness` | $\dfrac{\sum (\lambda - \text{centroid})^3 y / \sum y}{\text{rms\_width}^3}$ | 正值往長波長拖，負值往短波長拖 |
| `eq_width` | $\text{total\_area} / y_\text{max}$ | 等效寬度 (nm)：同峰高的矩形寬度 |
| `in_band_frac` | $\sum_{\text{band}} y / \sum y$ | band 內能量佔比 |
| `below_band_frac` / `above_band_frac` | band 以下 / 以上的能量佔比 | band 外浪費的能量 |
| `n_hm_lobes` | $y_n \ge 0.5$ 的連續區段數 | 半高以上的瓣數 |
| `hm_cover_nm` | band 內 $y_n \ge 0.5$ 的點數 × $\Delta\lambda$ | 半高以上覆蓋的寬度 |
| `hm_intervals` | $y_n \ge 0.5$ 的區段 list | 覆蓋位置，用於覆蓋圖 |

### 7.2 峰的彙總

「有效峰」以下簡稱 sig。**main** 為所有峰中 `rel_height` 最高者。

| 特徵 | 定義 | 用途 |
|---|---|---|
| `n_peaks` | ripple 合併後的峰數 | — |
| `n_sig_peaks` | 有效峰數 | 單純度，判定 `peaks` |
| `n_sidelobes` | `n_peaks − n_sig_peaks` | 次要峰數 |
| `n_shoulders` / `shoulder_wls` | 見 7.4 | 隱藏峰 |
| `sig_peak_wls` / `sig_fwhms` / `sig_area_fracs` / `sig_T_peaks` | 各有效峰的值（list，依波長排序） | 檢視與 `peak_ranges` 判定 |
| `main_peak_wl` / `main_fwhm` / `main_area_frac` / `main_Q` | 主峰的對應值 | 主通道特性 |
| `main_shape_factor` / `main_gauss_nrmse` | 主峰的峰形 | 單峰峰形 |
| `max_fwhm_sig` | sig 中最大 FWHM | 判定 `fwhm` |
| `max_shape_factor` | sig 中最大 shape_factor | 判定 `shape` |
| `max_edge` | sig 中 edge_l、edge_r 的最大值 | 邊緣陡峭度 |
| `max_ripple` | sig 中最大 ripple | 判定 `ripple` |
| `max_gauss_nrmse` | sig 中最大 gauss_nrmse | 判定 `gauss` |
| `n_hm_unresolved` | sig 中 `hm_unresolved` 的個數 | absolute 模式下無法解析的重疊峰 |
| `core_frac` | $\dfrac{\sum_{\lambda \in \cup_\text{sig}[\text{hm\_left}, \text{hm\_right}]} y}{\sum y}$ | 能量集中在有效峰核心的程度。單一 Gaussian 理論值 $\mathrm{erf}(\sqrt{\ln 2}) \approx 0.76$；Lorentzian 0.5；背景、尾巴越多越低 |
| `min_peak_sep` | 相鄰有效峰的最小間距 (nm) | 峰距 |
| `min_resolution` | $\min_i \dfrac{\lambda_{i+1} - \lambda_i}{(\text{FWHM}_i + \text{FWHM}_{i+1})/2}$ | < 1 代表峰重疊、難以分開 |
| `min_raw_pts_fwhm` | sig 中最小的 `n_raw_pts_fwhm` | 由 `add_sampling_info` 加入，判定 `sampling` |

### 7.3 漏光與 rejection

**leak-excluded window**：對每個有效峰，排除範圍為

$$
\left[\ \min\left(\text{fw10\_left},\ \text{hm\_left} - g \cdot \text{FWHM}\right),\ \ \max\left(\text{fw10\_right},\ \text{hm\_right} + g \cdot \text{FWHM}\right)\ \right]
$$

其中 $g$ = `leak_guard_fwhm`（預設 1）。保護帶的作用是避免把峰自身的裙擺算成漏光。

**阻帶**：band 內、所有排除範圍以外的區域 $S$。

| 特徵 | 公式 | 用途 |
|---|---|---|
| `leak_excl` | 排除範圍的區段 list | 作圖（底部紅條） |
| `leak_max_band` | $\max_{\lambda \in S} T_p(\lambda)$ | band 內最嚴重的漏光 |
| `leak_wl` | 上式的波長 | 漏光來源位置 |
| `leak_mean_band` | $\mathrm{mean}_{\lambda \in S}\, T_p(\lambda)$ | 平均漏光 |
| `leak_max_oob` | band 外的 $\max T_p$ | band 外的穿透 |
| `rejection_db` | $10 \log_{10}\left(\dfrac{T_\text{peak}}{\text{leak\_max\_band}}\right)$ | 峰值與最大漏光的對比；$S$ 為空時為 NaN |

rejection 與漏光比例的換算：3 dB = 50%，6 dB = 25%，10 dB = 10%，13 dB = 5%，20 dB = 1%。這是相對值；若在乎絕對漏光量，請直接使用 `leak_max_band`。

### 7.4 shoulder 偵測

shoulder 是主峰邊緣上的隆起，不是獨立的局部極大，所以 `find_peaks` 抓不到。偵測方式是找「單調邊緣上斜率的凹陷」：

$$
d_1 = \mathrm{SG}'(y_n;\ \texttt{d1\_window}, 2), \qquad a = \max |d_1|
$$

- 上升沿：$d_1$ 的局部極小，prominence ≥ `shoulder_prom`·$a$，且 $d_1 > 0.02a$（斜率仍為正，代表沒有形成獨立的峰）。
- 下降沿：$d_1$ 的局部極大，prominence ≥ `shoulder_prom`·$a$，且 $d_1 < -0.02a$。
- 只保留 $y_n > 0.2$ 的位置，並排除距離任何候選峰 2 個點以內的位置。

使用一階導數而非二階導數，是為了避免把 flat-top 通帶的兩個圓角誤判為 shoulder。

---

## 8. 判定條件（evaluate）

`evaluate(summary, crit=CRIT, peaks=None)` 逐項檢查，**全部通過才是 PASS**。沒通過的項目以逗號串接在 `fail_reasons`。特徵為 NaN 時，比較結果一律視為不通過。

| 項目 | 通過條件 | 預設值 |
|---|---|---|
| `peaks` | $1 \le$ `n_sig_peaks` $\le$ `max_sig_peaks`；設為 `None` 不檢查 | 3 |
| `fwhm` | `max_fwhm_sig` ≤ `max_fwhm` | 60 nm |
| `core` | `core_frac` ≥ `min_core` | 0.5 |
| `in_band` | `in_band_frac` ≥ `min_in_band` | 0.85 |
| `peak_T` | `T_peak` ≥ `min_peak_T` | 0.3 |
| `rejection` | `rejection_db` ≥ `min_rejection_db` | 10 dB |
| `ripple` | `max_ripple` ≤ `max_ripple` | 0.2 |
| `shoulder` | `n_shoulders` = 0，或 `allow_shoulders=True` | False |
| `T_range` | `n_T_gt1` = 0 且 `n_T_neg` = 0 | — |
| `sampling` | `min_raw_pts_fwhm` ≥ `min_raw_pts_fwhm`；summary 有此欄位才檢查 | 5 |
| `pk<lo>-<hi>` | `peak_ranges` 中各範圍的峰數條件 | 無 |
| `shape` | `max_shape_factor` ≤ 門檻；有設定才檢查 | 無 |
| `gauss` | `max_gauss_nrmse` ≤ 門檻；有設定才檢查 | 無 |
| `dominance` | `main_area_frac` ≥ `min_main_area_frac`；有設定才檢查 | 無 |
| `invalid` | 無效光譜（全為 0） | — |

### 指定波長範圍的峰數 `peak_ranges`

```python
crit = dict(CRIT, peak_ranges=[
    dict(range=(800, 900), min=1, max=1),                          # 800–900 nm 恰 1 個有效峰
    dict(range=(400, 500), max=0, kind="all", name="no_blue"),     # 400–500 nm 不可有任何峰（含 sidelobe）
])
ev = evaluate(summary, crit, peaks)                                # kind="all" 需傳入 peaks
```

計數依據是峰值波長是否落在範圍內，範圍包含兩端點。`count_peaks_in_range(summary, lo, hi)` 可單獨計算各光譜在範圍內的峰數。

---

## 9. 覆蓋篩選（select_coverage）

### 9.1 覆蓋定義

光譜 $i$ 在波長 $\lambda$（band 內）視為「有覆蓋」的條件：

$$
C_i(\lambda) = \left[\, y_{n,i}(\lambda) \ge \texttt{level} \,\right] \wedge \left[\, T_{p,i}(\lambda) \ge \texttt{min\_abs\_T} \,\right]
$$

預設 `level = 0.5`，即半高以上。

### 9.2 貪婪挑選

只從 PASS 的候選中挑選。設 $U$ 為目前已覆蓋的波長集合，每一輪計算：

$$
\text{gain}_i = \left|\{\lambda : C_i(\lambda)\} \setminus U\right|, \qquad \text{over}_i = \left|\{\lambda : C_i(\lambda)\} \cap U\right|
$$

$$
\text{score}_i = \text{gain}_i \cdot q_i - \texttt{overlap\_penalty} \cdot \text{over}_i - 10^{-3} \cdot \text{eq\_width}_i
$$

- $q_i$ 為 `quality` 權重（0–1），預設全部為 1。例如偏好接近 Gaussian 的光譜：`quality = (1 - s["max_gauss_nrmse"] / 0.05).clip(0, 1)`。
- `gain` < `min_gain_nm` 的候選不列入考慮。
- 最後一項是同分時偏好較窄光譜的微小調整。

選出 score 最高者後更新 $U$，重複直到 band 完全覆蓋、沒有可選的候選，或達到 `max_n`。

### 9.3 報告

| 欄位 | 定義 |
|---|---|
| `n_candidates` | PASS 候選數 |
| `n_selected` | 選出數 |
| `coverage` | band 內被覆蓋的比例 |
| `gaps` | 未覆蓋的區段 list |
| `flatness` | band 內 $\min \sum_{i \in \text{chosen}} T_{p,i} \,/\, \max \sum_{i \in \text{chosen}} T_{p,i}$，疊加後的平坦度 |

逐輪紀錄（log）包含每次選中的 `id`、有效峰位置、FWHM、T、新覆蓋 nm、重疊 nm、累積覆蓋率。

---

## 10. 視覺化標示說明

`plot_profile` / `plot_gallery` 的圖上標示：

| 標示 | 意義 |
|---|---|
| 黑線 | 處理後的 T（TP） |
| 灰線 | 原始 T（有提供 T_raw 且有 TP 時才畫） |
| 彩色填滿 | 有效峰的 valley 區段，數值為 A（`area_frac`） |
| 灰色填滿 | sidelobe 的區段 |
| ▼ | 有效峰的峰頂 |
| ⟷ 雙箭頭 | FWHM，畫在實際的半高水平 |
| 點線（峰的顏色） | 10% 高度全寬 |
| △（空心） | 被合併的 ripple 子峰 |
| ✕（紅） | shoulder |
| 紅色虛線與紅點 | `leak_max_band` 的水平與位置 |
| 底部紅色細條 | leak-excluded window |
| 紫色點虛線 | centroid |
| 灰色點線 | wl05 / wl95 |
| 淺灰背景 | band 外 |

峰標籤格式：

```
550 nm          ← peak_wl
FWHM 30.2       ← fwhm
A 45%  Q 18     ← area_frac、Q
edge 8/9  SF 1.84   ← edge_l / edge_r、shape_factor（非 compact 模式）
ripple 0.12     ← ripple（> 0 時，非 compact 模式）
```

右側文字框：有效峰 / sidelobe / shoulder 數、T_peak 與 rejection、core_frac 與 in_band_frac、span90、最大 FWHM 與邊緣寬、band 外最大 T、原始取樣點數（若有）、FDTD 數值警告，以及 PASS（綠底）或 FAIL 與原因（紅底）。

其他圖：

- `plot_overview`：左圖為主峰波長對最大 FWHM 的散佈圖（PASS 依有效峰數著色，FAIL 為灰色 ×，選中者紅圈）；右圖為 PASS 候選的半高區段，選中者為紅色。
- `plot_selection`：選中組合的 T 疊圖、包絡線，未覆蓋的缺口以紅色區塊標示。

---

## 11. 參數一覽（CFG / CRIT）

建議以 `dict(CFG, ...)`、`dict(CRIT, ...)` 的方式覆寫部分參數。缺少的參數會自動以預設值補齊。

### 11.1 CFG（影響特徵擷取，修改後需重新擷取）

| 參數 | 預設 | 說明 |
|---|---|---|
| `baseline_mode` | `"none"` | `"none"` / `"percentile"` / `"min"` |
| `baseline_pct` | 1.0 | percentile 模式的百分位數 |
| `sg_window` / `sg_order` | 0 / 2 | Savitzky–Golay 平滑；0 為不平滑 |
| `d1_window` | 7 | shoulder 偵測的一階導數視窗 |
| `shoulder_prom` | 0.1 | shoulder 斜率凹陷門檻（相對最大斜率）；調大較不敏感 |
| `prom_frac` | 0.05 | 候選峰 prominence 門檻（相對最大值） |
| `height_frac` | 0.03 | 候選峰高度門檻 |
| `min_dist_nm` | 3 | 候選峰最小間距 |
| `merge_frac` | 0.6 | ripple 合併門檻；調小合併較多 |
| `sig_area_frac` | 0.05 | 有效峰面積佔比門檻 |
| `sig_height_frac` | 0.2 | 有效峰相對高度門檻 |
| `width_ref` | `"prominence"` | 寬度基準：`"prominence"` / `"absolute"` |
| `width_ref_value` | 0.0 | absolute 模式的基準 T 值 |
| `leak_guard_fwhm` | 1.0 | 漏光保護帶寬度（FWHM 倍數） |
| `band` | (400, 1000) | 目標波段 |

### 11.2 CRIT（影響判定，修改後只需重跑 evaluate）

| 參數 | 預設 | 說明 |
|---|---|---|
| `max_sig_peaks` | 3 | 有效峰數上限；`None` 不檢查 |
| `max_fwhm` | 60.0 | 有效峰最大 FWHM (nm) |
| `min_core` | 0.5 | core_frac 下限 |
| `min_in_band` | 0.85 | in_band_frac 下限 |
| `min_peak_T` | 0.3 | T_peak 下限 |
| `min_rejection_db` | 10.0 | rejection_db 下限 (dB) |
| `max_ripple` | 0.2 | 通帶 ripple 上限 |
| `allow_shoulders` | False | 是否允許 shoulder |
| `min_raw_pts_fwhm` | 5 | 原始取樣點數下限（有 sampling 資訊才檢查） |
| `peak_ranges` | `[]` | 指定波長範圍的峰數條件 |
| `max_shape_factor` | （未設定） | 峰形：尾巴長度上限 |
| `max_gauss_nrmse` | （未設定） | 峰形：Gaussian 偏差上限 |
| `min_main_area_frac` | （未設定） | 峰形：主峰能量佔比下限 |

### 11.3 參數設定範例

```python
# 接近 Gaussian 的單峰
cfg_g = dict(CFG, sig_height_frac=0.4, sig_area_frac=0.10)
crit_g = dict(CRIT, max_sig_peaks=1, max_fwhm=60, min_peak_T=0.08, min_rejection_db=10,
              max_ripple=0.1, max_shape_factor=2.2, max_gauss_nrmse=0.05, min_main_area_frac=0.6)

# 寬鬆條件
crit_relaxed = dict(CRIT, max_sig_peaks=4, max_fwhm=80, min_peak_T=0.08,
                    min_rejection_db=6, max_ripple=0.35, allow_shoulders=True)
```

---

## 12. 合成測試資料

`make_synthetic(n, mix=None, seed=0)` 回傳 `T`、`ids`、`meta`。

| 類型 | 說明 | 預設比例 |
|---|---|---|
| `single_gauss` | 單一 Gaussian | 0.30 |
| `single_lorentz` | 單一 Lorentzian（長尾） | 0.15 |
| `flattop_ripple` | flat-top 通帶加 ripple | 0.10 |
| `double` | 雙峰 | 0.12 |
| `triple` | 三峰 | 0.08 |
| `shoulder` | 主峰加緊鄰的小峰 | 0.08 |
| `sidelobe` | 主峰加遠處的小穿透帶 | 0.07 |
| `broad` | 寬峰（FWHM 150–350 nm） | 0.05 |
| `ringing` | 振盪尾巴（可能 T < 0） | 0.03 |
| `T_gt1` | T > 1 的數值問題 | 0.02 |

每條光譜都加上 0.01–0.03 的背景與小幅正弦起伏。`meta` 欄位：`type`、`true_n_peaks`、`true_centers`、`true_fwhms`。`synthetic_check(summary, meta)` 回傳各類型的有效峰數分布、shoulder 偵出率，以及單峰類型的中心與 FWHM 誤差。

---

## 13. 函式一覽

| 函式 | 說明 |
|---|---|
| `from_array` | numpy array → 標準格式 |
| `to_uniform_grid` | 內插到等間隔格點 |
| `load_csv` / `save_dataset` / `load_dataset` | 檔案存取 |
| `check_grid` | 檢查格點、shape、單位、NaN |
| `extract_features` | 單條光譜特徵擷取 |
| `extract_batch` | 批次擷取（分塊、多核心、進度條） |
| `add_sampling_info` | 原始取樣密度檢查 |
| `recompute_arrays` | 為子集合重算 YN / TP |
| `count_peaks_in_range` | 計算範圍內峰數 |
| `evaluate` | 判定 PASS / FAIL |
| `select_coverage` | 貪婪覆蓋篩選 |
| `plot_profile` / `plot_gallery` / `plot_overview` / `plot_selection` | 視覺化 |
| `make_synthetic` / `synthetic_check` | 合成資料與驗證 |
| `run_synthetic` / `run_fdtd` | 完整測試流程 |

---

## 14. 大量資料與效能

- **平行處理**：`extract_batch(..., n_jobs=-1, chunk_size=2000)`。同時處理中的區塊數限制為 2 × n_jobs，避免記憶體暴增。有安裝 tqdm 時顯示進度條，否則印出文字進度與 ETA。
- **記憶體**：1M × 751 的 float32 約 3 GB。`keep_arrays=False` 時不保留 YN / TP，可省約 6 GB；之後以 `recompute_arrays` 只為 PASS 候選重算。`run_fdtd` 在超過 20 萬條時會自動切換。
- **memory-map**：`np.load("T.npy", mmap_mode="r")` 讀入後可直接傳入，不會一次載入記憶體。
- **結果存檔**：summary 與 peaks 建議存成 parquet。調整 CRIT 時只需重跑 `evaluate`。
- **估算時間**：先用 1–2 萬條測速，`extract_batch` 結束時會印出每秒處理條數。

---

## 15. 常見問題與注意事項

- **ids 型別**：`extract_batch(ids=None)` 產生的是整數 0..n−1；`from_array` / `run_fdtd` 產生的是字串 "0", "1", …。要與設計參數表對齊時，兩邊型別必須一致。
- **格點**：非 350–1100 nm 的資料必須傳入對應的 `wl`，且後續所有函式都要用同一個 `wl`。
- **FDTD 頻率取樣**：轉成波長後間隔不均且遞減，必須先用 `to_uniform_grid`。高 Q 峰附近取樣太稀疏時，峰高會被低估、FWHM 被高估，可用 `add_sampling_info` 檢查。
- **低穿透率的設計**：`min_peak_T` 與 `select_coverage` 的 `min_abs_T` 要依資料分布一起調低，否則覆蓋率會很低。
- **共振型濾光片**（metasurface、GMR）：峰形接近 Lorentzian，`shape_factor` 約 3、`core_frac` 約 0.5 屬於正常範圍，`min_core`、`min_rejection_db` 需對應放寬。
- **版本相容**：漏光保護帶加入後，`leak_*` 與 `rejection_db` 的數值與舊版不同。舊版 summary 缺少 `gauss_nrmse` 等新欄位時，使用對應的 crit 條件會報錯並提示重新擷取。建議存檔時一併記錄當次使用的 CFG 與 CRIT。
- **平行處理環境**：在 Windows、macOS 或 Jupyter 中，請將 script 存成模組再 import 使用，主程式放在 `if __name__ == "__main__":` 下。
