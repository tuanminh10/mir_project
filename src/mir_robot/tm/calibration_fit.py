#!/usr/bin/env python3
"""Script hiệu chuẩn: Tìm hàm bù trừ từ dữ liệu thực nghiệm TN3"""
import numpy as np

# === DỮ LIỆU THÔ TỪ CSV ===
# Forward_m (hệ thống đo được) cho từng mốc thước dây
data_3m = [3.106, 3.132, 3.080, 3.081, 3.105, 3.047, 3.144, 3.088, 3.116, 3.151]
data_3_5m = [3.898, 3.903, 3.923, 3.898, 3.910, 3.977, 3.854, 3.879, 3.868, 3.879]
data_4m = [4.360, 4.356, 4.404, 4.412, 4.382, 4.346, 4.412, 4.349, 4.431, 4.409]
data_4_5m = [5.246, 5.338, 5.354, 5.292, 5.379, 5.273, 5.360, 5.287, 5.316, 5.210]

# Z_raw cho từng mốc
zraw_3m = [3.674, 3.719, 3.674, 3.674, 3.696, 3.651, 3.719, 3.674, 3.696, 3.742]
zraw_3_5m = [4.457, 4.457, 4.490, 4.457, 4.490, 4.557, 4.424, 4.457, 4.424, 4.457]
zraw_4m = [4.928, 4.888, 4.968, 4.968, 4.928, 4.849, 4.968, 4.849, 4.968, 4.968]
zraw_4_5m = [5.773, 5.828, 5.885, 5.828, 5.885, 5.828, 5.885, 5.828, 5.828, 5.718]

print("=" * 60)
print("📊 PHÂN TÍCH DỮ LIỆU HIỆU CHUẨN TN3")
print("=" * 60)

# === BẢNG THỐNG KÊ ===
labels = ["3.0m", "3.5m", "4.0m", "4.5m"]
actuals = [3.0, 3.5, 4.0, 4.5]
all_fwd = [data_3m, data_3_5m, data_4m, data_4_5m]
all_zraw = [zraw_3m, zraw_3_5m, zraw_4m, zraw_4_5m]

print(f"\n{'Mốc':<8} {'Forward_m TB':>14} {'Z_raw TB':>10} {'Sai số Fwd':>12} {'Sai số %':>10}")
print("-" * 60)
for i, label in enumerate(labels):
    mean_fwd = np.mean(all_fwd[i])
    mean_zraw = np.mean(all_zraw[i])
    error = mean_fwd - actuals[i]
    pct = error / actuals[i] * 100
    print(f"{label:<8} {mean_fwd:>14.3f} {mean_zraw:>10.3f} {error:>+12.3f} {pct:>+10.1f}%")

# === FIT ĐA THỨC: actual = f(forward_m_raw) ===
# Gộp tất cả dữ liệu
x_all = []  # forward_m đo được (raw)
y_all = []  # khoảng cách thực (thước dây)
for i, actual in enumerate(actuals):
    for val in all_fwd[i]:
        x_all.append(val)
        y_all.append(actual)

x_all = np.array(x_all)
y_all = np.array(y_all)

# Fit bậc 1 (tuyến tính)
coeffs1 = np.polyfit(x_all, y_all, 1)
pred1 = np.polyval(coeffs1, x_all)
rmse1 = np.sqrt(np.mean((pred1 - y_all) ** 2))

# Fit bậc 2 (parabol)
coeffs2 = np.polyfit(x_all, y_all, 2)
pred2 = np.polyval(coeffs2, x_all)
rmse2 = np.sqrt(np.mean((pred2 - y_all) ** 2))

print(f"\n{'=' * 60}")
print(f"🔬 KẾT QUẢ FIT ĐA THỨC (forward_m)")
print(f"{'=' * 60}")
print(f"\n📐 Bậc 1 (Tuyến tính): y = {coeffs1[0]:.6f}*x + ({coeffs1[1]:.6f})")
print(f"   RMSE = {rmse1:.4f}m")
print(f"\n📐 Bậc 2 (Parabol):    y = {coeffs2[0]:.6f}*x² + ({coeffs2[1]:.6f})*x + ({coeffs2[2]:.6f})")
print(f"   RMSE = {rmse2:.4f}m")

best = "Bậc 2" if rmse2 < rmse1 else "Bậc 1"
print(f"\n✅ Hàm tốt nhất: {best}")

# === FIT TRÊN Z_RAW (để áp dụng trước pitch projection) ===
# Tính z_opt_true từ thước dây: true_fwd_cam = tape + 0.26
# true_fwd_cam ≈ z_opt_true * cos(20°) => z_opt_true ≈ true_fwd_cam / cos(20°)
import math
cos20 = math.cos(math.radians(20.0))

xz_all = []
yz_all = []
for i, actual in enumerate(actuals):
    true_fwd_cam = actual + 0.26
    true_z = true_fwd_cam / cos20  # Xấp xỉ (bỏ qua y_opt nhỏ)
    for val in all_zraw[i]:
        xz_all.append(val)
        yz_all.append(true_z)

xz_all = np.array(xz_all)
yz_all = np.array(yz_all)

coeffs_z1 = np.polyfit(xz_all, yz_all, 1)
pred_z1 = np.polyval(coeffs_z1, xz_all)
rmse_z1 = np.sqrt(np.mean((pred_z1 - yz_all) ** 2))

coeffs_z2 = np.polyfit(xz_all, yz_all, 2)
pred_z2 = np.polyval(coeffs_z2, xz_all)
rmse_z2 = np.sqrt(np.mean((pred_z2 - yz_all) ** 2))

print(f"\n{'=' * 60}")
print(f"🔬 KẾT QUẢ FIT ĐA THỨC (z_opt - trước pitch)")
print(f"{'=' * 60}")
print(f"\n📐 Bậc 1: z_corrected = {coeffs_z1[0]:.6f}*z_raw + ({coeffs_z1[1]:.6f})")
print(f"   RMSE = {rmse_z1:.4f}m")
print(f"\n📐 Bậc 2: z_corrected = {coeffs_z2[0]:.6f}*z² + ({coeffs_z2[1]:.6f})*z + ({coeffs_z2[2]:.6f})")
print(f"   RMSE = {rmse_z2:.4f}m")

# === KIỂM CHỨNG SAU BÙ TRỪ ===
print(f"\n{'=' * 60}")
print(f"📋 KIỂM CHỨNG SAU KHI ÁP DỤNG HÀM BÙ TRỪ (Bậc 2 trên forward_m)")
print(f"{'=' * 60}")
a, b, c = coeffs2
print(f"   Công thức: forward_corrected = {a:.6f}*fwd² + ({b:.6f})*fwd + ({c:.6f})")
print(f"\n{'Mốc':<8} {'Fwd Raw TB':>12} {'Fwd Corrected':>14} {'Sai số':>10}")
print("-" * 50)
for i, label in enumerate(labels):
    mean_raw = np.mean(all_fwd[i])
    corrected = a * mean_raw**2 + b * mean_raw + c
    error = corrected - actuals[i]
    print(f"{label:<8} {mean_raw:>12.3f} {corrected:>14.3f} {error:>+10.3f}")

print(f"\n{'=' * 60}")
print(f"📋 KIỂM CHỨNG SAU KHI ÁP DỤNG HÀM BÙ TRỪ (Bậc 2 trên z_opt)")
print(f"{'=' * 60}")
az, bz, cz = coeffs_z2
print(f"   Công thức: z_corrected = {az:.6f}*z² + ({bz:.6f})*z + ({cz:.6f})")
print(f"\n{'Mốc':<8} {'Z_raw TB':>10} {'Z_corrected':>12} {'Fwd Result':>12} {'Sai số':>10}")
print("-" * 60)
for i, label in enumerate(labels):
    mean_z = np.mean(all_zraw[i])
    z_corr = az * mean_z**2 + bz * mean_z + cz
    fwd_result = z_corr * cos20 - 0.26  # Simplified (ignoring y_opt)
    error = fwd_result - actuals[i]
    print(f"{label:<8} {mean_z:>10.3f} {z_corr:>12.3f} {fwd_result:>12.3f} {error:>+10.3f}")

print(f"\n{'=' * 60}")
print(f"🏆 CODE ĐỂ DÁN VÀO tn3.py (thay thế Far Mode):")
print(f"{'=' * 60}")
print(f"""
# HIỆU CHUẨN THỰC NGHIỆM: Hàm bù trừ Bậc 2 (Polynomial Fit)
# Fit từ dữ liệu TN3 calibration {len(x_all)} mẫu, RMSE={rmse2:.4f}m
a_cal, b_cal, c_cal = {a:.6f}, {b:.6f}, {c:.6f}
forward_m_corrected = a_cal * forward_m**2 + b_cal * forward_m + c_cal
forward_m = forward_m_corrected
""")
