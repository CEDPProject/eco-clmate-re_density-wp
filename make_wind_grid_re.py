# -*- coding: utf-8 -*-

"""
KIM LDAPS U80/V80 기반
풍력 에너지밀도 및 3MW 터빈 발전량 계산 (2월 ~ 8월 월평균 배치 처리)

입력:
    KIM LDAPS nc 파일 (/BASE_DIR/2026/MM/DD/12/l010_v040_korea_etc.2byte.ft*.nc)
출력:
    output/wind_grid/2026_02.npz ~ 2026_08.npz (density, e_turbine 포함)
"""

import os
import glob
import netCDF4 as nc
import numpy as np


# =====================================================
# PATH 및 물리 상수 / 터빈 제원 설정
# =====================================================

BASE_DIR = "/data/jyy/S82_2025/KIML_monthly"
OUT_DIR = os.path.join(BASE_DIR, "output", "wind_grid")

os.makedirs(OUT_DIR, exist_ok=True)

HUB_HEIGHT = 80
RHO = 1.225          # 표준 공기 밀도 (kg/m3)

# 3MW 풍력 터빈 제원 (Power Curve 조건)
R = 50.0             # 블레이드 반경 (m)
A = np.pi * (R ** 2)  # 로터 수광 면적 (m2) ≈ 7853.98 m2
CP = 0.476           # 최대 출력 계수

V_CUT_IN = 4.5       # Cut-in 풍속 (m/s)
V_RATED = 14.5       # Rated 풍속 (m/s)
V_CUT_OUT = 25.0     # Cut-out 풍속 (m/s)
P_RATED_KW = 3000.0  # 정격 출력 (kW)

MONTHS = ["02", "03", "04", "05", "06", "07", "08"]


# =====================================================
# 3MW 터빈 Power Curve 연산 함수
# =====================================================

def calculate_turbine_power_kw(v):
    """
    풍속(v, m/s)에 따른 3MW 터빈 출력(kW) 계산
    """
    p_kw = np.zeros_like(v, dtype=np.float32)
    
    # 1) 구동 영역 (4.5 <= v < 14.5 m/s)
    mask_gen = (v >= V_CUT_IN) & (v < V_RATED)
    p_kw[mask_gen] = (0.5 * RHO * A * (v[mask_gen] ** 3) * CP) / 1000.0
    
    # 2) 정격 출력 영역 (14.5 <= v <= 25.0 m/s) -> 3000 kW 고정
    mask_rated = (v >= V_RATED) & (v <= V_CUT_OUT)
    p_kw[mask_rated] = P_RATED_KW
    
    # 3) Cut-in 미만 / Cut-out 초과 -> 0 kW (자동 반영)
    return p_kw


# =====================================================
# 월별 배치 처리
# =====================================================

for mm in MONTHS:
    year = "2026"
    month_key = f"{year}_{mm}"
    
    day_dirs = sorted(glob.glob(os.path.join(BASE_DIR, year, mm, "*")))

    print(f"\n==========================================")
    print(f"PROCESS MONTH: {month_key}")
    print(f"FOUND DAYS: {len(day_dirs)}")
    print(f"==========================================")

    if len(day_dirs) == 0:
        print(f"⚠️ [WARNING] Day directories not found for {month_key}. Skipping...")
        continue

    sample_files = glob.glob(os.path.join(day_dirs[0], "*", "l010_v040_korea_etc.2byte.ft*.nc"))
    if not sample_files:
        print(f"⚠️ [WARNING] No sample file found for {month_key}. Skipping...")
        continue

    with nc.Dataset(sample_files[0]) as ds:
        lat = np.ma.filled(ds.variables["XLAT"][0, :, :], np.nan).astype(np.float32)
        lon = np.ma.filled(ds.variables["XLONG"][0, :, :], np.nan).astype(np.float32)

    daily_density_list = []    # 일별 에너지밀도 (kWh/m2/day)
    daily_e_turbine_list = []  # 일별 3MW 터빈 1기 발전량 (kWh/day)

    # 일(Day) 단위 순회
    for d_dir in day_dirs:
        files = sorted(glob.glob(os.path.join(d_dir, "12", "l010_v040_korea_etc.2byte.ft*.nc")))
        
        if len(files) < 24:
            continue

        # 24시간 누적용 배열 초기화
        daily_Wh = np.zeros_like(lat, dtype=np.float32)
        daily_e_turbine_kwh = np.zeros_like(lat, dtype=np.float32)

        # 24시간 적산
        for f in files:
            with nc.Dataset(f) as ds:
                u80 = np.ma.filled(ds.variables["U80"][0, :, :], np.nan).astype(np.float32)
                v80 = np.ma.filled(ds.variables["V80"][0, :, :], np.nan).astype(np.float32)

                wind_speed = np.sqrt(u80 * u80 + v80 * v80)

                # 1) 풍력 에너지 밀도 적산 (W/m2)
                power_density = 0.5 * RHO * (wind_speed ** 3)
                daily_Wh += np.where(np.isfinite(power_density), power_density, 0)

                # 2) 3MW 터빈 1기 발전량 적산 (kW * 1h = kWh)
                turbine_kw = calculate_turbine_power_kw(wind_speed)
                daily_e_turbine_kwh += np.where(np.isfinite(turbine_kw), turbine_kw, 0)

        daily_density_list.append(daily_Wh / 1000.0)
        daily_e_turbine_list.append(daily_e_turbine_kwh)

    if not daily_density_list:
        print(f"⚠️ [WARNING] No valid daily data processed for {month_key}.")
        continue

    # 월평균 계산
    monthly_avg_density = np.mean(daily_density_list, axis=0)
    monthly_avg_density = np.where(np.isfinite(monthly_avg_density), monthly_avg_density, np.nan)

    monthly_avg_e_turbine = np.mean(daily_e_turbine_list, axis=0)
    monthly_avg_e_turbine = np.where(np.isfinite(monthly_avg_e_turbine), monthly_avg_e_turbine, np.nan)

    # 통계 출력
    print(f"[{month_key}] Wind density (kWh/m2/day) mean: {np.nanmean(monthly_avg_density):.4f}")
    print(f"[{month_key}] 3MW E_turbine (kWh/day)     mean: {np.nanmean(monthly_avg_e_turbine):.4f}")

    # ★ [핵심] e_turbine 추가하여 .npz 저장
    outfile = os.path.join(OUT_DIR, f"{month_key}.npz")
    np.savez(
        outfile,
        lat=lat,
        lon=lon,
        density=monthly_avg_density,
        e_turbine=monthly_avg_e_turbine,  # ★ 2단계 연산용 필드 추가
        hub_height_m=HUB_HEIGHT
    )
    print(f"SAVE: {outfile}")

print("\nALL MONTHS (02~08) COMPLETED SUCCESSFULLY!")