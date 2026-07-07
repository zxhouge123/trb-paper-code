# Data Summary — TRB Traffic Speed Analysis

## Dataset Overview

This project analyzes traffic speed data on the **G20 expressway** (upstream direction, from **294.385 km** to **321.885 km**). Two primary data files are included:

---

## 1. CSV File: G20_上行_294.385-321.885.csv

### Basic Info
| Property | Value |
|---|---|
| **Rows** | 195,635 |
| **Columns** | 40 |
| **Road** | G20, upstream direction |
| **Segment Range** | 294.385 km – 321.885 km |
| **Time Period** | 2021-10-20 00:00 to 2021-12-31 23:30 |
| **Unique Time Entries** | 3,504 (5-minute intervals over 48 days) |
| **Unique Road Segments (ldbh)** | 56 |

### Key Numeric Columns
| Column | Description | Range |
|---|---|---|
| **Speed** | Average speed (km/h) | 18.24 – 113.64 |
| **Speed_std** | Speed standard deviation | – |
| **CI** | Congestion Index | 0.91 – 5.70 |
| **TOI** | Traffic Operation Index | 0.1821 – 0.9842 |
| **flow_percent** | Flow percentage | – |
| **Num_signs** | Number of traffic signs | – |

### Notable Columns
- ldbh / ldqdzh / ldzdzh: Road segment ID, start & end mileage
- qd_LON, qd_LAT, zd_LON, zd_LAT: GPS coordinates of segment start/end
- Slope, Changing_slope_angle: Road geometry
- Type_cross_section, Type_link: Road type attributes
- Num_entrances&exits, Pro_influ_entrances, Pro_influ_exits: Entrance/exit info
- SH_* columns: Speed-harmonic-related features (L-turn, R-turn, merge, acceleration, deceleration)
- TOItype: TOI category label
- 	ime: Timestamp at 5-minute resolution

---

## 2. NPZ File: speed_tensor.npz

### Structure
| Key | Shape | Dtype | Description |
|---|---|---|---|
| **speed** | (56, 73, 48) | float32 | Speed tensor: 56 segments × 73 intraday time slots × 48 days |
| **obs_mask** | (56, 73, 48) | float32 | Observation mask: 1 = observed, 0 = missing |
| **segs** | (56,) | <U14 | Road segment IDs (e.g., G20-0294.385-1) |

### Speed Tensor Statistics
| Metric | Value |
|---|---|
| **Valid Speed Range** | 18.24 – 113.64 km/h |
| **Mean Speed** | 94.12 km/h |
| **Std Dev** | 10.33 km/h |
| **Missing (NaN) Ratio** | 0.30% (589 / 196,224) |
| **Observation Rate** | 99.70% |

### Road Segments
56 segments covering G20 from **G20-0294.385-1** to **G20-0321.885-1**, spaced at 0.5 km intervals (every 500 meters).

### Tensor Dimensions
- **Axis 0 (Segments):** 56 road sections along the G20 expressway
- **Axis 1 (Time-of-day):** 73 intraday slots (5-minute intervals, covering ~06:00–18:00 or peak hours)
- **Axis 2 (Days):** 48 days from Oct 20 to Dec 31, 2021

---

## File Size Summary

| File | Size |
|---|---|
| CSV (raw data) | ~57 MB |
| NPZ (tensor) | ~270 KB |
| Python scripts | ~40 KB total |

*Note: Large data files (*.csv, *.npz) are excluded from version control.*
