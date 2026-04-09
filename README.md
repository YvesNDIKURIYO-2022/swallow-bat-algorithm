# A Hybrid Memetic Framework with Threat-Aware Evasion for Container Truck Routing in High-Risk Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Overview

This repository contains the implementation of the **Hybrid Memetic Framework with Threat-Aware Evasion**, a novel bio-inspired hybrid metaheuristic designed for solving **Threat-Aware Container Truck Routing Problems (CTRP)**. The algorithm uniquely combines swallow flocking behavior (for global exploration) with bat echolocation (for local exploitation), incorporating a dedicated threat-aware evasion operator to proactively mitigate risks in logistics routing.

This repository serves as the **official code and data supplement** for the manuscript:

> Ndikuriyo, Y., Zhang, Y., & Fom, D. D. (2026). *A Hybrid Memetic Framework with Threat-Aware Evasion for Container Truck Routing in High-Risk Environments.*

---

## 🎯 Key Features

- **Hybrid Memetic Framework**: Merges swarm intelligence principles from flocking dynamics and frequency-modulated search.
- **Threat-Aware Evasion Operator**: Proactively steers solutions away from spatial threats during optimization (not post-hoc penalty).
- **Multi-Objective Optimization**: Simultaneously minimizes operational cost, travel distance, and threat exposure.
- **Comprehensive Benchmarking**: Validated on modified Augerat instances, Set X (Uchoa et al., 2017), and Set XL (Queiroga et al., 2026).
- **Real-World Validation**: East African Community (EAC) corridor case study with 28 cities and 16 threat zones.
- **High Performance**: Outperforms ALNS, HGA, PSO, BA, and HHO across all metrics.

---

## 🚀 Algorithm Components

### 1. Global Exploration: Flocking-Inspired Mechanism
- **Separation**: Prevents overcrowding to maintain solution diversity.
- **Alignment**: Synchronizes velocity among neighboring agents for coordinated search.
- **Cohesion**: Promotes movement toward the local group center to balance exploration.

### 2. Local Exploitation: Frequency-Modulated Search
- **Frequency Adaptation**: Dynamically adjusts search frequency to balance exploration and exploitation.
- **Velocity Update**: Guides movement toward globally optimal solutions.
- **Local Search Refinement**: Uses Gaussian perturbation around best positions.

### 3. Threat-Aware Evasion: Domain Knowledge Meme
- **Proactive Risk Mitigation**: Dynamically steers routes away from hazardous zones during search.
- **Distance-Weighted Repulsion**: Stronger evasion when near threat zones; negligible when far.
- **Static Threat Modeling**: Circular restricted zones with fixed centers and radii.

---

## 📊 Benchmark Instance Specifications

This section documents all benchmark instances used in the experimental evaluation, organized hierarchically by problem scale.

### S1. Modified Capacitated Vehicle Routing Datasets (Augerat et al.)

Three instances from the Augerat et al. benchmark suite were adapted into threat-aware CTRP formulations by inserting synthetic threat zones representing operational hazards.

**Table S1. Modified Augerat benchmark instances**

| Instance | Customers | Depot Coordinates | Capacity | Vehicles | Threat Zones |
|:---------|----------:|:-----------------:|---------:|---------:|-------------:|
| A-n32-k5 | 21 | (82, 76) | 100 | 3 | 5 |
| A-n53-k7 | 34 | (24, 63) | 100 | 5 | 5 |
| A-n80-k10 | 51 | (92, 92) | 100 | 7 | 6 |

*Note: Instance naming follows A-n[total nodes]-k[vehicles]. Customers = n − 1 (excluding depot).*

---

### S2. Large-Scale Set X Instances (Uchoa et al., 2017)

Eight representative instances from Set X, ranging from 100 to 1,000 customers. These instances feature clustered customer distributions and uniform vehicle capacity (10,000 units).

**Table S2. Set X instances with tier classifications**

| Instance | Customers | Vehicles | Capacity | Known Optimum | Tier |
|:---------|----------:|---------:|---------:|--------------:|:-----|
| X-n101-k25 | 100 | 25 | 10,000 | 27,555 | Small |
| X-n200-k8 | 199 | 8 | 10,000 | 33,382 | Small |
| X-n300-k10 | 299 | 10 | 10,000 | 104,952 | Medium |
| X-n400-k12 | 399 | 12 | 10,000 | 310,696 | Medium |
| X-n500-k12 | 499 | 12 | 10,000 | 381,127 | Medium |
| X-n600-k12 | 599 | 12 | 10,000 | 473,968 | Large |
| X-n800-k12 | 799 | 12 | 10,000 | 642,510 | Large |
| X-n1000-k12 | 999 | 12 | 10,000 | 812,410 | Large |

**Set X Generation Characteristics:**
- Depot: (500, 500) in 1000×1000 Euclidean space
- Clusters: 5–20 cluster centers (∼1 per 50 customers)
- Spatial dispersion: Gaussian perturbation (σ = 50)
- Demands: Uniform [1, 20]

---

### S3. Ultra-Large Set XL Instances (Queiroga et al., 2026)

A stratified representative sample of 20 instances drawn from the Set XL benchmark suite (100 instances total, 1,000–10,000 customers). Instances are organized into four size tiers.

**Table S3. Representative sample of Set XL instances**

| Instance | Customers | Vehicles | Capacity | Best Known Solution (BKS) | Tier |
|:---------|----------:|---------:|---------:|-------------------------:|:-----|
| XL-n1094-k157 | 1,093 | 157 | 7 | 112,431 | Small |
| XL-n1328-k19 | 1,327 | 19 | 542 | 38,247 | Small |
| XL-n1654-k11 | 1,653 | 11 | 845 | 36,385 | Small |
| XL-n1981-k13 | 1,980 | 13 | 832 | 32,580 | Small |
| XL-n2307-k34 | 2,306 | 34 | 479 | 47,958 | Medium |
| XL-n2634-k17 | 2,633 | 17 | 898 | 31,641 | Medium |
| XL-n2961-k55 | 2,960 | 55 | 297 | 108,084 | Medium |
| XL-n3287-k30 | 3,286 | 30 | 111 | 40,229 | Medium |
| XL-n3804-k29 | 3,803 | 29 | 10,064 | 52,885 | Medium |
| XL-n4436-k48 | 4,435 | 48 | 706 | 61,477 | Medium |
| XL-n5061-k184 | 5,060 | 184 | 206 | 161,629 | Large |
| XL-n5526-k553 | 5,525 | 553 | 10 | 336,898 | Large |
| XL-n6034-k61 | 6,033 | 61 | 744 | 64,448 | Large |
| XL-n6588-k473 | 6,587 | 473 | 76 | 334,068 | Large |
| XL-n7037-k38 | 7,036 | 38 | 187 | 70,845 | Large |
| XL-n7683-k602 | 7,682 | 602 | 957 | 702,098 | Large |
| XL-n8207-k108 | 8,206 | 108 | 415 | 118,274 | Extra-Large |
| XL-n8766-k1032 | 8,765 | 1,032 | 637 | 906,406 | Extra-Large |
| XL-n9363-k209 | 9,362 | 209 | 45 | 205,575 | Extra-Large |
| XL-n10001-k1570 | 10,000 | 1,570 | 479 | 2,333,757 | Extra-Large |

**Table S4. Tier-level summary**

| Tier | Count | Min Customers | Max Customers | Mean Customers |
|:-----|------:|--------------:|--------------:|---------------:|
| Small | 4 | 1,093 | 1,980 | 1,513 |
| Medium | 6 | 2,306 | 4,435 | 3,237 |
| Large | 6 | 5,060 | 7,682 | 6,320 |
| Extra-Large | 4 | 8,206 | 10,000 | 9,083 |
| **Total** | **20** | — | — | **4,987** |

**Table S5. Quick test subset (for rapid validation)**

| Instance | Customers | Vehicles | Capacity | BKS | Tier |
|:---------|----------:|---------:|---------:|--------:|:-----|
| XL-n1328-k19 | 1,327 | 19 | 542 | 38,247 | Small |
| XL-n2961-k55 | 2,960 | 55 | 297 | 108,084 | Medium |
| XL-n5061-k184 | 5,060 | 184 | 206 | 161,629 | Large |
| XL-n7037-k38 | 7,036 | 38 | 187 | 70,845 | Large |
| XL-n8766-k1032 | 8,765 | 1,032 | 637 | 906,406 | Extra-Large |
| XL-n10001-k1570 | 10,000 | 1,570 | 479 | 2,333,757 | Extra-Large |

**Set XL Generation Characteristics:**
- Depot: (500, 500) in 1000×1000 Euclidean space
- Clusters: 2–6 cluster centers (∼1 per 200 customers)
- Spatial dispersion: Gaussian perturbation (σ = 60)
- Demands: Uniform [1, 100]

<details>
<summary><b>📎 Click to expand: Complete Set XL Collection (100 instances)</b></summary>

| Instance | Customers | Vehicles | Capacity | BKS |
|:---------|----------:|---------:|---------:|--------:|
| XL-n1048-k237 | 1,047 | 237 | 128 | 380,211 |
| XL-n1094-k157 | 1,093 | 157 | 7 | 112,431 |
| XL-n1141-k112 | 1,140 | 112 | 761 | 95,727 |
| XL-n1188-k96 | 1,187 | 96 | 782 | 104,415 |
| XL-n1234-k55 | 1,233 | 55 | 126 | 96,647 |
| XL-n1281-k29 | 1,280 | 29 | 2,267 | 31,101 |
| XL-n1328-k19 | 1,327 | 19 | 542 | 38,247 |
| XL-n1374-k278 | 1,373 | 278 | 248 | 233,049 |
| XL-n1421-k232 | 1,420 | 232 | 309 | 384,826 |
| XL-n1468-k151 | 1,467 | 151 | 726 | 250,166 |
| XL-n1514-k106 | 1,513 | 106 | 107 | 92,425 |
| XL-n1561-k75 | 1,560 | 75 | 21 | 101,549 |
| XL-n1608-k39 | 1,607 | 39 | 337 | 48,021 |
| XL-n1654-k11 | 1,653 | 11 | 845 | 36,385 |
| XL-n1701-k562 | 1,700 | 562 | 227 | 521,136 |
| XL-n1748-k271 | 1,747 | 271 | 270 | 173,896 |
| XL-n1794-k163 | 1,793 | 163 | 11 | 141,729 |
| XL-n1841-k126 | 1,840 | 126 | 186 | 214,038 |
| XL-n1888-k82 | 1,887 | 82 | 173 | 143,623 |
| XL-n1934-k46 | 1,933 | 46 | 2,166 | 53,013 |
| XL-n1981-k13 | 1,980 | 13 | 832 | 32,580 |
| XL-n2028-k617 | 2,027 | 617 | 247 | 544,403 |
| XL-n2074-k264 | 2,073 | 264 | 401 | 421,627 |
| XL-n2121-k186 | 2,120 | 186 | 62 | 283,211 |
| XL-n2168-k138 | 2,167 | 138 | 800 | 127,298 |
| XL-n2214-k131 | 2,213 | 131 | 17 | 154,676 |
| XL-n2261-k54 | 2,260 | 54 | 319 | 98,907 |
| XL-n2307-k34 | 2,306 | 34 | 479 | 47,958 |
| XL-n2354-k631 | 2,353 | 631 | 28 | 940,825 |
| XL-n2401-k408 | 2,400 | 408 | 303 | 463,473 |
| XL-n2447-k290 | 2,446 | 290 | 150 | 218,706 |
| XL-n2494-k194 | 2,493 | 194 | 661 | 361,205 |
| XL-n2541-k121 | 2,540 | 121 | 21 | 146,390 |
| XL-n2587-k66 | 2,586 | 66 | 2,986 | 73,394 |
| XL-n2634-k17 | 2,633 | 17 | 898 | 31,641 |
| XL-n2681-k540 | 2,680 | 540 | 251 | 798,603 |
| XL-n2727-k546 | 2,726 | 546 | 5 | 431,134 |
| XL-n2774-k286 | 2,773 | 286 | 731 | 407,847 |
| XL-n2821-k208 | 2,820 | 208 | 179 | 216,763 |
| XL-n2867-k120 | 2,866 | 120 | 180 | 165,990 |
| XL-n2914-k95 | 2,913 | 95 | 1,663 | 88,990 |
| XL-n2961-k55 | 2,960 | 55 | 297 | 108,084 |
| XL-n3007-k658 | 3,006 | 658 | 25 | 522,319 |
| XL-n3054-k461 | 3,053 | 461 | 497 | 782,739 |
| XL-n3101-k311 | 3,100 | 311 | 159 | 245,937 |
| XL-n3147-k232 | 3,146 | 232 | 102 | 256,626 |
| XL-n3194-k161 | 3,193 | 161 | 1,012 | 148,728 |
| XL-n3241-k115 | 3,240 | 115 | 1,404 | 221,370 |
| XL-n3287-k30 | 3,286 | 30 | 111 | 40,229 |
| XL-n3334-k934 | 3,333 | 934 | 20 | 1,452,698 |
| XL-n3408-k524 | 3,407 | 524 | 353 | 678,643 |
| XL-n3484-k436 | 3,483 | 436 | 8 | 703,355 |
| XL-n3561-k229 | 3,560 | 229 | 779 | 209,386 |
| XL-n3640-k211 | 3,639 | 211 | 130 | 189,724 |
| XL-n3721-k77 | 3,720 | 77 | 371 | 162,862 |
| XL-n3804-k29 | 3,803 | 29 | 10,064 | 52,885 |
| XL-n3888-k1010 | 3,887 | 1,010 | 128 | 1,880,368 |
| XL-n3975-k687 | 3,974 | 687 | 32 | 525,901 |
| XL-n4063-k347 | 4,062 | 347 | 598 | 548,931 |
| XL-n4153-k291 | 4,152 | 291 | 726 | 356,034 |
| XL-n4245-k203 | 4,244 | 203 | 21 | 229,659 |
| XL-n4340-k148 | 4,339 | 148 | 2,204 | 244,226 |
| XL-n4436-k48 | 4,435 | 48 | 706 | 61,477 |
| XL-n4535-k1134 | 4,534 | 1,134 | 4 | 1,203,566 |
| XL-n4635-k790 | 4,634 | 790 | 294 | 610,650 |
| XL-n4738-k487 | 4,737 | 487 | 499 | 760,501 |
| XL-n4844-k321 | 4,843 | 321 | 188 | 404,652 |
| XL-n4951-k203 | 4,950 | 203 | 1,848 | 285,269 |
| XL-n5061-k184 | 5,060 | 184 | 206 | 161,629 |
| XL-n5174-k55 | 5,173 | 55 | 520 | 61,382 |
| XL-n5288-k1246 | 5,287 | 1,246 | 318 | 1,960,101 |
| XL-n5406-k783 | 5,405 | 783 | 38 | 1,040,536 |
| XL-n5526-k553 | 5,525 | 553 | 10 | 336,898 |
| XL-n5649-k401 | 5,648 | 401 | 181 | 644,866 |
| XL-n5774-k290 | 5,773 | 290 | 1,012 | 250,207 |
| XL-n5902-k122 | 5,901 | 122 | 2,663 | 217,447 |
| XL-n6034-k61 | 6,033 | 61 | 744 | 64,448 |
| XL-n6168-k1922 | 6,167 | 1,922 | 162 | 1,530,010 |
| XL-n6305-k1042 | 6,304 | 1,042 | 268 | 1,177,528 |
| XL-n6445-k628 | 6,444 | 628 | 77 | 996,623 |
| XL-n6588-k473 | 6,587 | 473 | 76 | 334,068 |
| XL-n6734-k330 | 6,733 | 330 | 1,534 | 448,031 |
| XL-n6884-k148 | 6,883 | 148 | 357 | 181,809 |
| XL-n7037-k38 | 7,036 | 38 | 187 | 70,845 |
| XL-n7193-k1683 | 7,192 | 1,683 | 32 | 2,958,979 |
| XL-n7353-k1471 | 7,352 | 1,471 | 5 | 1,537,811 |
| XL-n7516-k859 | 7,515 | 859 | 439 | 573,902 |
| XL-n7683-k602 | 7,682 | 602 | 957 | 702,098 |
| XL-n7854-k365 | 7,853 | 365 | 223 | 659,221 |
| XL-n8028-k294 | 8,027 | 294 | 1,386 | 266,900 |
| XL-n8207-k108 | 8,206 | 108 | 415 | 118,274 |
| XL-n8389-k2028 | 8,388 | 2,028 | 208 | 3,358,731 |
| XL-n8575-k1297 | 8,574 | 1,297 | 36 | 1,089,137 |
| XL-n8766-k1032 | 8,765 | 1,032 | 637 | 906,406 |
| XL-n8960-k634 | 8,959 | 634 | 106 | 773,383 |
| XL-n9160-k379 | 9,159 | 379 | 237 | 324,092 |
| XL-n9363-k209 | 9,362 | 209 | 45 | 205,575 |
| XL-n9571-k55 | 9,570 | 55 | 8,773 | 106,791 |
| XL-n9784-k2774 | 9,783 | 2,774 | 19 | 4,078,217 |
| XL-n10001-k1570 | 10,000 | 1,570 | 479 | 2,333,757 |

</details>

---

### S4. Threat Zone Configuration

All instances use a standardized configuration of 10 threat zones, defined with normalized coordinates and scaled to each instance's spatial bounds.

**Table S6. Normalized threat zone definitions**

| Zone ID | Center (x, y) | Radius | Description |
|:--------|:-------------:|-------:|:------------|
| T1 | (0.25, 0.25) | 0.08 | Urban congestion |
| T2 | (0.75, 0.25) | 0.08 | Industrial accident |
| T3 | (0.25, 0.75) | 0.08 | Flood-prone area |
| T4 | (0.75, 0.75) | 0.08 | Conflict zone |
| T5 | (0.50, 0.50) | 0.10 | Central high-risk |
| T6 | (0.20, 0.50) | 0.06 | Road construction |
| T7 | (0.80, 0.50) | 0.06 | Landslide risk |
| T8 | (0.50, 0.20) | 0.06 | Port security |
| T9 | (0.50, 0.80) | 0.06 | Environmental protection |
| T10 | (0.35, 0.65) | 0.05 | Border checkpoint |

**Scaling Procedure:** For an instance with bounds $(x_{\min}, y_{\min})$ and $(x_{\max}, y_{\max})$:
- Center: $c_x = x_{\min} + n_x \cdot (x_{\max} - x_{\min})$
- Center: $c_y = y_{\min} + n_y \cdot (y_{\max} - y_{\min})$
- Radius: $r = n_r \cdot \max(x_{\max} - x_{\min}, y_{\max} - y_{\min})$

---

### S5. East African Community Case Study Data

The real-world validation uses a digital twin of the EAC logistics network:

- **28 cities** across Kenya, Uganda, Tanzania, Rwanda, Burundi
- **16 threat zones** (security, infrastructure, climate hazards)
- **5 vehicles** with 280-ton capacity each
- **1,118 tons** aggregate demand

**Table S7. EAC city coordinates and demands**

| City | Latitude | Longitude | Demand (tons) | Corridor | In Threat Zone |
|:-----|---------:|----------:|--------------:|:---------|:--------------:|
| Mombasa | -4.0435 | 39.6682 | 0 | Northern | No |
| Nairobi | -1.2921 | 36.8219 | 65 | Northern | Yes |
| Nakuru | -0.3031 | 36.0800 | 32 | Northern | No |
| Eldoret | 0.5204 | 35.2697 | 26 | Northern | No |
| Kisumu | -0.0917 | 34.7679 | 42 | Northern | No |
| Thika | -1.0333 | 37.0833 | 40 | Connecting | Yes |
| Machakos | -1.5167 | 37.2667 | 39 | Connecting | Yes |
| Embu | -0.5333 | 37.4500 | 33 | Connecting | No |
| Dar es Salaam | -6.7924 | 39.2083 | 31 | Central | No |
| Morogoro | -6.8167 | 37.6667 | 68 | Central | No |
| Dodoma | -6.1620 | 35.7516 | 59 | Central | Yes |
| Tanga | -5.0667 | 39.1000 | 30 | Central | Yes |
| Arusha | -3.3869 | 36.6830 | 62 | Connecting | No |
| Moshi | -3.3348 | 37.3404 | 52 | Connecting | No |
| Singida | -4.8167 | 34.7500 | 27 | Connecting | No |
| Kampala | 0.3476 | 32.5825 | 26 | Northern | No |
| Entebbe | 0.0500 | 32.4600 | 30 | Northern | No |
| Jinja | 0.4244 | 33.2042 | 38 | Northern | No |
| Mbale | 1.0806 | 34.1753 | 39 | Northern | No |
| Tororo | 0.6833 | 34.1667 | 57 | Connecting | No |
| Masaka | -0.3333 | 31.7333 | 63 | Connecting | No |
| Kigali | -1.9706 | 30.1044 | 26 | Connecting | No |
| Huye | -2.6000 | 29.7500 | 60 | Connecting | No |
| Bujumbura | -3.3614 | 29.3599 | 37 | Central | No |
| Gitega | -3.4264 | 29.9306 | 70 | Connecting | No |
| Ngozi | -2.9075 | 29.8306 | 66 | Connecting | No |

**Table S8. EAC threat zone specifications**

| Zone Name | Type | Risk Level | Center Lat | Center Lon | Radius (km) |
|:----------|:-----|:-----------|-----------:|-----------:|------------:|
| M23 Rebel Activity - Rutshuru | Security | Very High | -1.40 | 28.80 | 80 |
| M23 Controlled Areas - Masisi | Security | Very High | -1.60 | 29.20 | 60 |
| M23 Presence - Goma Perimeter | Security | Very High | -1.68 | 29.22 | 40 |
| ADF Main Camps - Irumu | Security | Very High | 1.20 | 29.80 | 120 |
| ADF Activity - Beni Territory | Security | Very High | 0.80 | 29.50 | 100 |
| ADF Stronghold - Mambasa | Security | Very High | 1.00 | 29.30 | 90 |
| ADF Camps - Komanda Area | Security | Very High | 1.50 | 30.20 | 80 |
| M23-ADF Overlap - Lubero | Security | Very High | -1.20 | 28.60 | 70 |
| Joint M23-ADF - Southern Beni | Security | Very High | -0.80 | 29.00 | 90 |
| Lamu Corridor - ASWJ Militant | Security | High | -2.00 | 40.90 | 100 |
| Thika Road - Construction | Infrastructure | Medium | -1.20 | 37.00 | 80 |
| Naivasha - Seasonal Flooding | Climate | Medium | -0.80 | 36.30 | 50 |
| Central Tanzania - Drought | Climate | Medium | -6.50 | 36.00 | 70 |
| Tanga Corridor - Maintenance | Infrastructure | Medium | -5.00 | 39.00 | 60 |
| Rwanda-DRC Border - Bunagana | Security | High | -1.2833 | 29.6167 | 40 |
| Rusizi-DRC Border Town | Security | High | -2.4833 | 28.9000 | 40 |

---

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.8+
- NumPy 1.20+
- Matplotlib 3.3+
- OSRM (Open Source Routing Machine, for real-world distance calculations)

### Installation
```bash
git clone https://github.com/YvesNDIKURIYO-2022/hybrid-memetic-framework-threat-aware-ctrp.git
cd hybrid-memetic-framework-threat-aware-ctrp
pip install -r requirements.txt
