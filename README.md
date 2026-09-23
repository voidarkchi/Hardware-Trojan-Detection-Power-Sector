# Hardware Trojan Detection for Power-Sector Hardware

## Overview

Hardware Trojans are malicious modifications embedded into integrated circuits that may alter functionality, leak sensitive information, or affect circuit behaviour.

This project develops a machine-learning based approach for detecting hardware Trojans from circuit-level structural and power characteristics. The project focuses on a software-based detection pipeline using an existing benchmark dataset rather than requiring physical hardware.

The system uses **Random Forest classification** together with **circuit-family-based clean baselines** to identify deviations associated with Trojan-infected circuits.

---

## Problem Statement

Hardware Trojans can remain difficult to detect because their effects may be small compared with the overall characteristics of a legitimate circuit.

Traditional hardware-level inspection and physical verification can be expensive and difficult to scale. This project investigates whether measurable circuit characteristics such as structural properties and power-related features can be used for automated Trojan detection.

---

## Dataset

The project uses the `HEROdata2.xlsx` dataset containing circuit-level hardware characteristics.

Each record contains:

- 49 numerical hardware features
- `Label` — Trojan Free / Trojan Infected
- `Circuit` — circuit identifier

The numerical features include structural and power characteristics such as:

- Number of references
- Number of cells
- Number of nets
- Number of ports
- Number of sequential cells
- Switching power
- Internal power
- Total power

The dataset contains multiple Trojan variants belonging to different circuit families.

### Dataset preprocessing

The dataset is processed using the following steps:

1. Clean label and circuit-name values.
2. Remove duplicate records.
3. Identify the circuit family from the circuit identifier.
4. Identify circuit families containing Trojan-Free reference circuits.
5. Exclude families for which no clean baseline is available.
6. Handle missing numerical values using median imputation.

After preprocessing:

- 907 original records
- 903 unique records
- 880 records with usable clean-family baselines
- 862 Trojan Infected
- 18 Trojan Free

---

## Proposed Methodology

The detection pipeline consists of:

```text
HEROdata2 Dataset
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Circuit Family Identification
        ↓
Trojan-Free Family Baseline
        ↓
Baseline-Based Feature Normalization
        ↓
Random Forest Classifier
        ↓
Trojan Probability
        ↓
0.70 Decision Threshold
        ↓
Trojan Free / Trojan Infected
