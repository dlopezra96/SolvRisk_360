# 🔥 SolvRisk 360 — Fire Risk Concentration Analyzer (v1.0)

[![Python ≥3.13](https://img.shields.io/badge/Python-%3E%3D3.13-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-latest-blue.svg)](https://dash.plotly.com/)
[![R ≥4.0](https://img.shields.io/badge/R-%3E%3D4.0-brightgreen.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

An interactive, multi-page dashboard for pinpointing high-risk clusters of insured assets under Solvency II, complete with benchmarking and export capabilities.

---

## 📋 Table of Contents

1. [🚀 Features](#-features)
2. [🎯 Use Cases](#-use-cases)
3. [⚙️ Prerequisites](#️-prerequisites)
4. [🔧 R Integration (.env)](#-r-integration-env)
5. [🛠️ Installation](#️-installation)
6. [🏃 Usage](#-usage)
7. [📂 Project Structure](#-project-structure)
8. [🧠 Algorithms](#-algorithms)
9. [📝 License & Author](#-license--author)

---

## 🚀 Features

* **Interactive Map**: Dark-themed CartoDB basemap with policy markers and popups.
* **Dynamic KPIs**: Global & zone-level metrics cards (totals, quartiles, mean, etc.).
* **Spatial Algorithms**:

  * Exhaustive brute-force
  * Multi-start metaheuristic
  * R grid-search via `spatialrisk::highest_concentration`
  * Optimized Python variants (`fast_exhaustive`, `fast_with_grid`, `fast_with_multilocal`)
* **Benchmark Scripts**: Compare runtime and result quality (see `/tests/`).
* **Export**: Download CSV now; PDF reporting coming soon.
* **Modular Dash App**: Multi-page structure (`Home`, `Select Algorithm`, `Results`).

---

## 🎯 Use Cases

* Identify geographic clusters with maximum insured capital.
* Support SCR calculations under Solvency II.
* Evaluate and select the fastest or most accurate algorithm.
* Deliver actionable geospatial insights to underwriting teams.

---

## ⚙️ Prerequisites

* **Python** ≥ 3.13
* **pip**
* **R** ≥ 4.0 (for `spatialrisk` grid-search)
* **git** (optional, for cloning)

---

## 🔧 R Integration (.env)

The R-based grid search requires `R_HOME` to point at your R installation. Create a `.env` file at the project root:

```ini
# .env
R_HOME="C:/Program Files/R/R-4.4.2"
```

Adjust the path for your OS and R version. This is loaded automatically by `python-dotenv`. 

Install the required packages in R Gui:
```ini
install.packages("readr")
install.packages("spatialrisk")
```

---

## 🛠️ Installation

```bash
# 1. Clone repository
git clone https://github.com/dlopezra96/SolvRisk_360.git
cd SolvRisk_360

# 2. Create & activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Usage

```bash
# Launch the Dash app
python app.py
```

Open your browser at [http://127.0.0.1:8050](http://127.0.0.1:8050).

---

## 📂 Project Structure

```text
SolvRisk_360/
├── .env                    # R_HOME configuration
├── .gitignore              # Files and directories ignored by Git
├── .pylintrc               # Pylint configuration
├── requirements.txt        # Python dependencies
├── setup.cfg               # Centralized config for linters, formatters, etc.
├── app.py                  # Entry point (multi-page Dash app)
├── data/                   # Sample or uploaded CSV datasets
├── modules/                # Core logic (distance, index, algorithms)
├── pages/                  # Dash pages (Home, Select Algorithm, Results)
├── assets/                 # Static assets (CSS, images)
├── tests/                  # Benchmark scripts
├── reports/                # Output CSVs & PDF reports
└── README.md               # Project overview & instructions
```

---

## 🧠 Algorithms

* **Exhaustive**: Brute-force cluster search (Badal-Valero et al.).
* **Metaheuristic**: Multi-start continuous pattern search (Gomes et al.).
* **R Grid Search**: C++ accelerated via `spatialrisk` (Haringa M.).
* **Fast Python (own algorithms)**:

  * `fast_exhaustive` (BallTree + sparse matrix)
  * `fast_with_grid` (anchor + local grid refinement)
  * `fast_with_multilocal` (anchor + micro multi-start)

---

## 📝 License & Author

**License**: CC BY NC 4.0  
**Author**: David López Raluy  
Master’s in Data Science, Universitat Oberta de Catalunya (UOC)
