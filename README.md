# 🌍 EcoStream | Air Quality Intelligence Dashboard

EcoStream is a high-performance environmental intelligence dashboard designed to analyze global air quality data with a focus on visual integrity and analytical depth. It implements advanced dimensionality reduction (PCA) and high-density temporal analysis to reveal pollution patterns across residential and industrial zones.

## 🚀 Features

- **Advanced Dim. Reduction (PCA)**: Project 6-dimensional pollutant profiles into 2D space to distinguish between industrial and residential ecological signatures.
- **High-Density Temporal Analysis**: Station × Month heatmaps to identify city-wide pollution events and diurnal/seasonal periodicities.
- **Distribution Modeling (Tail Integrity)**: Log-scaled survival functions to quantify extreme risk beyond standard histograms.
- **Tufte-Compliant Visualizations**: Designed with a high data-ink ratio, eliminating chartjunk and using perceptually uniform color scales (Plasma, Inter typeface).
- **Pro-Level Tech Stack**: Powered by Streamlit, DuckDB (Out-of-core processing), Plotly, and Scikit-Learn.

## 📂 Project Structure

```text
.
├── src/
│   ├── app.py                # Main Dashboard Entry Point
│   ├── core/
│   │   └── data_fetcher.py   # OpenAQ API Data Pipeline
│   ├── analysis/             # Domain Logic & Visualization
│   │   ├── pca_analysis.py
│   │   ├── task3_distribution.py
│   │   ├── task4_visual_integrity.py
│   │   └── temporal_analysis.py
│   └── utils/                # Maintenance & Discovery Scripts
├── data/                     # Raw CSV Store
├── tests/                    # Unit Tests
├── assets/                   # Media & Original Plots
├── requirements.txt          # Dependency Manifest
└── README.md                 # Documentation
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Data_Science_Ass_02_final
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Setup**:
   Ensure `locations_metadata.json` and `data/` folder are present in the root directory.

## 📈 Usage

Launch the interactive dashboard:
```bash
streamlit run src/app.py
```


## 🎨 Visualization Principles (Tufte-Compliance)

This project strictly adheres to Edward Tufte's principles of visual integrity:
1. **Minimize the Lie Factor**: Avoid 3D perspectives and volume-based scaling that distort magnitudes.
2. **Maximize Data-Ink Ratio**: Removed borders, redundant grid lines, and decorative elements (chartjunk).
3. **Escaping Flatland**: Using color (Plasma) and size (Bivariate Bubble Maps) to encode multivariate relationships without clutter.
4. **Perceptual Uniformity**: Sequential luminance-based colormaps to ensure magnitude honesty.

---
*Created for Advanced Data Science Analytics | 2026*
