## Project Directory Structure

This repository is organized to ensure clarity, reproducibility, and scalability for the IEEE EMBS BHI 2025 Data Competition.

📂 **IEEE_EMBS_BHI_25_CSOSEN/**  
│  
├── 📄 `README.md` – Project overview, dataset info, and instructions  
├── 📄 `requirements.txt` – Python dependencies  
├── 📄 `environment.yml` – (Optional) Conda environment  
│  
├── 📂 **data/**  
│   ├── 📂 `raw/` – Original dataset files from the competition  
│   ├── 📂 `processed/` – Cleaned or preprocessed datasets  
│   └── 📂 `interim/` – Temporary intermediate data files (optional)  
│  
├── 📂 **notebooks/**  
│   ├── 📓 `01_EDA.ipynb` – Stage 1 Exploratory Data Analysis  
│   ├── 📓 `02_FeatureEngineering.ipynb` – Feature creation and transformation  
│   └── 📓 `03_BaselineModel.ipynb` – Baseline model experiments for BDI-II prediction  
│  
├── 📂 **src/** – Reusable Python modules  
│   ├── 📄 `__init__.py`  
│   ├── 📄 `data_loader.py` – Functions to load and inspect datasets  
│   ├── 📄 `preprocessing.py` – Data cleaning and preprocessing functions  
│   ├── 📄 `eda.py` – Functions for exploratory data analysis  
│   ├── 📄 `features.py` – Feature engineering functions  
│   └── 📄 `visualization.py` – Plotting helper functions  
│  
├── 📂 **reports/**  
│   ├── 📂 `figures/` – Save all plots and visualizations  
│   ├── 📄 `midterm_report.pdf` – Draft mid-term report  
│   └── 📄 `final_report.pdf` – Final competition report  
│  
└── 📂 **scripts/**  
    ├── 📄 `run_eda.py` – Script to execute all EDA tasks end-to-end  
    └── 📄 `run_baseline_model.py` – Script to train and evaluate baseline models

### Notes:

- **Separation of concerns:** Raw data, processed data, notebooks, scripts, and reusable code are organized separately.
- **Reproducibility:** Scripts in `Scripts/` can reproduce all EDA, preprocessing, and baseline modeling results.
- **Professionalism:** All plots and figures are stored in `Reports/figures/` for easy reference in reports.
- **Scalability:** The structure can accommodate additional stages such as advanced modeling, results tracking, and experiments.
