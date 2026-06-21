# Space Weather Dashboard

An end-to-end Data Science, Machine Learning, and Space Systems project that collects, analyzes, visualizes, and forecasts space weather conditions using real-world NOAA datasets.

This project is being developed as a portfolio project to learn professional software development, data engineering, data analysis, machine learning, scientific computing, and dashboard deployment while working with real scientific datasets from the field of Space Weather.

---

## Project Objectives

The primary objectives of this project are:

* Collect real-time space weather data from NOAA APIs
* Build automated data pipelines
* Analyze solar and geomagnetic activity
* Create interactive visualizations
* Develop forecasting models
* Build a production-style dashboard
* Apply machine learning to scientific datasets
* Practice professional Git and GitHub workflows

---

## Space Weather Variables

This project will work with:

### Solar Activity

* Solar Flares
* Coronal Mass Ejections (CMEs)
* Sunspot Activity
* F10.7 Solar Flux

### Solar Wind

* Solar Wind Speed
* Proton Density
* Plasma Temperature
* Interplanetary Magnetic Field (IMF)

### Geomagnetic Activity

* Kp Index
* Dst Index
* Auroral Activity
* Geomagnetic Storm Events

---

## Technology Stack

### Programming

* Python 3.11

### Data Science

* Pandas
* NumPy

### Data Acquisition

* Requests
* NOAA APIs

### Visualization

* Plotly

### Dashboard

* Streamlit

### Machine Learning

* Scikit-Learn
* XGBoost (Planned)

### Development Tools

* Git
* GitHub
* VS Code
* Jupyter Notebook

---

## Project Structure

```text
Space_Weather_Dashboard/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── Day2_practice.ipynb
│   └── NOAA_Solar_Wind_EDA.ipynb
│
├── src/
│   ├── get_noaa_data.py
│   ├── data_collection/
│   ├── data_processing/
│   ├── visualization/
│   └── forecasting/
│
├── dashboard/
│
├── models/
│
├── docs/
│   └── Day3_Notes.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Development Roadmap

### Phase 1 — Foundation

* [x] Project Structure
* [x] Python Environment
* [x] Virtual Environment Setup
* [x] Package Management
* [x] Git & GitHub Setup
* [x] Initial Repository
* [x] Development Workflow

### Phase 2 — Data Collection

* [x] NOAA API Integration
* [x] Solar Wind Dataset Download
* [x] JSON Data Parsing
* [x] DataFrame Creation
* [x] Data Type Conversion
* [x] Multi Dataset Integration
* [ ] Automated Data Retrieval
* [ ] Historical Data Pipeline
* [ ] Raw Data Storage Pipeline

### Phase 3 — Data Analysis

* [x] Exploratory Data Analysis (EDA)
* [x] Data Type Conversion
* [x] Descriptive Statistics
* [x] Correlation Analysis
* [x] Solar Wind Event Investigation
* [x] IMF Statistical Analysis
* [x] IMF Event Investigation
* [x] Cross Dataset Correlation Analysis
* [x] Geomagnetic Activity Analysis
* [x] Kp Index Statistical Analysis
* [x] Kp Event Investigation
* [x] Sun-Earth Coupling Analysis

### Phase 4 — Visualization

* [x] Solar Wind Speed Visualization
* [x] Solar Wind Density Visualization
* [x] Solar Wind Temperature Visualization
* [x] IMF Bz Visualization
* [x] IMF Bt Visualization
* [x] Speed Vs Temperature Scatter Plot
* [x] Kp Index Visualization
* [x] Sun-Earth Correlation Heatmap
* [x] Bz vs Kp Scatter Plot
* [ ] Multi-Variable Dashboard
* [ ] Real-Time Monitoring
* [ ] Interactive Controls
* [ ] Live Data Refresh

### Phase 5 — Machine Learning

* [ ] Feature Engineering
* [ ] Kp Index Forecasting
* [ ] Dst Index Forecasting
* [ ] Model Training
* [ ] Model Evaluation
* [ ] Prediction Pipeline

### Phase 6 — Deployment

* [ ] Streamlit Application
* [ ] Dashboard Deployment
* [ ] Documentation
* [ ] Public Release

---

## Current Progress

### Phase 1 Completed

* Environment Setup
* GitHub Repository
* Development Workflow
* Project Foundation

### Phase 2 In Progress

* NOAA API Connected
* Solar Wind Dataset Downloaded
* JSON Parsing Implemented
* DataFrame Construction Completed
* IMF Dataset Downloaded
* IMF Data Cleaning Completed
* IMF Analysis Completed 
* Solar Wind + IMF integration Completed
* Data Cleaning Completed
* Kp Dataset Downloaded
* Kp Data Cleaning Completed

### Phase 3 In Progress

* Descriptive Statistics Completed
* Correlation Analysis Completed
* Solar Wind Event Investigation Completed
* IMF Statistical Analysis Completed
* IMF Event Investigation Completed
* Cross-Dataset Correlation Analysis Completed
* Space Weather Relationship Analysis Completed
* Kp Statistical Analysis Completed
* Kp Event Investigation Completed
* Sun-Earth Coupling Analysis Completed
* Correlation Heatmap Analysis Completed

---

## First NOAA Dataset Analysis

### Dataset 

NOAA Solar Wind Plasma Dataset

The dataset used for this analysis starts from 13 june 2026 to 20 june 2026.

Variables:

* Time Tag
* Density
* Speed
* Temperature

Dataset Size:

* ~9,200 observations
* Approximately 7 days of solar wind measurements


## Second NOAA Dataset Analysis

### Dataset

NOAA Interplanetary Magnetic Field (IMF) Dataset

The dataset used for this analysis starts from 14 june 2026 to 21 june 2026.

Variables:

* Time Tag
* Bx
* By
* Bz
* Bt

Dataset Size:

* ~9,600 observations
* Approximately 7 days of IMF measurements

## Second NOAA Dataset Analysis

## Third NOAA Dataset Analysis

### Dataset

NOAA Planetary Kp Index Dataset

The dataset used for this analysis starts from 14 June 2026 to 21 June 2026.

Variables:

* Time Tag
* Kp
* a_running
* station_count

Dataset Size:

* 59 observations
* Approximately 7 days of geomagnetic measurements


---

## Statistical Summary

### Solar Wind Speed

* Average Speed: ~435 km/s
* Maximum Speed: ~607 km/s
* Minimum Speed: ~357 km/s

### Solar Wind Density

* Average Density: ~6.53 particles/cm³
* Maximum Density: ~17.49 particles/cm³
* Minimum Density: ~0.09 particles/cm³

### Solar Wind Temperature

* Average Temperature: ~112,257 K
* Maximum Temperature: ~552,298 K
* Minimum Temperature: ~2,000 K

---

## Correlation Analysis

| Variables             | Correlation |
| --------------------- | ----------: |
| Density ↔ Speed       |      -0.189 |
| Density ↔ Temperature |       0.345 |
| Speed ↔ Temperature   |       0.522 |

### Initial Findings

* Solar Wind Speed and Temperature show a moderate positive relationship.
* Density and Speed show a weak inverse relationship.
* Faster solar wind streams generally tend to be hotter.
* Density appears to be more variable and less predictive than speed.

---

## Event Investigation

### Fastest Solar Wind Event

* Time: 2026-06-13 15:59
* Speed: 606.9 km/s
* Density: 0.27 particles/cm³
* Temperature: 8,092 K

### Densest Solar Wind Event

* Time: 2026-06-17 01:55
* Density: 17.49 particles/cm³
* Speed: 403.2 km/s
* Temperature: 73,222 K

### Hottest Solar Wind Event

* Time: 2026-06-14 03:33
* Temperature: 552,298 K
* Density: 12.85 particles/cm³
* Speed: 530.7 km/s

---

## IMF Statistical Summary

### Bz

* Average: ~0.56 nT
* Maximum: 11.40 nT
* Minimum: -7.51 nT

### Bt

* Average: ~6.00 nT
* Maximum: 12.00 nT
* Minimum: 0.62 nT

## IMF Event Investigation

### Strongest Southward IMF Event

* Time: 2026-06-16 13:47
* Bz: -7.51 nT
* Bt: 7.75 nT

### Strongest Magnetic Field Event

* Time: 2026-06-17 05:32
* Bt: 12.00 nT
* Bz: 7.58 nT

## IMF Direction Analysis

### Positive Bz

* 5,408 observations

### Negative Bz

* 4,194 observations

### Findings

* Approximately 56% of IMF observations were northward.
* Approximately 44% of IMF observations were southward.
* Bz remained one of the most important forecasting variables identified during analysis.

## Solar Wind + IMF Integrated Analysis

* The Solar Wind Plasma and IMF datasets were merged using NOAA timestamps. (for this Again the solar wind data is from dates 14 June and 21 June)

### Correlation Results
| Variables             | Correlation | Interpretation |
| --------------------- | ----------: | -------------- |
| Speed ↔ Temperature   |       0.810 | Strong positive relationship |
| Speed ↔ Bz            |       0.046 | No meaningful relationship |
| Density ↔ Bz          |       0.251 | Weak positive relationship |
| Temperature ↔ Bz      |      -0.181 | Weak negative relationship |
| Bz ↔ Bt               |       0.247 | Weak positive relationship |

### Major Observations

* Solar wind speed strongly correlates with plasma temperature.
* Solar wind properties show little relationship with Bz orientation.
* Bz behaves largely independently from plasma conditions.
* Plasma properties and magnetic field orientation should be analyzed separately when assessing space weather conditions.

## Advanced Space Weather Concepts Explored

* Interplanetary Magnetic Field (IMF)
* Frozen-In Magnetic Fields
* Bx, By, Bz Components
* Total Magnetic Field Strength (Bt)
* Southward Bz and Magnetic Reconnection
* Geoeffective Space Weather Conditions
* Solar Wind and IMF Coupling
* Forecasting Variables Used in Geomagnetic Storm Prediction
* IMF Persistence and Duration Analysis


## Kp Statistical Summary

### Kp Index

* Average: 1.63
* Maximum: 3.00
* Minimum: 0.33

### Interpretation

* Average conditions remained geomagnetically quiet.
* No geomagnetic storm conditions were observed.
* The highest activity level reached Kp = 3 (Unsettled Conditions).


## Kp Event Investigation

### Highest Kp Event

* Time: 2026-06-19 00:00
* Kp: 3.00
* a_running: 15

### Lowest Kp Event

* Time: 2026-06-17 18:00
* Kp: 0.33
* a_running: 2

## Solar Wind + IMF + Kp Integrated Analysis

The Solar Wind, IMF, and Kp datasets were merged using NOAA timestamps and common 3-hour aggregation windows.

This analysis represents the first direct investigation of Sun-Earth coupling within the project.

### Correlation Results

| Variables | Correlation |
| ---------- | ----------: |
| Kp ↔ Density | 0.161 |
| Kp ↔ Speed | -0.107 |
| Kp ↔ Temperature | -0.017 |
| Kp ↔ Bz | -0.188 |
| Kp ↔ Bt | 0.190 |

### Findings

* Kp showed weak relationships with all investigated solar wind and IMF variables during the study period.
* The negative correlation between Kp and Bz is consistent with space weather theory.
* Southward IMF conditions were associated with increased geomagnetic activity.
* The study period remained geomagnetically quiet, limiting the strength of observed relationships.
* Results suggest that geomagnetic activity depends on multiple interacting variables rather than a single parameter.

## Geomagnetic Activity Concepts Explored

* Planetary K Index (Kp)
* Geomagnetic Activity Monitoring
* Earth's Response to Solar Activity
* Quiet, Unsettled, and Storm Conditions
* Kp Scale Interpretation
* Geomagnetic Disturbance Classification
* Sun-Earth Coupling Analysis
* Geomagnetic Storm Forecasting Indicators
* Kp Event Investigation
* Multi-Dataset Space Weather Analysis
* Solar Wind, IMF, and Kp Integration
* Correlation Heatmap Analysis
* Bz-Kp Relationship Investigation
* Geomagnetic Activity Persistence
* Time-Lag Effects in Space Weather


## Lessons Learned

Through this project the following concepts have been implemented and explored:

* API Fundamentals
* HTTP Requests
* JSON Data Structures
* NOAA Space Weather Data Sources
* Pandas DataFrames
* Data Cleaning
* Data Type Conversion
* Exploratory Data Analysis
* Statistical Analysis
* Correlation Analysis
* Time-Series Visualization
* Scientific Data Investigation
* Interplanetary Magnetic Field (IMF)
* Frozen-In Magnetic Fields
* Bx, By, Bz Components
* Total Magnetic Field Strength (Bt)
* Southward Bz and Magnetic Reconnection
* Solar Wind and IMF Relationship Analysis
* Geoeffective Space Weather Conditions
* Multi-Dataset Integration
* Cross-Dataset Correlation Analysis
* Scientific Hypothesis Testing
* Space Weather Forecasting Fundamentals
* Kp Index Interpretation
* Geomagnetic Activity Analysis
* Geomagnetic Activity Classification
* Kp Index Event Investigation
* Duration-Based Event Analysis
* IMF Persistence Analysis
* Bz-Kp Relationship Investigation
* Sun-Earth Coupling Analysis
* Sun-Earth Data Aggregation Techniques
* Time-Series Resampling and Alignment
* Correlation Heatmap Visualization
* Scatter Plot Analysis
* Space Weather Event Interpretation
* Geomagnetic Storm Forecasting Concepts

---

## Next Milestone

### Dst Index Analysis

Upcoming objectives:

* Download NOAA Dst dataset
* Study ring current dynamics
* Compare Kp and Dst responses
* Investigate geomagnetic storm intensity
* Analyze magnetospheric energy storage
* Continue Sun-Earth coupling research

Dst analysis will provide a second measure of Earth's response and help distinguish between geomagnetic activity levels and geomagnetic storm intensity.

---

## Long-Term Vision

The final system will:

* Collect real-time NOAA space weather data
* Process and clean incoming datasets
* Visualize current solar and geomagnetic conditions
* Monitor solar wind and magnetic field behavior
* Forecast geomagnetic storm activity
* Predict Kp and Dst indices using machine learning
* Provide an interactive dashboard for monitoring space weather

---

## Author

**Manas Anumala**

Bachelor of Mechanical Engineering

Postgraduate Certificate in Space Exploration Systems

Interests:

* Space Systems
* Space Weather
* Artificial Intelligence
* Machine Learning
* Data Science
* Scientific Computing
* Software Development

Building projects at the intersection of Space Systems, Data Science, Artificial Intelligence, and Scientific Computing.
