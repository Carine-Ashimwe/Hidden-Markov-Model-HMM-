# Human Activity Recognition using Hidden Markov Models

A machine learning project that uses Hidden Markov Models to recognize and classify human activities based on accelerometer and gyroscope sensor data.

## Overview

This project implements a Hidden Markov Model-based system for recognizing human activities (standing, walking, jumping, and remaining still) from inertial measurement unit (IMU) sensor data. The system processes raw accelerometer and gyroscope readings from multiple subjects and uses HMMs to learn and predict activity patterns.

## Project Structure

```
├── HMM_Human_Recognition.ipynb    # Main analysis and model training notebook
├── README.md                        # This file
├── data/                            # Raw and processed sensor data
│   ├── merged_features_dataset.csv # Combined feature dataset
│   ├── Armstrong/                  # Data from Armstrong subject
│   │   ├── Jumping/               # Jumping activity samples (8 instances)
│   │   ├── Standing/              # Standing activity samples (8 instances)
│   │   ├── Still/                 # Stationary activity samples (8 instances)
│   │   └── Walking/               # Walking activity samples (7 instances)
│   ├── Carine/                     # Data from Carine subject
│   │   ├── Jumping/               # Jumping activity samples (8 instances)
│   │   ├── Standing/              # Standing activity samples (8 instances)
│   │   ├── Still/                 # Stationary activity samples (8 instances)
│   │   └── Walking/               # Walking activity samples (8 instances)
│   ├── merged/                     # Merged activity data
│   │   ├── jumping_merged.csv
│   │   ├── standing_merged.csv
│   │   ├── still_merged.csv
│   │   └── walking_merged.csv
│   └── features/                   # Extracted features
│       └── merged_features_dataset.csv
├── Scripts/                         # Utility and processing scripts
│   └── feature_extraction.py       # Feature extraction pipeline
└── Results & Plots/                # Generated analysis results and visualizations
```

## Data Description

### Raw Sensor Data

Each activity sample contains:

- **Accelerometer.csv** - 3-axis acceleration data (X, Y, Z)
- **Gyroscope.csv** - 3-axis angular velocity data (X, Y, Z)

### Activities Recognized

1. **Jumping** - Rapid vertical movement
2. **Standing** - Static upright posture
3. **Still** - Minimal or no movement
4. **Walking** - Continuous forward movement

### Subjects

- **Armstrong** - Primary subject with multiple activity samples
- **Carine** - Secondary subject with multiple activity samples

## Workflow

1. **Data Collection** - Raw sensor readings from accelerometer and gyroscope
2. **Feature Extraction** - Statistical features computed from raw sensor data (`feature_extraction.py`)
3. **Data Merging** - Activity samples combined into unified datasets
4. **Model Training** - HMM models trained on extracted features (in notebook)
5. **Classification** - Activity recognition and prediction
6. **Visualization** - Results and performance metrics

## Usage

Open and run the Jupyter notebook to execute the full analysis pipeline:

```bash
jupyter notebook HMM_Human_Recognition.ipynb
```

The notebook includes:

- Data loading and exploration
- Feature engineering
- HMM model training for each activity
- Activity classification and prediction
- Performance evaluation and visualization

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook
- Matplotlib (for visualization)
- hmmlearn (Hidden Markov Model library)

## Contributing

Contributions and improvements are welcome. Please feel free to submit issues or pull requests.

## License

MIT License
