# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the RMS Titanic using various passenger features and advanced data analysis techniques.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Model Performance](#model-performance)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🚢 Overview

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. This project uses machine learning algorithms to predict which passengers were more likely to survive based on features like passenger class, sex, age, and other attributes.

**Objective**: Build a predictive model that answers the question: "What sorts of people were more likely to survive the Titanic disaster?"

## 📊 Dataset

The dataset contains information about Titanic passengers with the following key attributes:

| Feature | Description |
|---------|-------------|
| PassengerId | Unique identifier for each passenger |
| Survived | Survival status (0 = No, 1 = Yes) |
| Pclass | Ticket class (1st, 2nd, 3rd) |
| Name | Passenger name |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

**Data Source**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

## ✨ Features

- **Comprehensive Data Analysis**: Exploratory data analysis with visualizations
- **Data Preprocessing**: Handling missing values, feature engineering, and data cleaning
- **Multiple ML Models**: Implementation of various algorithms including:
  - Logistic Regression
  - Random Forest
  - Neural Networks
- **Feature Engineering**: Creating new meaningful features from existing data
- **Model Evaluation**: Cross-validation and performance metrics
- **Visualization**: Interactive plots and charts for better insights

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/chetan4077/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv titanic_env
   source titanic_env/bin/activate  # On Windows: titanic_env\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter plotly
   ```

## 🚀 Usage

### Running the Analysis

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   Navigate to `Titanic_Survival_Prediction.ipynb` and run all cells

3. **Alternative: Run Python script**
   ```bash
   python titanic_prediction.py
   ```

### Quick Start Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
train_data = pd.read_csv('data/train.csv')

# Basic preprocessing
# (See full notebook for complete preprocessing pipeline)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 📈 Data Analysis

### Key Insights Discovered:

- **Gender Impact**: Women had a significantly higher survival rate (74%) compared to men (19%)
- **Class Matters**: First-class passengers had better survival rates than lower classes
- **Age Factor**: Children and younger passengers had higher survival chances
- **Family Size**: Passengers with small families (1-3 members) had better survival rates
- **Fare Correlation**: Higher fare generally correlated with better survival odds

### Visualizations Include:
- Survival rate by gender and class
- Age distribution analysis
- Correlation heatmaps
- Feature importance plots
- Missing data patterns

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Random Forest | 80.0% | 0.77 | 0.72 | 0.74 |
| Logistic Regression | 78.0% | 0.75 | 0.70 | 0.72 |

**Best Model**: Random Forest Classifier with 80.0% accuracy

### Cross-Validation Results:
- Mean CV Score: 79.0%
- Standard Deviation: 2.5%

## 📊 Results

The final model successfully identifies key survival factors:

1. **Most Important Features**:
   - Gender (Sex)
   - Passenger Class (Pclass)
   - Fare
   - Age
   - Family Size

2. **Survival Probability Insights**:
   - First-class female passengers: ~95% survival rate
   - Third-class male passengers: ~15% survival rate
   - Children under 10: ~60% survival rate

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Jupyter Notebook**: For interactive analysis
- **Git**: Version control

## 📁 Project Structure

```
Titanic-Survival-Prediction/
│
├── data/
│   ├── Cleaned Data/            # Processed datasets
│   ├── gender_submission.csv    # Sample submission format
│   ├── test.csv                # Test dataset
│   └── train.csv               # Training dataset
│
├── Submissions/
│   ├── logistic_submission.csv           # Logistic regression predictions
│   ├── logistic_updated_submission.csv   # Updated logistic regression predictions
│   ├── random_forest_submission.csv      # Random forest predictions
│   ├── random_forest_updated_2_submission.csv  # Random forest v2 predictions
│   └── random_forest_updated_submission.csv    # Updated random forest predictions
│
├── Data_Cleaning.ipynb         # Data preprocessing and cleaning notebook
├── model.ipynb                 # Model training and evaluation notebook
└── README.md                   # Project documentation
```


##  Acknowledgments

- [Kaggle](https://www.kaggle.com/c/titanic) for providing the dataset
- The open-source community for the amazing tools and libraries
- Historical records and research about the RMS Titanic

## 📞 Contact

**Chetan** - [GitHub Profile](https://github.com/chetan4077)

Project Link: [https://github.com/chetan4077/Titanic-Survival-Prediction](https://github.com/chetan4077/Titanic-Survival-Prediction)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
