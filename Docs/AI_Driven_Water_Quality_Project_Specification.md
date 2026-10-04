# AI-Driven Water Quality Prediction & Explainable Decision Support System

## 1. Project Overview

### Project Title

**AI-Driven Water Quality Prediction & Explainable Decision Support
System**

### Project Type

Data Science / Machine Learning / Explainable AI / Decision Support

### Core Idea

The project develops an end-to-end Data Science application that
transforms water-quality measurements into:

1.  **Water-quality predictions**
2.  **Comparative evaluation of multiple machine-learning models**
3.  **Explainable predictions using SHAP**
4.  **Identification of the parameters that contributed most to a
    prediction**
5.  **Interpretation of abnormal parameters**
6.  **Preventive/corrective decision recommendations**
7.  **An interactive Streamlit dashboard**

The project is intended to be **dataset-driven**. It does not require
physical IoT sensors or live sensor hardware for the core
implementation.

The central flow is:

``` text
Water Quality Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Water Quality Classification
        ↓
Multiple ML Models
        ↓
Model Evaluation & Comparison
        ↓
Best Model
        ↓
SHAP Explainability
        ↓
Key Contributing Parameters
        ↓
Parameter Interpretation
        ↓
Decision-Support Rules
        ↓
Preventive / Corrective Recommendations
        ↓
Interactive Streamlit Dashboard
```

------------------------------------------------------------------------

# 2. Problem Statement

Traditional water-quality assessment commonly depends on periodic
sampling and laboratory analysis. This can make continuous assessment,
pattern identification, and early recognition of deterioration
difficult.

Water-quality datasets contain multiple physical, chemical, and
potentially biological parameters. These parameters interact with one
another, and the final water-quality condition may not be obvious from
looking at individual measurements.

The project addresses the following problem:

> **How can machine-learning models be used to classify water quality
> from relevant water-quality parameters, how can different models be
> comparatively evaluated, and how can the resulting predictions be
> explained and translated into actionable decision support?**

The system therefore needs to answer four questions:

### Question 1 --- What is the water quality?

Example:

``` text
Predicted Water Quality: Poor
```

### Question 2 --- Why did the model make this prediction?

Example:

``` text
Main contributing parameters:
- High BOD
- High turbidity
- Elevated TDS
- Low dissolved oxygen
```

### Question 3 --- What do these parameters indicate?

The system interprets abnormal parameter conditions using
domain-informed rules.

Example:

``` text
High BOD:
Associated with increased biodegradable organic load.

High turbidity:
Indicates increased suspended material and may be associated
with runoff or other contamination sources.

Low dissolved oxygen:
Indicates oxygen depletion and can accompany degraded
water conditions.
```

### Question 4 --- What could be done?

The decision-support layer maps detected parameter problems to
appropriate preventive or corrective actions.

Example:

``` text
Recommended actions:
- Investigate potential wastewater or runoff sources.
- Check treatment effectiveness where applicable.
- Increase monitoring of the affected parameters.
- Investigate persistent abnormal readings.
```

------------------------------------------------------------------------

# 3. What the Project Is and Is Not

## 3.1 What it is

The project is:

-   A machine-learning-based water-quality classification system
-   A comparative ML study
-   An explainable AI system
-   A rule-based decision-support prototype
-   An interactive Data Science dashboard
-   A dataset-driven research/course project

## 3.2 What it is not

The core project is **not**:

-   A physical water-quality sensor system
-   An Arduino/ESP32 hardware project
-   A live IoT deployment
-   A laboratory water-testing system
-   A system that physically identifies a pollution source
-   A system that proves that a particular factory/person caused
    pollution
-   A replacement for laboratory or regulatory water testing

The model can identify **statistical/model associations and influential
parameters**, while the decision-support layer can provide **possible
interpretations and recommended investigations/actions**.

It should not claim that ML has physically proven the source of
pollution.

------------------------------------------------------------------------

# 4. Project Objectives

## Primary Objective

Develop an end-to-end Data Science application that classifies water
quality from water-quality parameters and provides interpretable
explanations and decision support.

## Supporting Objectives

1.  Collect and select a suitable water-quality dataset.
2.  Clean and preprocess the data.
3.  Perform exploratory data analysis.
4.  Identify important relationships between water-quality parameters.
5.  Engineer useful features where justified.
6.  Develop multiple machine-learning classification models.
7.  Evaluate the models using appropriate classification metrics.
8.  Compare model performance under a consistent experimental setup.
9.  Select a suitable final model based on the evaluation results.
10. Apply SHAP to explain model predictions.
11. Identify parameters contributing to individual predictions.
12. Detect abnormal/problematic parameter conditions.
13. Map parameter conditions to domain-informed recommendations.
14. Build an interactive Streamlit dashboard.
15. Present predictions, explanations, trends, and recommendations in an
    understandable form.

------------------------------------------------------------------------

# 5. Research Questions

The project can be structured around these research questions:

### RQ1

How effectively can different machine-learning algorithms classify
water-quality conditions using the selected dataset?

### RQ2

How do the selected models compare across accuracy, precision, recall,
F1-score, and confusion-matrix performance?

### RQ3

Which water-quality parameters contribute most strongly to the
predictions of the selected model?

### RQ4

Can explainable model outputs be translated into understandable
parameter-level interpretations?

### RQ5

Can parameter-level interpretations be converted into useful
preventive/corrective decision-support recommendations?

------------------------------------------------------------------------

# 6. Project Scope

## Core Scope --- Must Implement

### Data

-   Dataset acquisition
-   Data validation
-   Missing-value handling
-   Duplicate handling
-   Outlier investigation
-   Data-type correction
-   Exploratory analysis

### Machine Learning

-   Classification target definition
-   Train/validation/test strategy
-   Multiple ML models
-   Hyperparameter tuning where appropriate
-   Model evaluation
-   Model comparison
-   Final model selection

### Explainability

-   SHAP global feature importance
-   SHAP local explanation for an individual prediction
-   Direction/contribution of important features where supported

### Decision Support

-   Identify problematic parameters
-   Interpret abnormal values
-   Map conditions to recommendations
-   Clearly distinguish model explanation from domain interpretation

### Application

-   Streamlit dashboard
-   User input
-   Prediction
-   Explanation
-   Parameter visualization
-   Recommendation output
-   Basic historical/trend visualization if the dataset contains
    suitable temporal information

------------------------------------------------------------------------

# 7. Optional Extensions

Only implement these if the core pipeline is stable.

### Possible extensions

-   LIME
-   Counterfactual explanations
-   Prediction confidence/uncertainty
-   Temporal forecasting
-   Anomaly detection
-   Dataset drift detection
-   GIS visualization
-   Additional datasets
-   Advanced ensemble methods
-   Deep learning models

These are extensions, not prerequisites for the core project.

------------------------------------------------------------------------

# 8. Features to Avoid Making Mandatory

The following should not become mandatory requirements unless the
project timeline and dataset justify them:

-   Live IoT sensors
-   Blockchain
-   Mobile application
-   CTGAN/TVAE synthetic data
-   Complex GIS infrastructure
-   LSTM/GRU/TFT forecasting
-   Real-time streaming infrastructure
-   Cloud deployment infrastructure
-   Sensor hardware

The project should remain a coherent Data Science system rather than
becoming an accumulation of unrelated technologies.

------------------------------------------------------------------------

# 9. Target ML Problem

## Recommended Primary Task: Classification

The recommended primary task is:

> **Multi-class water-quality classification**

The model receives water-quality parameters as input and predicts a
water-quality category.

Conceptually:

``` text
Input:
pH
Temperature
Turbidity
TDS
Electrical Conductivity
DO
BOD
COD
Nitrate
Other available parameters
        ↓
ML Model
        ↓
Water Quality Class
```

Possible classes depend entirely on the selected dataset and its
legitimate labeling scheme.

Examples could include:

``` text
Good
Moderate
Poor
Very Poor
```

However, **do not invent these labels** if the dataset does not contain
them.

The final class definitions must be based on the selected dataset's
existing labels or a clearly documented, defensible water-quality
standard/index methodology.

------------------------------------------------------------------------

# 10. Why Classification Is the Primary Task

Classification fits the project's intended decision-support output well.

The final system can communicate:

``` text
Predicted Class → Explanation → Interpretation → Recommendation
```

This is easier to present and demonstrate than making WQI regression the
sole objective.

For example:

``` text
Prediction:
POOR

SHAP:
BOD        +0.31
Turbidity  +0.22
COD        +0.17
DO         -0.14

Interpretation:
Several pollution-related parameters contributed strongly
to the predicted poor condition.

Decision Support:
Investigate potential organic pollution and wastewater/runoff
sources and review treatment effectiveness.
```

------------------------------------------------------------------------

# 11. WQI Regression --- Optional Secondary Task

Water Quality Index (WQI) regression can be considered as a secondary
experiment if the chosen dataset and methodology support it.

The model would predict a numerical WQI value:

``` text
Input parameters
       ↓
Regression model
       ↓
Predicted WQI
```

Possible metrics:

-   MAE
-   RMSE
-   R²

However, WQI should not be added simply because it sounds more advanced.

Before implementing WQI regression, verify:

1.  The dataset contains a valid WQI value or supports a defensible
    calculation.
2.  The WQI calculation methodology is clearly defined.
3.  The target is not accidentally derived from exactly the same input
    variables in a way that creates leakage.
4.  The project has enough time to evaluate regression separately.

------------------------------------------------------------------------

# 12. Dataset Selection

## Preferred Starting Direction

An India-focused water-quality dataset is desirable because it makes the
project contextually relevant.

Potential sources identified during the project planning include:

-   CPCB / India's National Water Data Portal
-   USGS Water Quality Portal
-   Kaggle water-quality datasets
-   UCI-related water-quality datasets
-   Other credible public datasets

The dataset document prepared for the project identifies CPCB
surface-water data as a strong starting point for an India-focused
project and lists parameters such as pH, turbidity, temperature,
electrical conductivity, TDS/total solids, dissolved oxygen, BOD, COD,
nitrate, and coliform-related measurements as potentially useful
variables.

## Dataset Selection Criteria

The final dataset should ideally have:

-   Sufficient number of records
-   Clearly documented parameters
-   Meaningful target labels
-   Minimal ambiguity in class definitions
-   Reasonable feature coverage
-   Enough samples per class
-   Manageable missingness
-   No obvious target leakage
-   A reproducible source
-   Documentation of measurement units
-   Ideally, location and/or time information

------------------------------------------------------------------------

# 13. Important Dataset Decision

Do not finalize the ML architecture before finalizing the dataset.

The correct order is:

``` text
Candidate datasets
       ↓
Inspect schema
       ↓
Inspect target/labels
       ↓
Check class distribution
       ↓
Check missing values
       ↓
Check parameter coverage
       ↓
Check leakage
       ↓
Select final dataset
       ↓
Freeze experimental dataset
       ↓
Design final ML pipeline
```

The dataset determines:

-   Whether classification is possible
-   Number of classes
-   Available input features
-   Appropriate preprocessing
-   Appropriate models
-   Evaluation strategy
-   SHAP feature set
-   Decision-support rules

------------------------------------------------------------------------

# 14. Water-Quality Features

Potential input parameters include:

-   pH
-   Temperature
-   Turbidity
-   Electrical Conductivity
-   TDS / Total Solids
-   Dissolved Oxygen (DO)
-   Biochemical Oxygen Demand (BOD)
-   Chemical Oxygen Demand (COD)
-   Nitrate
-   Phosphate, if available
-   Coliform measures, if available
-   Other documented physical/chemical parameters

Do not automatically include every parameter.

A feature should be included only if:

1.  It is available in the final dataset.
2.  It has acceptable data quality.
3.  It is not the target itself.
4.  It does not cause target leakage.
5.  Its measurement meaning is understood.

------------------------------------------------------------------------

# 15. Data Cleaning Pipeline

## Step 1 --- Schema Inspection

Check:

-   Column names
-   Data types
-   Number of records
-   Number of unique values
-   Units
-   Target column
-   Identifier columns
-   Date/time columns
-   Location columns

## Step 2 --- Duplicate Handling

Identify:

-   Exact duplicate rows
-   Duplicate measurements
-   Repeated records caused by data collection processes

Do not automatically delete repeated observations without understanding
whether they represent legitimate repeated measurements.

## Step 3 --- Missing Values

Analyze:

-   Missing percentage per column
-   Missing percentage per row
-   Missingness patterns
-   Whether missingness is concentrated in particular locations/time
    periods/classes

Possible strategies:

-   Remove features with excessive missingness
-   Remove rows when justified
-   Median/mean imputation for suitable numeric features
-   Model-based imputation if justified
-   Preserve missingness indicators where useful

The method must be documented.

## Step 4 --- Outliers

Use:

-   IQR
-   Box plots
-   Z-scores where appropriate
-   Domain ranges

Do not automatically delete every statistical outlier.

Water-quality measurements can contain genuine extreme events.

An outlier should be removed only when there is evidence that it is
invalid, erroneous, or unsuitable for the selected modeling objective.

## Step 5 --- Data Types and Units

Standardize:

-   Numeric types
-   Dates
-   Categories
-   Units

Examples:

``` text
mg/L
µS/cm
NTU
°C
```

Unit inconsistencies must be resolved before modeling.

------------------------------------------------------------------------

# 16. Exploratory Data Analysis

EDA should answer actual questions rather than only generate plots.

## Required EDA

### Distribution Analysis

For each important parameter:

-   Histogram
-   Box plot
-   Summary statistics

Questions:

-   Is the feature skewed?
-   Are there extreme values?
-   Are there suspicious ranges?

### Correlation Analysis

Use a correlation matrix for numeric variables.

Questions:

-   Which parameters are strongly correlated?
-   Are some features redundant?
-   Are there relationships worth investigating?

### Class Analysis

Check:

``` text
Class → Number of samples → Percentage
```

This is essential because class imbalance can significantly affect
classification performance.

### Parameter vs Class

Visualize important parameters across classes.

Examples:

-   BOD by class
-   DO by class
-   Turbidity by class
-   pH by class
-   TDS by class

This helps connect the statistical analysis with later SHAP
interpretation.

### Temporal Analysis

Only if timestamps are available.

Analyze:

-   Monthly trends
-   Seasonal patterns
-   Long-term changes
-   Sudden changes

------------------------------------------------------------------------

# 17. Feature Engineering

Feature engineering should be driven by the dataset and domain
understanding.

Potential derived features may include:

``` text
TDS / EC ratio
pH deviation from reference range
DO-related indicators
Temperature-adjusted measurements
Season
Month
Location category
```

However, engineered features must be carefully checked for:

-   Data leakage
-   Redundancy
-   Artificial relationships
-   Availability at prediction time

Do not create dozens of features simply to increase model complexity.

------------------------------------------------------------------------

# 18. Train/Test Strategy

The dataset must be split before final model evaluation.

For a standard classification dataset:

``` text
Dataset
   ↓
Training Set
   ├── Model training
   └── Cross-validation / tuning
   ↓
Final Test Set
   ↓
Final evaluation
```

A typical starting point could be:

``` text
80% Training
20% Testing
```

with stratification for classification.

The exact split should depend on dataset size and structure.

If the dataset has temporal observations, a random split may cause
temporal leakage.

In that situation, use a time-aware split.

If measurements come from multiple locations, consider whether the
evaluation should test generalization to unseen locations.

------------------------------------------------------------------------

# 19. Data Leakage

Data leakage is one of the most important risks in this project.

Examples:

-   Using the target label to create a feature
-   Using a WQI calculated directly from variables to predict that same
    WQI without recognizing the relationship
-   Performing imputation/scaling using the full dataset before
    splitting
-   Selecting features using the test set
-   Using future measurements to predict past conditions
-   Duplicated samples appearing in both train and test sets

Correct approach:

``` text
Split data
    ↓
Fit preprocessing on training data
    ↓
Transform training data
    ↓
Transform test data using training-fitted preprocessing
```

Use scikit-learn Pipelines where possible.

------------------------------------------------------------------------

# 20. Models to Compare

The current project proposal identifies these candidate models:

1.  Logistic Regression
2.  Decision Tree
3.  Random Forest
4.  Support Vector Machine (SVM)
5.  XGBoost
6.  K-Nearest Neighbors (KNN)

These are candidates, not an obligation to use all six.

The final model set should be determined after examining:

-   Dataset size
-   Feature types
-   Class balance
-   Non-linearity
-   Computational cost
-   Interpretability
-   Performance

------------------------------------------------------------------------

# 21. Why Compare Multiple Models?

The project is not supposed to assume that one algorithm is
automatically best.

The comparison provides the research component.

For every model, use the same:

-   Dataset split
-   Target
-   Preprocessing logic
-   Evaluation set
-   Evaluation metrics

Then compare results objectively.

------------------------------------------------------------------------

# 22. Classification Evaluation

Primary metrics:

### Accuracy

``` text
Correct predictions / Total predictions
```

Useful when classes are reasonably balanced.

### Precision

Measures how often predictions of a class are correct.

Important when false positives have consequences.

### Recall

Measures how many actual instances of a class are detected.

Important when missing poor-quality conditions is costly.

### F1-score

Balances precision and recall.

### Confusion Matrix

Shows:

``` text
Actual class
      vs
Predicted class
```

This is particularly useful for understanding which water-quality
categories the model confuses.

------------------------------------------------------------------------

# 23. Do Not Select the Best Model Using Accuracy Alone

If the dataset is imbalanced, a model can achieve high accuracy while
performing poorly on minority classes.

Therefore evaluate:

-   Macro F1
-   Weighted F1
-   Per-class precision
-   Per-class recall
-   Confusion matrix

The final model-selection criterion should be documented before
interpreting results.

Do not decide the criterion after seeing which model wins.

------------------------------------------------------------------------

# 24. Class Imbalance

First calculate:

``` text
Class distribution
```

If significant imbalance exists, investigate:

-   Class weights
-   Stratified splitting
-   Oversampling on training data only
-   SMOTE where appropriate
-   Undersampling where appropriate

Do not apply SMOTE before train/test splitting.

If using SMOTE, the correct sequence is:

``` text
Training data
    ↓
SMOTE
    ↓
Model training

Test data
    ↓
NO SMOTE
    ↓
Final evaluation
```

------------------------------------------------------------------------

# 25. Hyperparameter Tuning

Use a controlled tuning strategy.

Possible methods:

-   GridSearchCV
-   RandomizedSearchCV
-   Optuna, if needed

Do not perform excessive tuning on every model.

The project should prioritize a fair comparison over computationally
expensive optimization.

------------------------------------------------------------------------

# 26. Model Comparison Output

Create a table like:

  Model                   Accuracy   Precision   Recall    F1
  --------------------- ---------- ----------- -------- -----
  Logistic Regression          ...         ...      ...   ...
  Decision Tree                ...         ...      ...   ...
  Random Forest                ...         ...      ...   ...
  SVM                          ...         ...      ...   ...
  XGBoost                      ...         ...      ...   ...
  KNN                          ...         ...      ...   ...

Also produce:

-   Confusion matrix for each model
-   Per-class metrics
-   Training time where useful
-   Cross-validation results where applicable

The table should contain actual experimental results only after
training.

------------------------------------------------------------------------

# 27. Selecting the Final Model

The final model should be selected using a documented criterion.

Example:

``` text
Primary:
Macro F1

Secondary:
Recall for poor-quality classes
Overall accuracy
Precision
Model stability
Interpretability
Computational requirements
```

Do not call a model "best" solely because it has the highest accuracy.

The selection must be tied to the project's objective.

------------------------------------------------------------------------

# 28. SHAP Explainability

## What is SHAP?

SHAP stands for:

> **SHapley Additive exPlanations**

SHAP is an explainability method, not a machine-learning model.

It estimates how individual features contribute to a model prediction.

Conceptually:

``` text
Model prediction
       =
Baseline prediction
       +
Feature contributions
```

For one water-quality prediction:

``` text
Prediction: Poor

Feature contributions:
BOD        → strong contribution
Turbidity  → strong contribution
COD        → moderate contribution
pH         → moderate contribution
Temperature→ small contribution
```

------------------------------------------------------------------------

# 29. SHAP --- Global Explanation

Global SHAP analysis answers:

> Which parameters are generally important to the model?

Example:

``` text
1. BOD
2. Turbidity
3. DO
4. COD
5. TDS
```

This can be displayed as a SHAP summary plot.

------------------------------------------------------------------------

# 30. SHAP --- Local Explanation

Local SHAP analysis answers:

> Why did the model make this particular prediction for this particular
> sample?

Example:

``` text
Sample #1024

Prediction: Poor

Contributors:
BOD        → pushes prediction toward Poor
Turbidity  → pushes prediction toward Poor
DO         → pushes prediction toward Poor
pH         → smaller contribution
```

This is essential to the project's "Explainable Decision Support"
component.

------------------------------------------------------------------------

# 31. SHAP Does Not Prove Causation

This distinction must be maintained throughout the project.

Incorrect:

> "SHAP proved that high BOD caused the pollution."

Correct:

> "SHAP indicates that BOD was an important contributor to the model's
> prediction."

The model learns statistical relationships. SHAP explains the model's
behavior.

It does not independently establish physical causality.

------------------------------------------------------------------------

# 32. From SHAP to Decision Support

This is the key part of the project.

The decision-support layer should have two separate stages.

## Stage 1 --- Model Explanation

SHAP:

``` text
Prediction → Important contributing features
```

## Stage 2 --- Domain Rule Interpretation

Rules:

``` text
Feature condition → Interpretation → Recommended action
```

Together:

``` text
Model Prediction
      ↓
SHAP
      ↓
Important Parameters
      ↓
Check Parameter Values
      ↓
Domain Rules
      ↓
Interpretation
      ↓
Recommendation
```

------------------------------------------------------------------------

# 33. Decision-Support Rule Structure

Each rule should contain:

``` text
Parameter
Condition
Interpretation
Possible contributing factors
Recommended action
Priority
```

Example structure:

``` python
{
    "parameter": "Turbidity",
    "condition": "above_defined_threshold",
    "interpretation": "Elevated suspended material",
    "possible_factors": [
        "Runoff",
        "Sediment input",
        "Other contamination sources"
    ],
    "recommendations": [
        "Investigate nearby runoff/discharge sources",
        "Review filtration/treatment effectiveness",
        "Increase monitoring"
    ],
    "priority": "High"
}
```

The actual thresholds must come from the chosen dataset/standard and
must be documented.

Do not hard-code arbitrary values.

------------------------------------------------------------------------

# 34. Example Decision-Support Rules

These are conceptual examples. Final thresholds and interpretations must
be validated against the selected water-quality standard and dataset.

## High Turbidity

Possible interpretation:

-   Increased suspended material
-   Possible runoff/sediment input
-   Potential contamination indicator depending on context

Possible actions:

-   Investigate runoff sources
-   Investigate discharge points
-   Review filtration/treatment
-   Increase monitoring

## High BOD

Possible interpretation:

-   Increased biodegradable organic load

Possible actions:

-   Investigate untreated or insufficiently treated wastewater
-   Review wastewater treatment performance
-   Monitor BOD and dissolved oxygen

## High COD

Possible interpretation:

-   Elevated oxygen-demanding chemical/organic load

Possible actions:

-   Investigate industrial/domestic discharge
-   Review treatment performance
-   Increase monitoring

## Low Dissolved Oxygen

Possible interpretation:

-   Oxygen depletion

Possible actions:

-   Investigate organic loading
-   Examine BOD/COD conditions
-   Monitor affected locations more frequently

## Abnormal pH

Possible interpretation:

-   Water is outside the relevant acceptable pH range

Possible actions:

-   Investigate acidic/alkaline discharge sources
-   Review treatment processes
-   Verify measurement quality before acting

## High TDS / Conductivity

Possible interpretation:

-   Increased dissolved ionic/soluble material

Possible actions:

-   Investigate potential dissolved-solute sources
-   Check nearby discharge/runoff
-   Verify conductivity/TDS consistency

Again, these are **decision-support interpretations**, not proof of a
specific pollution source.

------------------------------------------------------------------------

# 35. Recommendation Priority

The system can classify recommendations into:

``` text
High Priority
Medium Priority
Low Priority
```

But priority should be rule-based and documented.

For example:

``` text
High:
Multiple critical parameters abnormal
OR
poor-quality class + severe parameter violations

Medium:
One or more significant abnormal parameters

Low:
Minor deviations / monitoring recommendation
```

The actual logic should be finalized after the dataset and applicable
thresholds are known.

------------------------------------------------------------------------

# 36. Confidence

The dashboard may display a model confidence indicator.

However:

> **Do not automatically call a predicted probability "confidence" in a
> statistical sense.**

For example:

``` text
Predicted Class: Poor
Model probability: 0.82
```

is safer than:

``` text
Confidence: 82%
```

unless the probabilities have been appropriately calibrated and you have
defined what "confidence" means.

If probability calibration is implemented, document the calibration
method.

------------------------------------------------------------------------

# 37. Dashboard Design

The dashboard should be simple and demonstrate the entire pipeline.

## Page/Section 1 --- Overview

Display:

-   Project name
-   Dataset information
-   Number of samples
-   Number of parameters
-   Target classes

## Section 2 --- Input

Allow the user to enter or select:

``` text
pH
Temperature
Turbidity
TDS
DO
BOD
COD
...
```

Only show features available in the final model.

## Section 3 --- Prediction

Display:

``` text
Predicted Water Quality
POOR
```

and, where properly supported:

``` text
Predicted class probabilities
```

## Section 4 --- Explanation

Display:

-   SHAP feature importance
-   Local SHAP explanation
-   Top contributing parameters

Example:

``` text
Why this prediction?

BOD         █████████
Turbidity   ███████
COD         █████
DO          ████
```

## Section 5 --- Parameter Analysis

Show:

-   Current value
-   Reference/threshold where available
-   Normal/abnormal status
-   Visualization

## Section 6 --- Decision Support

Display:

``` text
Key observation
Possible interpretation
Recommended action
Priority
```

## Section 7 --- Trends

If timestamps are available:

-   Historical parameter trends
-   Class trends
-   WQI trends if available

------------------------------------------------------------------------

# 38. Suggested Streamlit Layout

``` text
----------------------------------------------------
AI-Driven Water Quality Decision Support
----------------------------------------------------

[Input Parameters]       [Prediction]
                         POOR
                         Probability: ...

----------------------------------------------------

[Why was this predicted?]

SHAP Explanation
-----------------------------------------
BOD          █████████
Turbidity    ███████
COD          █████
DO           ███
-----------------------------------------

----------------------------------------------------

[Parameter Status]

BOD          HIGH
Turbidity    HIGH
DO           LOW
pH           NORMAL

----------------------------------------------------

[Decision Support]

Priority: HIGH

Observation:
Elevated BOD and turbidity contributed strongly
to the predicted poor-quality condition.

Recommended actions:
• Investigate potential wastewater/runoff sources
• Review treatment effectiveness
• Increase monitoring

----------------------------------------------------

[Historical Trends]
----------------------------------------------------
```

------------------------------------------------------------------------

# 39. Software Architecture

Recommended architecture:

``` text
                    ┌──────────────────────┐
                    │   Water Quality Data │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Data Validation &    │
                    │ Preprocessing        │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ EDA + Feature        │
                    │ Engineering          │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ ML Training Pipeline │
                    │ Multiple Models      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Evaluation & Model   │
                    │ Comparison           │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Selected Model       │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ SHAP Explainability │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Decision Support     │
                    │ Rule Engine          │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────────────────┘
```

------------------------------------------------------------------------

# 40. Recommended Technology Stack

## Data Processing

-   Python
-   Pandas
-   NumPy

## Visualization

-   Matplotlib
-   Seaborn
-   Plotly

## Machine Learning

-   Scikit-learn
-   XGBoost

## Explainable AI

-   SHAP

## Dashboard

-   Streamlit

## Development

-   Jupyter Notebook for exploration
-   Python modules for reusable pipeline code
-   Git/GitHub for version control

Optional:

-   Joblib for model serialization
-   YAML/JSON for configuration
-   MLflow if experiment tracking becomes necessary

Do not add infrastructure unless it solves an actual project need.

------------------------------------------------------------------------

# 41. Suggested Project Structure

``` text
water-quality-ai/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_shap_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   └── decision_support/
│       └── rules.py
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   ├── figures/
│   └── model_comparison.csv
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   └── config.yaml
│
├── requirements.txt
├── README.md
└── LICENSE
```

------------------------------------------------------------------------

# 42. Implementation Modules

## Module 1 --- Data Loader

Responsibilities:

-   Load raw dataset
-   Validate required columns
-   Log dataset shape
-   Detect missing values
-   Detect duplicate rows

## Module 2 --- Preprocessing

Responsibilities:

-   Clean column names
-   Convert data types
-   Handle missing values
-   Handle duplicates
-   Handle invalid records
-   Perform appropriate encoding/scaling

## Module 3 --- EDA

Responsibilities:

-   Generate statistics
-   Generate distributions
-   Generate correlation plots
-   Generate class-distribution plots
-   Identify relationships

## Module 4 --- Feature Engineering

Responsibilities:

-   Create justified derived features
-   Encode categorical variables
-   Create temporal features if appropriate
-   Maintain reproducibility

## Module 5 --- Model Training

Responsibilities:

-   Train candidate models
-   Cross-validation
-   Hyperparameter tuning
-   Save model artifacts

## Module 6 --- Evaluation

Responsibilities:

-   Calculate metrics
-   Generate confusion matrices
-   Generate comparison tables
-   Save results

## Module 7 --- SHAP

Responsibilities:

-   Load final model
-   Generate global explanation
-   Generate local explanation
-   Return top contributing features

## Module 8 --- Decision Support

Responsibilities:

-   Check parameter conditions
-   Match relevant rules
-   Generate interpretations
-   Generate recommendations
-   Assign rule-based priority

## Module 9 --- Dashboard

Responsibilities:

-   Accept user inputs
-   Run preprocessing
-   Generate prediction
-   Generate SHAP explanation
-   Generate recommendations
-   Display visualizations

------------------------------------------------------------------------

# 43. Reproducibility

The project must be reproducible.

Record:

-   Dataset source
-   Dataset version/date
-   Cleaning decisions
-   Feature list
-   Target definition
-   Train/test split
-   Random seed
-   Model hyperparameters
-   Evaluation metrics
-   SHAP configuration
-   Decision-support thresholds
-   Python/package versions

Set random seeds where appropriate.

------------------------------------------------------------------------

# 44. Experimental Protocol

Every model should follow the same experimental protocol.

``` text
1. Load dataset
2. Validate schema
3. Clean data
4. Define target
5. Separate X and y
6. Split train/test
7. Fit preprocessing on training data
8. Train models
9. Tune models if required
10. Evaluate on untouched test set
11. Compare metrics
12. Select final model
13. Run SHAP on final model
14. Build decision-support layer
15. Integrate dashboard
```

------------------------------------------------------------------------

# 45. Final Model Artifact

The saved model should include everything required to reproduce
inference.

Prefer a scikit-learn Pipeline:

``` text
Input
 ↓
Preprocessing
 ↓
Model
 ↓
Prediction
```

This avoids applying different preprocessing during dashboard inference.

If preprocessing and model are saved separately, version them together.

------------------------------------------------------------------------

# 46. Prediction Pipeline in the Dashboard

When a user enters values:

``` text
User Input
    ↓
Input Validation
    ↓
Preprocessing
    ↓
Best Model
    ↓
Prediction
    ↓
Probability (if available)
    ↓
SHAP Explanation
    ↓
Parameter Status
    ↓
Decision-Support Rules
    ↓
Recommendation
```

The dashboard must use the exact same preprocessing pipeline used during
training.

------------------------------------------------------------------------

# 47. Example End-to-End Scenario

Suppose a user enters:

``` text
pH = abnormal
Turbidity = high
TDS = high
BOD = high
DO = low
```

The system produces:

### Prediction

``` text
Water Quality: POOR
```

### Model Explanation

``` text
Main contributing parameters:
1. BOD
2. Turbidity
3. DO
4. TDS
```

### Parameter Interpretation

``` text
BOD:
Elevated organic load indicator.

Turbidity:
Elevated suspended material indicator.

DO:
Reduced dissolved oxygen.

TDS:
Elevated dissolved material.
```

### Decision Support

``` text
Priority: HIGH

Recommended investigation:
- Check potential wastewater/runoff sources.
- Review treatment effectiveness.
- Verify abnormal measurements.
- Increase monitoring of BOD, DO, turbidity and TDS.
```

The exact recommendation should depend on the parameter values, class,
available metadata, and validated rules.

------------------------------------------------------------------------

# 48. What "Reason for Poor Quality" Means in This Project

The project should use three levels of wording.

## Level 1 --- Model-level evidence

``` text
BOD had a strong contribution to the model prediction.
```

This comes from SHAP.

## Level 2 --- Parameter interpretation

``` text
Elevated BOD is consistent with increased biodegradable
organic loading.
```

This is domain interpretation.

## Level 3 --- Decision support

``` text
Investigate possible wastewater inputs and treatment
performance.
```

This is a recommended action.

Do not collapse all three into:

``` text
"The model found that Factory X caused the pollution."
```

That claim is outside what this project can establish.

------------------------------------------------------------------------

# 49. Limitations

The final report should explicitly state:

1.  Model predictions depend on dataset quality.
2.  Dataset bias can affect generalization.
3.  Missing or inconsistent measurements can affect predictions.
4.  SHAP explains model behavior, not physical causality.
5.  Decision-support recommendations are rule-based and domain-informed.
6.  Recommendations do not replace laboratory analysis or regulatory
    assessment.
7.  The project does not physically identify a pollution source.
8.  Performance on unseen regions/time periods may differ from the test
    dataset.
9.  Probability outputs may not be calibrated unless calibration is
    explicitly performed.
10. The system's validity depends on the quality and applicability of
    the selected dataset and labeling methodology.

------------------------------------------------------------------------

# 50. How the Base Paper Fits

The base paper should provide the methodological foundation for the
**comparative ML component**.

The intended relationship is:

``` text
BASE PAPER
    ↓
Problem formulation
Dataset methodology
Candidate ML models
Evaluation methodology
    ↓
OUR PROJECT
    ↓
Reproduce/adapt the core comparative ML idea
    +
SHAP Explainability
    +
Parameter interpretation
    +
Decision-support rules
    +
Interactive dashboard
```

The base paper should preferably be closely related to:

-   Water-quality prediction/classification
-   Multiple ML algorithms
-   Comparative evaluation
-   Relevant water-quality parameters
-   A reproducible methodology

A paper focused primarily on IoT hardware should not be selected as the
main methodological base if the project itself is dataset-driven.

------------------------------------------------------------------------

# 51. Literature-to-Project Mapping

The literature survey identified several broad directions:

### Existing direction 1 --- ML prediction

Models such as:

-   Random Forest
-   XGBoost
-   SVM
-   Decision Trees
-   Regression models

### Existing direction 2 --- Deep learning

Examples:

-   ANN
-   CNN
-   LSTM
-   GRU

### Existing direction 3 --- IoT monitoring

Examples:

-   Sensor data collection
-   Real-time monitoring
-   Cloud/edge systems

### Existing direction 4 --- Explainable/intelligent systems

Examples:

-   Explainable AI
-   Anomaly detection
-   Decision support
-   Pollution prediction

Your project combines the relevant dataset-driven components into one
course-project workflow:

``` text
Prediction
+
Model Comparison
+
Explainability
+
Decision Support
```

This framing is already reflected in the project proposal.

------------------------------------------------------------------------

# 52. Project Deliverables

The final project should produce:

## Deliverable 1

Cleaned and analyzed water-quality dataset.

## Deliverable 2

EDA report/notebook.

## Deliverable 3

Multiple trained ML models.

## Deliverable 4

Comparative model evaluation.

## Deliverable 5

Final selected model.

## Deliverable 6

SHAP explainability implementation.

## Deliverable 7

Parameter interpretation and decision-support rule engine.

## Deliverable 8

Interactive Streamlit dashboard.

## Deliverable 9

Technical documentation.

## Deliverable 10

Final project report and presentation.

The current proposal explicitly lists cleaned/analyzed data, EDA,
multiple ML models, comparative evaluation, best-model selection, SHAP
explainability, Streamlit dashboard, and water-quality decision support
as expected deliverables.

------------------------------------------------------------------------

# 53. Suggested Development Phases

## Phase 1 --- Dataset Finalization

Tasks:

-   Collect candidate datasets
-   Inspect schemas
-   Compare target availability
-   Select final dataset
-   Document source

Output:

``` text
Final dataset + data dictionary
```

## Phase 2 --- Data Cleaning

Tasks:

-   Missing values
-   Duplicates
-   Invalid records
-   Outliers
-   Units
-   Data types

Output:

``` text
Clean dataset
```

## Phase 3 --- EDA

Tasks:

-   Distribution
-   Correlation
-   Class balance
-   Feature-vs-class analysis
-   Temporal analysis if available

Output:

``` text
EDA notebook + figures
```

## Phase 4 --- Baseline ML

Train:

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   SVM
-   XGBoost
-   KNN

Output:

``` text
Initial comparison
```

## Phase 5 --- Model Improvement

Tasks:

-   Preprocessing pipelines
-   Class balancing if required
-   Hyperparameter tuning
-   Cross-validation

Output:

``` text
Final comparison
```

## Phase 6 --- Final Model

Tasks:

-   Select model using predefined criteria
-   Freeze model
-   Save pipeline

Output:

``` text
Final inference pipeline
```

## Phase 7 --- Explainability

Tasks:

-   SHAP global analysis
-   SHAP local analysis
-   Top contributing parameters

Output:

``` text
Explainability module
```

## Phase 8 --- Decision Support

Tasks:

-   Define validated parameter thresholds
-   Create rule definitions
-   Map conditions to interpretations
-   Map interpretations to recommendations

Output:

``` text
Decision-support engine
```

## Phase 9 --- Dashboard

Tasks:

-   Build Streamlit UI
-   Add input
-   Add prediction
-   Add SHAP
-   Add parameter status
-   Add recommendations
-   Add trends

Output:

``` text
Working application
```

## Phase 10 --- Validation and Documentation

Tasks:

-   Test pipeline
-   Test edge cases
-   Verify model consistency
-   Verify dashboard output
-   Document limitations

Output:

``` text
Final project
```

------------------------------------------------------------------------

# 54. Testing Strategy

## Data Tests

Check:

-   Required columns exist
-   Numeric fields are numeric
-   No unexpected nulls
-   Valid ranges
-   No duplicate prediction inputs

## Model Tests

Check:

-   Model loads
-   Prediction works
-   Probability output works if supported
-   Input shape is correct
-   Preprocessing matches training

## SHAP Tests

Check:

-   Explanation generated for valid input
-   Feature names match model features
-   Contributions correspond to the correct prediction

## Decision-Support Tests

Test:

-   Normal values
-   One abnormal parameter
-   Multiple abnormal parameters
-   Missing values
-   Boundary values

## UI Tests

Check:

-   Invalid input handling
-   Prediction rendering
-   Explanation rendering
-   Recommendation rendering
-   Charts
-   Error messages

------------------------------------------------------------------------

# 55. Important Edge Cases

The dashboard must handle:

### Missing input

``` text
Please provide all required parameters.
```

### Impossible/invalid value

``` text
Input value is outside the allowed data range.
Please verify the measurement.
```

### Out-of-distribution input

If implemented, warn when the input is substantially different from
training data.

### Conflicting indicators

Example:

``` text
Most parameters indicate acceptable conditions,
but one parameter is strongly abnormal.
```

The system should report the condition rather than hiding it.

### Low-quality prediction

If model probability is ambiguous:

``` text
Prediction is uncertain.
Consider additional measurement or laboratory verification.
```

This is preferable to presenting an uncertain result as absolute truth.

------------------------------------------------------------------------

# 56. Security and Reliability

For a course project, basic safeguards are sufficient:

-   Validate user input
-   Do not execute arbitrary user code
-   Keep model files read-only
-   Avoid storing unnecessary personal information
-   Log errors without exposing sensitive information
-   Version datasets and models
-   Document model versions

------------------------------------------------------------------------

# 57. Success Criteria

The project can be considered successfully implemented when:

### Data

-   Final dataset is documented.
-   Cleaning is reproducible.
-   EDA is complete.

### ML

-   Multiple models are trained.
-   Models are evaluated fairly.
-   A final model is selected using documented criteria.

### XAI

-   Global SHAP explanation works.
-   Local SHAP explanation works.
-   Important contributing parameters can be shown for a prediction.

### Decision Support

-   Abnormal parameters are identified.
-   Interpretations are generated.
-   Recommendations are generated through documented rules.

### Application

-   Streamlit dashboard runs.
-   User can provide water-quality measurements.
-   Prediction is generated.
-   Explanation is displayed.
-   Recommendations are displayed.

------------------------------------------------------------------------

# 58. Final System Definition

The project can be summarized as:

> **An end-to-end machine-learning system that classifies water quality
> from measured water-quality parameters, compares multiple ML models,
> explains predictions using SHAP, identifies the parameters
> contributing to the predicted condition, and translates
> parameter-level abnormalities into domain-informed preventive or
> corrective recommendations through an interactive Streamlit
> decision-support dashboard.**

------------------------------------------------------------------------

# 59. One-Minute Explanation for the Guide

If asked, "What exactly are you building?", explain:

> We are building a Data Science application for water-quality
> classification. We will first clean and analyze a real-world
> water-quality dataset and compare multiple machine-learning models to
> determine how well they classify water-quality conditions. After
> selecting the final model, we will use SHAP to explain why it produced
> a particular prediction and identify the parameters that contributed
> most to that result. We will then interpret abnormal parameters using
> domain-informed rules and provide preventive or corrective
> recommendations. All of this will be presented through an interactive
> Streamlit dashboard.

------------------------------------------------------------------------

# 60. Short Project Pitch

> **The system takes water-quality measurements as input and predicts
> the water-quality condition. Unlike a conventional prediction model,
> it also explains which parameters influenced the prediction and
> provides decision-support recommendations based on abnormal parameter
> conditions. Multiple ML models are compared before selecting the final
> model, and SHAP is used to make the prediction interpretable.**

------------------------------------------------------------------------

# 61. Final Architecture to Freeze

Unless the dataset study reveals a major constraint, the recommended
architecture is:

``` text
                    WATER QUALITY DATA
                           │
                           ▼
              DATA VALIDATION & CLEANING
                           │
                           ▼
                       EDA
                           │
                           ▼
                 FEATURE ENGINEERING
                           │
                           ▼
               WATER QUALITY CLASSIFIER
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       Logistic       Random Forest      XGBoost
       Regression
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  MODEL COMPARISON
                           │
                           ▼
                    FINAL MODEL
                           │
                           ▼
                  SHAP EXPLAINABILITY
                           │
                           ▼
             CONTRIBUTING PARAMETERS
                           │
                           ▼
              PARAMETER INTERPRETATION
                           │
                           ▼
               DECISION-SUPPORT RULES
                           │
                           ▼
             PREVENTIVE / CORRECTIVE
                 RECOMMENDATIONS
                           │
                           ▼
                 STREAMLIT DASHBOARD
```

------------------------------------------------------------------------

# 62. Final Implementation Principle

The project should follow this hierarchy:

``` text
                    CORE
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
     Data             ML            Evaluation
       │              │              │
       └──────────────┼──────────────┘
                      ▼
                Explainability
                      │
                      ▼
              Decision Support
                      │
                      ▼
                  Dashboard
```

The priority is:

**Correct data → reliable experiment → fair model comparison →
interpretable prediction → useful recommendations → usable interface.**

Do not reverse this order by building the dashboard or advanced AI
features before the underlying dataset and ML experiment are sound.

------------------------------------------------------------------------

# 63. Items That Must Be Finalized Before Coding the Full System

These are the remaining project decisions:

### 1. Final dataset

Which dataset will be used as the primary experimental dataset?

### 2. Target label

What exactly constitutes the water-quality class?

### 3. Water-quality standard

What standard or labeling methodology supports the target classes?

### 4. Feature list

Which parameters are available and valid for prediction?

### 5. Base paper

Which published study will provide the methodological foundation for the
comparative ML component?

### 6. Final model set

Which candidate algorithms are appropriate after inspecting the dataset?

### 7. Decision-support rules

Which parameter thresholds and recommendations will be used, and what
authoritative source supports them?

### 8. Dashboard scope

Which visualizations and interactions are necessary for the final
demonstration?

These should be frozen before extensive implementation.

------------------------------------------------------------------------

# 64. Recommended Final Scope

### Must Have

-   One well-documented dataset
-   Data cleaning
-   EDA
-   Feature engineering
-   Multi-class classification
-   4--6 ML models
-   Fair model comparison
-   Final model
-   SHAP global explanation
-   SHAP local explanation
-   Parameter abnormality detection
-   Rule-based decision support
-   Preventive/corrective recommendations
-   Streamlit dashboard

### Good to Have

-   Probability calibration
-   LIME
-   Counterfactual explanations
-   Temporal trends
-   Basic anomaly detection

### Future Scope

-   Live IoT sensors
-   Real-time streaming
-   GIS
-   Multi-source data fusion
-   Deep temporal forecasting
-   Synthetic data generation
-   Sensor drift monitoring
-   Cloud-scale deployment

------------------------------------------------------------------------

# 65. Final Project Philosophy

The project should not simply answer:

> **"What is the water quality?"**

It should create a chain of evidence:

``` text
WHAT?
What is the predicted water-quality condition?

        ↓

WHY?
Which parameters influenced the prediction?

        ↓

WHAT DOES IT MEAN?
What do those abnormal parameters indicate?

        ↓

WHAT CAN BE DONE?
What preventive/corrective actions should be considered?

        ↓

HOW CERTAIN IS IT?
How reliable or ambiguous is the model output?
```

That chain is the core of the **AI-Driven Water Quality Prediction &
Explainable Decision Support System**.
