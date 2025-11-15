# Survival Analysis - Customer Churn Project

## Overview

This project analyzes customer churn in a telecommunications dataset using Accelerated Failure Time (AFT) models. The analysis compares three parametric survival models (Weibull, Log-Normal, and Log-Logistic) to predict customer lifetime and identify key factors affecting churn. Based on model selection using AIC and concordance metrics, significant features are identified and used to calculate Customer Lifetime Value (CLV) for business insights and retention strategy recommendations.

## Getting Started

### 1. Clone the Repository and Navigate to HW3

git clone <https://github.com/arpine2004/Homework.git> cd Homework/HW3

### 2. Create Virtual Environment

python -m venv venv

### 3. Activate Virtual Environment

**Windows:** venv\Scripts\activate

**Mac/Linux:** source venv/bin/activate

### 4. Install Requirements

pip install -r requirements.txt

### 5. Open Notebook

jupyter notebook

Then open your .ipynb file in the browser.

### 6. Deactivate (when done)

deactivate

## Files

-   Survival_Analysis.ipynb - Main analysis notebook
-   telco.csv - Customer dataset
-   requirements.txt - Required packages
-   README.md - This file

## Requirements

-   Python 3.8+
-   Git installed
-   See requirements.txt for package dependencies
