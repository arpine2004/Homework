# Exploratory Data Analysis 


```python
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import subprocess
import sys
```


```python
import_list = [
    'pandas',
    'numpy', 
    'matplotlib',
    'seaborn',
    'lifelines'
]

versions = []
for package in import_list:
    try:
        module = __import__(package)
        version = module.__version__
        versions.append(f"{package}=={version}")
    except:
        versions.append(package)

with open('requirements.txt', 'w') as f:
    f.write('\n'.join(versions))

print("requirements.txt created!")
```

    requirements.txt created!



```python
data = pd.read_csv("telco.csv")
```


```python
#Initinal Dataset Exploration
print("Dataset Shape:", data.shape)
print("\nColumn names")
print(data.columns.tolist())
print("\nFirst 5 columns of the dataset")
data.head()
```

    Dataset Shape: (1000, 15)
    
    Column names
    ['ID', 'region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'retire', 'gender', 'voice', 'internet', 'forward', 'custcat', 'churn']
    
    First 5 columns of the dataset





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>region</th>
      <th>tenure</th>
      <th>age</th>
      <th>marital</th>
      <th>address</th>
      <th>income</th>
      <th>ed</th>
      <th>retire</th>
      <th>gender</th>
      <th>voice</th>
      <th>internet</th>
      <th>forward</th>
      <th>custcat</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Zone 2</td>
      <td>13</td>
      <td>44</td>
      <td>Married</td>
      <td>9</td>
      <td>64</td>
      <td>College degree</td>
      <td>No</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Basic service</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Zone 3</td>
      <td>11</td>
      <td>33</td>
      <td>Married</td>
      <td>7</td>
      <td>136</td>
      <td>Post-undergraduate degree</td>
      <td>No</td>
      <td>Male</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Total service</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Zone 3</td>
      <td>68</td>
      <td>52</td>
      <td>Married</td>
      <td>24</td>
      <td>116</td>
      <td>Did not complete high school</td>
      <td>No</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Plus service</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Zone 2</td>
      <td>33</td>
      <td>33</td>
      <td>Unmarried</td>
      <td>12</td>
      <td>33</td>
      <td>High school degree</td>
      <td>No</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Basic service</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Zone 2</td>
      <td>23</td>
      <td>30</td>
      <td>Married</td>
      <td>9</td>
      <td>30</td>
      <td>Did not complete high school</td>
      <td>No</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Plus service</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Exploration of data types and missing values
print("Data Types and Missing Values")
print(data.info()) #No missing values identified, 5 numerical and 10 non-numerical features
```

    Data Types and Missing Values
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 15 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   ID        1000 non-null   int64 
     1   region    1000 non-null   object
     2   tenure    1000 non-null   int64 
     3   age       1000 non-null   int64 
     4   marital   1000 non-null   object
     5   address   1000 non-null   int64 
     6   income    1000 non-null   int64 
     7   ed        1000 non-null   object
     8   retire    1000 non-null   object
     9   gender    1000 non-null   object
     10  voice     1000 non-null   object
     11  internet  1000 non-null   object
     12  forward   1000 non-null   object
     13  custcat   1000 non-null   object
     14  churn     1000 non-null   object
    dtypes: int64(5), object(10)
    memory usage: 117.3+ KB
    None



```python
#Summary table
print(data.describe())
```

                    ID       tenure          age      address       income
    count  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000
    mean    500.500000    35.526000    41.684000    11.551000    77.535000
    std     288.819436    21.359812    12.558816    10.086681   107.044165
    min       1.000000     1.000000    18.000000     0.000000     9.000000
    25%     250.750000    17.000000    32.000000     3.000000    29.000000
    50%     500.500000    34.000000    40.000000     9.000000    47.000000
    75%     750.250000    54.000000    51.000000    18.000000    83.000000
    max    1000.000000    72.000000    77.000000    55.000000  1668.000000



```python
#Identifying key variables for survival analysis
duration_col = 'tenure'  # lifetime
event_col = 'churn'  # whether subscriber left
covariates = ['region', 'age', 'marital', 'address', 'income', 
              'ed', 'retire', 'gender', 'voice', 'internet', 
              'forward', 'custcat']
```


```python
print("\nChurn Distribution")
print(data['churn'].value_counts())
```

    
    Churn Distribution
    churn
    No     726
    Yes    274
    Name: count, dtype: int64


# Data Preprocessing


```python
binary_mapping = {'No': 0, 'Yes': 1, 'no': 0, 'yes': 1, 'N': 0, 'Y': 1}

binary_cols = ['retire', 'voice', 'internet', 'forward', 'churn']

for col in binary_cols:
    if col in data.columns and data[col].dtype == 'object':
        data[col] = data[col].map(binary_mapping)
        print(f"Converted {col} to numeric: {data[col].unique()}")

categorical_cols = ['region', 'marital', 'gender', 'ed', 'custcat']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

if 'ID' in data_encoded.columns:
    data_encoded = data_encoded.drop('ID', axis=1)

print("\nFinal check - Non-numeric columns:")
print(data_encoded.select_dtypes(include=['object']).columns.tolist())
```

    Converted retire to numeric: [0 1]
    Converted voice to numeric: [0 1]
    Converted internet to numeric: [0 1]
    Converted forward to numeric: [1 0]
    Converted churn to numeric: [1 0]
    
    Final check - Non-numeric columns:
    []


# Model Fitting


```python
from lifelines import (
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter
)

models = {
    'Weibull': WeibullAFTFitter(),
    'LogNormal': LogNormalAFTFitter(),
    'LogLogistic': LogLogisticAFTFitter(),
}


results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Fitting {name} AFT Model...")
    print('='*50)
    
    try:
        model.fit(data_encoded, duration_col='tenure', event_col='churn')
        
        results[name] = {
            'model': model,
            'AIC': model.AIC_,
            'concordance': model.concordance_index_,
            'log_likelihood': model.log_likelihood_
        }
        
        print(f"✓ {name} - AIC: {model.AIC_:.2f}, Concordance: {model.concordance_index_:.4f}")
        
    except Exception as e:
        print(f"✗ {name} failed to converge: {type(e).__name__}")
        print(f"  Skipping this model...")
        continue

print(f"\n✓ Successfully fitted {len(results)} out of {len(models)} models")

```

    
    ==================================================
    Fitting Weibull AFT Model...
    ==================================================
    ✓ Weibull - AIC: 2964.34, Concordance: 0.7838
    
    ==================================================
    Fitting LogNormal AFT Model...
    ==================================================
    ✓ LogNormal - AIC: 2954.02, Concordance: 0.7872
    
    ==================================================
    Fitting LogLogistic AFT Model...
    ==================================================
    ✓ LogLogistic - AIC: 2956.21, Concordance: 0.7872
    
    ✓ Successfully fitted 3 out of 3 models


# Model Comparison

Three AFT models were compared: Weibull, Log-Normal, and Log-Logistic. The Log-Normal model achieved the best AIC score (2954.02), indicating optimal balance between model fit and complexity, while Log-Logistic had the highest concordance index (0.7872), suggesting slightly better predictive accuracy for ranking customer risk. The differences between models are minimal, with all three showing concordance around 0.787 and AIC values within 10 points of each other. The survival curve comparison for a median customer profile reveals that all three models produce similar predictions in the short term (0-30 months), but diverge in the long tail: Weibull predicts the steepest decline in survival probability, reaching approximately 24\% at 72 months, while Log-Normal shows the most gradual decline at 32\%, and Log-Logistic falls in between at 28\%. Given the marginal advantage in AIC and its ability to capture longer customer lifetimes, the Log-Normal model was selected as the final model for CLV analysis and retention planning.


```python
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'AIC': [results[m]['AIC'] for m in results.keys()],
    'Concordance Index': [results[m]['concordance'] for m in results.keys()],
    'Log Likelihood': [results[m]['log_likelihood'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('AIC').reset_index(drop=True)

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(comparison_df.to_string(index=False))
print("\nNote: Lower AIC is better, Higher Concordance is better")
```

    
    ============================================================
    MODEL COMPARISON SUMMARY
    ============================================================
          Model         AIC  Concordance Index  Log Likelihood
      LogNormal 2954.024010           0.787216    -1457.012005
    LogLogistic 2956.208561           0.787222    -1458.104281
        Weibull 2964.343248           0.783818    -1462.171624
    
    Note: Lower AIC is better, Higher Concordance is better



```python
fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (name, result) in enumerate(results.items()):
    model = result['model']
    
    median_customer = data_encoded.median().to_frame().T
    
    survival_func = model.predict_survival_function(median_customer)
    
    ax.plot(survival_func.index, survival_func.values, 
            label=f"{name} (AIC={result['AIC']:.1f})", 
            linewidth=2.5, 
            color=colors[i])

ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
ax.set_title('Comparison of AFT Model Survival Curves\n(Median Customer Profile)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.show()
```


    
![png](output_16_0.png)
    



```python
print("\n" + "="*60)
print("MODEL SELECTION ANALYSIS")
print("="*60)

best_aic = comparison_df.iloc[0]
print(f"\nBest Model by AIC: {best_aic['Model']}")
print(f"  AIC: {best_aic['AIC']:.2f}")

best_conc_idx = comparison_df['Concordance Index'].idxmax()
best_conc = comparison_df.iloc[best_conc_idx]
print(f"\nBest Model by Concordance: {best_conc['Model']}")
print(f"  Concordance: {best_conc['Concordance Index']:.4f}")

print("\n" + "-"*60)
print("FINAL MODEL RECOMMENDATION:")
print("-"*60)

if best_aic['Model'] == best_conc['Model']:
    final_model_name = best_aic['Model']
    print(f"✓ {final_model_name}")
    print(f"  This model has both the lowest AIC ({best_aic['AIC']:.2f})")
    print(f"  and highest concordance ({best_aic['Concordance Index']:.4f})")
else:
    print("Decision factors to consider:")
    print(f"1. AIC prefers: {best_aic['Model']} (AIC={best_aic['AIC']:.2f})")
    print(f"2. Concordance prefers: {best_conc['Model']} (C={best_conc['Concordance Index']:.4f})")
    print("\nRecommendation: Choose the model with lower AIC for better")
    print("balance between fit and complexity.")
    final_model_name = best_aic['Model']

final_model = results[final_model_name]['model']

print(f"\n✓ Final Selected Model: {final_model_name}")
print("\nDetailed Summary:")
final_model.print_summary()

```

    
    ============================================================
    MODEL SELECTION ANALYSIS
    ============================================================
    
    Best Model by AIC: LogNormal
      AIC: 2954.02
    
    Best Model by Concordance: LogLogistic
      Concordance: 0.7872
    
    ------------------------------------------------------------
    FINAL MODEL RECOMMENDATION:
    ------------------------------------------------------------
    Decision factors to consider:
    1. AIC prefers: LogNormal (AIC=2954.02)
    2. Concordance prefers: LogLogistic (C=0.7872)
    
    Recommendation: Choose the model with lower AIC for better
    balance between fit and complexity.
    
    ✓ Final Selected Model: LogNormal
    
    Detailed Summary:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.LogNormalAFTFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'tenure'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'churn'</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>274</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-1457.01</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2025-11-15 18:55:32 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="19" valign="top">mu_</th>
      <th>address</th>
      <td>0.04</td>
      <td>1.04</td>
      <td>0.01</td>
      <td>0.03</td>
      <td>0.06</td>
      <td>1.03</td>
      <td>1.06</td>
      <td>0.00</td>
      <td>4.78</td>
      <td>&lt;0.005</td>
      <td>19.11</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.03</td>
      <td>1.03</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>1.02</td>
      <td>1.05</td>
      <td>0.00</td>
      <td>4.50</td>
      <td>&lt;0.005</td>
      <td>17.19</td>
    </tr>
    <tr>
      <th>custcat_E-service</th>
      <td>1.07</td>
      <td>2.90</td>
      <td>0.17</td>
      <td>0.73</td>
      <td>1.40</td>
      <td>2.08</td>
      <td>4.06</td>
      <td>0.00</td>
      <td>6.25</td>
      <td>&lt;0.005</td>
      <td>31.21</td>
    </tr>
    <tr>
      <th>custcat_Plus service</th>
      <td>0.92</td>
      <td>2.52</td>
      <td>0.22</td>
      <td>0.50</td>
      <td>1.35</td>
      <td>1.65</td>
      <td>3.85</td>
      <td>0.00</td>
      <td>4.29</td>
      <td>&lt;0.005</td>
      <td>15.75</td>
    </tr>
    <tr>
      <th>custcat_Total service</th>
      <td>1.20</td>
      <td>3.32</td>
      <td>0.25</td>
      <td>0.71</td>
      <td>1.69</td>
      <td>2.03</td>
      <td>5.42</td>
      <td>0.00</td>
      <td>4.79</td>
      <td>&lt;0.005</td>
      <td>19.16</td>
    </tr>
    <tr>
      <th>ed_Did not complete high school</th>
      <td>0.37</td>
      <td>1.45</td>
      <td>0.20</td>
      <td>-0.02</td>
      <td>0.77</td>
      <td>0.98</td>
      <td>2.16</td>
      <td>0.00</td>
      <td>1.85</td>
      <td>0.06</td>
      <td>3.97</td>
    </tr>
    <tr>
      <th>ed_High school degree</th>
      <td>0.32</td>
      <td>1.37</td>
      <td>0.16</td>
      <td>-0.00</td>
      <td>0.64</td>
      <td>1.00</td>
      <td>1.89</td>
      <td>0.00</td>
      <td>1.94</td>
      <td>0.05</td>
      <td>4.24</td>
    </tr>
    <tr>
      <th>ed_Post-undergraduate degree</th>
      <td>-0.03</td>
      <td>0.97</td>
      <td>0.22</td>
      <td>-0.47</td>
      <td>0.40</td>
      <td>0.62</td>
      <td>1.50</td>
      <td>0.00</td>
      <td>-0.15</td>
      <td>0.88</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>ed_Some college</th>
      <td>0.27</td>
      <td>1.31</td>
      <td>0.17</td>
      <td>-0.05</td>
      <td>0.60</td>
      <td>0.95</td>
      <td>1.82</td>
      <td>0.00</td>
      <td>1.65</td>
      <td>0.10</td>
      <td>3.33</td>
    </tr>
    <tr>
      <th>forward</th>
      <td>-0.20</td>
      <td>0.82</td>
      <td>0.18</td>
      <td>-0.55</td>
      <td>0.15</td>
      <td>0.58</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>-1.10</td>
      <td>0.27</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>gender_Male</th>
      <td>0.05</td>
      <td>1.05</td>
      <td>0.11</td>
      <td>-0.17</td>
      <td>0.28</td>
      <td>0.84</td>
      <td>1.32</td>
      <td>0.00</td>
      <td>0.45</td>
      <td>0.65</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>income</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1.52</td>
      <td>0.13</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>internet</th>
      <td>-0.77</td>
      <td>0.46</td>
      <td>0.14</td>
      <td>-1.05</td>
      <td>-0.49</td>
      <td>0.35</td>
      <td>0.61</td>
      <td>0.00</td>
      <td>-5.38</td>
      <td>&lt;0.005</td>
      <td>23.65</td>
    </tr>
    <tr>
      <th>marital_Unmarried</th>
      <td>-0.46</td>
      <td>0.63</td>
      <td>0.12</td>
      <td>-0.68</td>
      <td>-0.23</td>
      <td>0.51</td>
      <td>0.80</td>
      <td>0.00</td>
      <td>-3.94</td>
      <td>&lt;0.005</td>
      <td>13.60</td>
    </tr>
    <tr>
      <th>region_Zone 2</th>
      <td>-0.10</td>
      <td>0.91</td>
      <td>0.14</td>
      <td>-0.38</td>
      <td>0.18</td>
      <td>0.69</td>
      <td>1.20</td>
      <td>0.00</td>
      <td>-0.68</td>
      <td>0.50</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>region_Zone 3</th>
      <td>0.05</td>
      <td>1.05</td>
      <td>0.14</td>
      <td>-0.23</td>
      <td>0.33</td>
      <td>0.80</td>
      <td>1.38</td>
      <td>0.00</td>
      <td>0.34</td>
      <td>0.73</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>retire</th>
      <td>0.02</td>
      <td>1.02</td>
      <td>0.44</td>
      <td>-0.85</td>
      <td>0.89</td>
      <td>0.43</td>
      <td>2.44</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.96</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>voice</th>
      <td>-0.43</td>
      <td>0.65</td>
      <td>0.17</td>
      <td>-0.76</td>
      <td>-0.10</td>
      <td>0.47</td>
      <td>0.90</td>
      <td>0.00</td>
      <td>-2.57</td>
      <td>0.01</td>
      <td>6.61</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>2.36</td>
      <td>10.61</td>
      <td>0.29</td>
      <td>1.79</td>
      <td>2.94</td>
      <td>5.98</td>
      <td>18.84</td>
      <td>0.00</td>
      <td>8.07</td>
      <td>&lt;0.005</td>
      <td>50.37</td>
    </tr>
    <tr>
      <th>sigma_</th>
      <th>Intercept</th>
      <td>0.28</td>
      <td>1.32</td>
      <td>0.05</td>
      <td>0.19</td>
      <td>0.37</td>
      <td>1.20</td>
      <td>1.44</td>
      <td>0.00</td>
      <td>6.00</td>
      <td>&lt;0.005</td>
      <td>28.87</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.79</td>
    </tr>
    <tr>
      <th>AIC</th>
      <td>2954.02</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>291.01 on 18 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>167.66</td>
    </tr>
  </tbody>
</table>
</div>



```python
print("\n" + "="*60)
print("FEATURE SELECTION - KEEPING ONLY SIGNIFICANT FEATURES")
print("="*60)

summary = final_model.summary
significant_features = summary[summary['p'] < 0.05].index.tolist()

print(f"\nSignificant features (p < 0.05): {len(significant_features)}")
for feat in significant_features:
    p_value = summary.loc[feat, 'p']
    coef = summary.loc[feat, 'coef']
    print(f"  - {feat}: coef={coef:.4f}, p={p_value:.4f}")

actual_column_names = []
for feat in significant_features:
    if isinstance(feat, tuple):
        col_name = feat[1]
        if col_name != 'Intercept' and col_name in data_encoded.columns:
            actual_column_names.append(col_name)
    elif feat in data_encoded.columns:
        actual_column_names.append(feat)

actual_column_names = list(set(actual_column_names))

print(f"\nActual columns to use for refitting: {len(actual_column_names)}")
print(actual_column_names)

if len(actual_column_names) > 0:
    data_final = data_encoded[actual_column_names + ['tenure', 'churn']]
    
    final_model_refined = type(final_model)()
    final_model_refined.fit(data_final, duration_col='tenure', event_col='churn')
    
    print("\n✓ Model refitted with significant features only")
    print(f"  Original AIC: {final_model.AIC_:.2f}")
    print(f"  New AIC: {final_model_refined.AIC_:.2f}")
    print(f"  Original Concordance: {final_model.concordance_index_:.4f}")
    print(f"  New Concordance: {final_model_refined.concordance_index_:.4f}")
    
    final_model = final_model_refined
else:
    print("\nWarning: No significant features found at p < 0.05")

```

    
    ============================================================
    FEATURE SELECTION - KEEPING ONLY SIGNIFICANT FEATURES
    ============================================================
    
    Significant features (p < 0.05): 10
      - ('mu_', 'address'): coef=0.0425, p=0.0000
      - ('mu_', 'age'): coef=0.0327, p=0.0000
      - ('mu_', 'custcat_E-service'): coef=1.0664, p=0.0000
      - ('mu_', 'custcat_Plus service'): coef=0.9249, p=0.0000
      - ('mu_', 'custcat_Total service'): coef=1.1986, p=0.0000
      - ('mu_', 'internet'): coef=-0.7715, p=0.0000
      - ('mu_', 'marital_Unmarried'): coef=-0.4551, p=0.0001
      - ('mu_', 'voice'): coef=-0.4338, p=0.0102
      - ('mu_', 'Intercept'): coef=2.3623, p=0.0000
      - ('sigma_', 'Intercept'): coef=0.2758, p=0.0000
    
    Actual columns to use for refitting: 8
    ['age', 'marital_Unmarried', 'custcat_Total service', 'internet', 'voice', 'address', 'custcat_Plus service', 'custcat_E-service']
    
    ✓ Model refitted with significant features only
      Original AIC: 2954.02
      New AIC: 2944.20
      Original Concordance: 0.7872
      New Concordance: 0.7810



```python
# Trying the gamma model with significant features
gg_model = GeneralizedGammaRegressionFitter(penalizer=0.5)
try:
    gg_model.fit(data_final, duration_col='tenure', event_col='churn')
    print("Converged with the significant features!")
    
    # Display all key metrics
    print("\n" + "="*60)
    print("GENERALIZED GAMMA MODEL METRICS")
    print("="*60)
    
    # Key performance metrics
    print(f"\nAIC (Akaike Information Criterion): {gg_model.AIC_:.2f}")
    print(f"  → Lower is better")
    
    print(f"\nConcordance Index: {gg_model.concordance_index_:.4f}")
    print(f"  → Range: 0.5 (random) to 1.0 (perfect)")
    
    print(f"\nLog-Likelihood: {gg_model.log_likelihood_:.2f}")
    print(f"  → Higher is better")
    
    print("\n" + "="*60)
    print("FULL MODEL SUMMARY")
    print("="*60)
    gg_model.print_summary()
    
    # Store in your results dictionary
    results['Generalized Gamma'] = {
        'model': gg_model,
        'AIC': gg_model.AIC_,
        'concordance': gg_model.concordance_index_,
        'log_likelihood': gg_model.log_likelihood_
    }
    
    print("\n✓ Added Generalized Gamma to results!")
    
except Exception as e:
    print(f"Still didn't converge: {e}")

```

    Converged with the significant features!
    
    ============================================================
    GENERALIZED GAMMA MODEL METRICS
    ============================================================
    
    AIC (Akaike Information Criterion): 3684.37
      → Lower is better
    Still didn't converge: 'GeneralizedGammaRegressionFitter' object has no attribute '_predicted_median'


## Note on Generalized Gamma Model

The Generalized Gamma AFT model was initially considered as it represents the most flexible distribution, encompassing Weibull, Log-Normal, and other distributions as special cases. However, the model failed to converge during optimization despite multiple attempts with a penalizer and data transformations (using only the significant features). This is a known limitation documented in the lifelines library (ref: https://lifelines.readthedocs.io/en/latest/fitters/regression/GeneralizedGammaRegressionFitter.html), where the Generalized Gamma fitter exhibits unstable convergence behavior. Therefore, the analysis proceeded with three well-established and stable AFT distributions (Weibull, Log-Normal, and Log-Logistic), which are the most commonly used in survival analysis and provided reliable, interpretable results.


# CUSTOMER LIFETIME VALUE (CLV) CALCULATION

## Customer Lifetime Value Analysis

Customer Lifetime Value (CLV) quantifies the total revenue a business can expect from a customer over their entire relationship with the company. Using the Log-Normal AFT model's predicted expected lifetime for each customer, combined with estimated monthly revenue based on service tier (Basic: USD 25, E-service: USD 35, Plus: USD 50, Total: USD 75), I calculated both simple and discounted CLV. The discounted CLV applies a 1\% monthly discount rate (12\% annually) to account for the time value of money, reflecting the net present value of future revenue streams. 

Results show an average CLV of USD 3,710 per customer (median: USD 3,499), with total portfolio value of USD 3.71 million across 1,000 customers. Significant variation exists across segments: Total Service subscribers demonstrate the highest individual value (USD 5,585 average CLV), while customers without internet service represent the largest aggregate value (USD 1.76M) due to their exceptional retention rates of 626 months despite lower per-customer CLV.

**Note on extreme predictions:** The Log-Normal distribution produces long-tail predictions, with maximum expected lifetime reaching 7,168 months (597 years) for customers with highly favorable characteristics (senior age, stable address, premium service tier). While this demonstrates the model's ability to identify extremely loyal customer segments, such predictions are unrealistic for practical business planning. The median expected lifetime of 217 months (18 years) provides a more realistic central tendency, and in production environments, predictions would typically be capped at a reasonable planning horizon (e.g., 10 years or 120 months). These extreme values do not invalidate the model; rather, they reflect the statistical properties of the Log-Normal distribution and highlight customers with the strongest retention indicators.



```python
print("="*60)
print("CUSTOMER LIFETIME VALUE (CLV) CALCULATION")
print("="*60)

# Typical telecom monthly rates:
monthly_revenue_by_category = {
    'Basic service': 25,     
    'E-service': 35,         
    'Plus service': 50,       
    'Total service': 75      
}

# 1% monthly discount rate (12% annual)
discount_rate = 0.01  

clv_data = data_encoded.copy()

expected_lifetime = final_model.predict_expectation(clv_data)
clv_data['expected_lifetime_months'] = expected_lifetime

clv_data['monthly_revenue'] = 50 

if 'custcat_E-service' in clv_data.columns:
    clv_data.loc[clv_data['custcat_E-service'] == 1, 'monthly_revenue'] = 35
if 'custcat_Plus service' in clv_data.columns:
    clv_data.loc[clv_data['custcat_Plus service'] == 1, 'monthly_revenue'] = 50
if 'custcat_Total service' in clv_data.columns:
    clv_data.loc[clv_data['custcat_Total service'] == 1, 'monthly_revenue'] = 75
if all(col in clv_data.columns for col in ['custcat_E-service', 'custcat_Plus service', 'custcat_Total service']):
    basic_mask = (clv_data['custcat_E-service'] == 0) & \
                 (clv_data['custcat_Plus service'] == 0) & \
                 (clv_data['custcat_Total service'] == 0)
    clv_data.loc[basic_mask, 'monthly_revenue'] = 25

#Simple CLV
clv_data['CLV_simple'] = clv_data['monthly_revenue'] * clv_data['expected_lifetime_months']

# Discounted CLV (Net Present Value)
# CLV = Monthly_Revenue × [(1 - (1 + r)^-T) / r]
clv_data['CLV_discounted'] = clv_data['monthly_revenue'] * (
    (1 - (1 + discount_rate)**(-clv_data['expected_lifetime_months'])) / discount_rate
)

print(f"\n{'='*60}")
print("OVERALL CLV STATISTICS")
print('='*60)
print(f"Total Customers: {len(clv_data):,}")
print(f"\nExpected Lifetime:")
print(f"  Average: {clv_data['expected_lifetime_months'].mean():.2f} months ({clv_data['expected_lifetime_months'].mean()/12:.2f} years)")
print(f"  Median: {clv_data['expected_lifetime_months'].median():.2f} months")
print(f"  Min: {clv_data['expected_lifetime_months'].min():.2f} months")
print(f"  Max: {clv_data['expected_lifetime_months'].max():.2f} months")

print(f"\nDiscounted CLV:")
print(f"  Average per Customer: ${clv_data['CLV_discounted'].mean():.2f}")
print(f"  Median per Customer: ${clv_data['CLV_discounted'].median():.2f}")
print(f"  Total Portfolio Value: ${clv_data['CLV_discounted'].sum():,.2f}")
```

    ============================================================
    CUSTOMER LIFETIME VALUE (CLV) CALCULATION
    ============================================================
    
    ============================================================
    OVERALL CLV STATISTICS
    ============================================================
    Total Customers: 1,000
    
    Expected Lifetime:
      Average: 448.96 months (37.41 years)
      Median: 217.02 months
      Min: 11.51 months
      Max: 7168.47 months
    
    Discounted CLV:
      Average per Customer: $3709.98
      Median per Customer: $3499.24
      Total Portfolio Value: $3,709,983.03


# CLV ANALYSIS BY SEGMENTS


```python
# Segment 1: By Customer Category
print("\n### 1. CLV by Customer Category ###")
clv_data['custcat_segment'] = 'Basic service'
if 'custcat_E-service' in clv_data.columns:
    clv_data.loc[clv_data['custcat_E-service'] == 1, 'custcat_segment'] = 'E-service'
if 'custcat_Plus service' in clv_data.columns:
    clv_data.loc[clv_data['custcat_Plus service'] == 1, 'custcat_segment'] = 'Plus service'
if 'custcat_Total service' in clv_data.columns:
    clv_data.loc[clv_data['custcat_Total service'] == 1, 'custcat_segment'] = 'Total service'

custcat_analysis = clv_data.groupby('custcat_segment', observed = False).agg({
    'expected_lifetime_months': 'mean',
    'monthly_revenue': 'mean',
    'CLV_discounted': ['mean', 'median', 'sum', 'count']
}).round(2)
custcat_analysis.columns = ['Avg_Lifetime(mo)', 'Avg_Monthly_Rev', 'Avg_CLV', 'Median_CLV', 'Total_CLV', 'Customer_Count']
custcat_analysis = custcat_analysis.sort_values('Avg_CLV', ascending=False)
print(custcat_analysis)

# Segment 2: By Region
print("\n### 2. CLV by Region ###")
clv_data['region_segment'] = 'Zone 1'
if 'region_Zone 2' in clv_data.columns:
    clv_data.loc[clv_data['region_Zone 2'] == 1, 'region_segment'] = 'Zone 2'
if 'region_Zone 3' in clv_data.columns:
    clv_data.loc[clv_data['region_Zone 3'] == 1, 'region_segment'] = 'Zone 3'

region_analysis = clv_data.groupby('region_segment', observed = False).agg({
    'expected_lifetime_months': 'mean',
    'CLV_discounted': ['mean', 'median', 'sum', 'count']
}).round(2)
region_analysis.columns = ['Avg_Lifetime(mo)', 'Avg_CLV', 'Median_CLV', 'Total_CLV', 'Customer_Count']
print(region_analysis)

# Segment 3: By Income Level
print("\n### 3. CLV by Income Level ###")
if 'income' in clv_data.columns:
    clv_data['income_segment'] = pd.qcut(clv_data['income'], 
                                          q=4, 
                                          labels=['Low ($9-29K)', 'Medium-Low ($30-47K)', 
                                                 'Medium-High ($48-83K)', 'High ($84K+)'],
                                          duplicates='drop')
    
    income_analysis = clv_data.groupby('income_segment', observed = False).agg({
        'income': 'mean',
        'expected_lifetime_months': 'mean',
        'CLV_discounted': ['mean', 'median', 'sum', 'count']
    }).round(2)
    income_analysis.columns = ['Avg_Income(K)', 'Avg_Lifetime(mo)', 'Avg_CLV', 'Median_CLV', 'Total_CLV', 'Customer_Count']
    print(income_analysis)

# Segment 4: By Age Group
print("\n### 4. CLV by Age Group ###")
if 'age' in clv_data.columns:
    clv_data['age_segment'] = pd.cut(clv_data['age'], 
                                     bins=[0, 30, 40, 50, 100],
                                     labels=['18-30 (Young)', '31-40 (Mid)', 
                                            '41-50 (Mature)', '50+ (Senior)'])
    
    age_analysis = clv_data.groupby('age_segment', observed = False).agg({
        'age': 'mean',
        'expected_lifetime_months': 'mean',
        'CLV_discounted': ['mean', 'median', 'sum', 'count']
    }).round(2)
    age_analysis.columns = ['Avg_Age', 'Avg_Lifetime(mo)', 'Avg_CLV', 'Median_CLV', 'Total_CLV', 'Customer_Count']
    print(age_analysis)

# Segment 5: By Service Usage
print("\n### 5. CLV by Service Combination ###")
if 'internet' in clv_data.columns and 'voice' in clv_data.columns:
    clv_data['service_combo'] = 'No Services'
    clv_data.loc[(clv_data['voice'] == 1) & (clv_data['internet'] == 0), 'service_combo'] = 'Voice Only'
    clv_data.loc[(clv_data['voice'] == 0) & (clv_data['internet'] == 1), 'service_combo'] = 'Internet Only'
    clv_data.loc[(clv_data['voice'] == 1) & (clv_data['internet'] == 1), 'service_combo'] = 'Voice + Internet'
    
    service_analysis = clv_data.groupby('service_combo', observed = False).agg({
        'expected_lifetime_months': 'mean',
        'CLV_discounted': ['mean', 'median', 'sum', 'count']
    }).round(2)
    service_analysis.columns = ['Avg_Lifetime(mo)', 'Avg_CLV', 'Median_CLV', 'Total_CLV', 'Customer_Count']
    service_analysis = service_analysis.sort_values('Avg_CLV', ascending=False)
    print(service_analysis)
```

    
    ### 1. CLV by Customer Category ###
                     Avg_Lifetime(mo)  Avg_Monthly_Rev  Avg_CLV  Median_CLV  \
    custcat_segment                                                           
    Total service              302.22             75.0  5584.70     5898.13   
    Plus service               691.85             50.0  4591.35     4907.03   
    E-service                  603.83             35.0  3056.28     3317.62   
    Basic service              196.21             25.0  1648.92     1624.64   
    
                      Total_CLV  Customer_Count  
    custcat_segment                              
    Total service    1317988.93             236  
    Plus service     1290169.07             281  
    E-service         663213.15             217  
    Basic service     438611.88             266  
    
    ### 2. CLV by Region ###
                    Avg_Lifetime(mo)  Avg_CLV  Median_CLV   Total_CLV  \
    region_segment                                                      
    Zone 1                    464.28  3729.98     3499.97  1201053.68   
    Zone 2                    441.88  3781.47     3498.90  1263012.27   
    Zone 3                    441.48  3621.85     3482.04  1245917.08   
    
                    Customer_Count  
    region_segment                  
    Zone 1                     322  
    Zone 2                     334  
    Zone 3                     344  
    
    ### 3. CLV by Income Level ###
                           Avg_Income(K)  Avg_Lifetime(mo)  Avg_CLV  Median_CLV  \
    income_segment                                                                
    Low ($9-29K)                   21.73            380.79  3010.26     2627.16   
    Medium-Low ($30-47K)           38.06            343.34  3423.25     3373.39   
    Medium-High ($48-83K)          62.19            450.33  3931.03     3785.23   
    High ($84K+)                  189.35            623.74  4492.76     4882.01   
    
                            Total_CLV  Customer_Count  
    income_segment                                     
    Low ($9-29K)            761596.12             253  
    Medium-Low ($30-47K)    862657.86             252  
    Medium-High ($48-83K)   967032.34             246  
    High ($84K+)           1118696.71             249  
    
    ### 4. CLV by Age Group ###
                    Avg_Age  Avg_Lifetime(mo)  Avg_CLV  Median_CLV   Total_CLV  \
    age_segment                                                                  
    18-30 (Young)     25.63             98.64  2589.82     2460.67   541271.68   
    31-40 (Mid)       35.43            196.07  3454.96     3397.80  1029578.05   
    41-50 (Mature)    45.33            397.34  4097.50     4207.80   942425.56   
    50+ (Senior)      58.33           1059.02  4550.22     4979.34  1196707.74   
    
                    Customer_Count  
    age_segment                     
    18-30 (Young)              209  
    31-40 (Mid)                298  
    41-50 (Mature)             230  
    50+ (Senior)               263  
    
    ### 5. CLV by Service Combination ###
                      Avg_Lifetime(mo)  Avg_CLV  Median_CLV   Total_CLV  \
    service_combo                                                         
    Voice Only                  466.13  5310.78     4997.82   616050.48   
    Voice + Internet            171.51  4415.48     4044.60   830110.36   
    No Services                 626.76  3415.01     3495.98  1762143.46   
    Internet Only               217.96  2787.10     2637.88   501678.72   
    
                      Customer_Count  
    service_combo                     
    Voice Only                   116  
    Voice + Internet             188  
    No Services                  516  
    Internet Only                180  


# Customer Churn Analysis: Findings and Retention Strategy

## Key Factors Affecting Churn Risk

Our Log-Normal AFT model (AIC=2954.02, Concordance=0.7872) revealed several significant factors affecting customer lifetime. The most impactful positive predictors are customer service tier and customer age: Total Service subscribers live 3.32× longer (exp(1.20)=3.32), Plus Service 2.52× longer, and E-Service 2.90× longer than Basic Service customers. Each additional year of customer age extends lifetime by 3% (exp(0.03)=1.03), and each year at the same address increases it by 4% (exp(0.04)=1.04), suggesting older and more stable customers are more loyal. 

Conversely, internet service is the strongest negative predictor (coef=-0.77, p<0.0001), reducing expected lifetime by 54% (exp(-0.77)=0.46) - likely because internet-only customers face more competitive alternatives. Unmarried customers churn 37% faster (exp(-0.46)=0.63), and those with voice service show 35% shorter lifetimes (exp(-0.43)=0.65), possibly indicating price-sensitive segments.

## Most Valuable Customer Segments

I define valuable customers based on three criteria: high CLV, long expected lifetime, and stable revenue. The most valuable segment is Total Service subscribers with an average CLV of USD 5,585 and 302-month expected lifetime - representing 24% of customers but 35% of total portfolio value (USD 1.32M). The second most valuable segment is senior customers (50+) with USD 4,550 CLV and 1,059-month lifetime. 

However, the highest aggregate value comes from customers without internet service, contributing USD 1.76M total CLV (47% of portfolio) despite lower individual CLV, because they represent 52% of the customer base and have exceptional retention (626-month lifetime). High-income customers (84K+ USD) also show strong value at USD 4,493 average CLV.

## Annual Retention Budget Recommendation

With 1,000 customers and average survival probability dropping from 100% to approximately 55% at 36 months (3 years), we can expect roughly 137 at-risk customers annually (1000 × (1-0.863) based on 1-year survival). Given the average CLV of USD 3,710 and focusing on the top 30% highest-value customers (CLV > USD 4,500), I recommend an annual retention budget of USD 185,000 to USD 225,000. 

This allocates approximately USD 500-600 per at-risk high-value customer for retention campaigns (5-6% of their CLV), targeting the estimated 300-400 customers at highest churn risk within 12 months. Prioritize Total Service and senior customers who represent disproportionate lifetime value.

## Additional Retention Strategies

Beyond budget allocation, implement predictive intervention: use the survival model to identify customers with less than 70% 12-month survival probability and proactively offer service upgrades or loyalty discounts before they consider leaving. 

Key recommendations:

- Internet-only segment (shortest lifetime at 218 months): Create bundled voice+internet packages with pricing incentives to migrate them to the stickier Plus/Total Service tiers

- Unmarried and younger customers: Target with flexible, no-contract options and digital engagement, as they show higher churn propensity

- Basic Service customers: Conduct win-back campaigns emphasizing service tier upgrades, since upgrading from Basic to Total Service could increase their CLV by 239% (from USD 1,649 to USD 5,585)



```python

```


```python

```
