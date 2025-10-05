# Tecno Pocket Go Innovation Diffusion Analysis

## Bass Diffusion Model Analysis of AR Gaming Device Adoption

---

## Folder Structure
```
HW1/
│
├── data/
│ └── AR_Revenue_Dataset.xlsx # Historical AR market revenue data (2017-2029)
│
├── img/
│ ├── actual_predicted_plot.png # Model fit visualization
│ ├── Analysis.png # Comprehensive analysis plots
│ └── historical_data_plot.png # Historical data visualization
│
├── report/
│ └── (PDF report documents)
│
├── script.ipynb # Main Jupyter notebook
│
└── README.md # This file
```
---

## Requirements

### Python Libraries
- pandas
- numpy
- scipy
- matplotlib

### Installation

pip install pandas numpy scipy matplotlib openpyxl


---

## Getting Started

### 1. Data Preparation
- Ensure `AR_Revenue_Dataset.xlsx` is in the `data/` folder
- Dataset should contain AR B2C market revenue data from 2017-2029
- Data source: Statista

### 2. Running the Notebook

Open the Jupyter notebook:

jupyter notebook script.ipynb

---


## Generated Images (in `img/` folder)
- `historical_data_plot.png` - Historical AR market revenue visualization
- `actual_predicted_plot.png` - Model fit comparison
- `Analysis.png` - Comprehensive diffusion analysis with multiple subplots


---

## Usage Instructions

### Step-by-Step Execution

1. **Run all cells sequentially** from top to bottom
2. **First-time setup**: Ensure file paths match your local directory structure
3. **Update file path** in the data import section if needed:

file_path = 'data/AR_Revenue_Dataset.xlsx'

### Key Parameters to Observe

- **p (Innovation coefficient)**: External influence factor
- **q (Imitation coefficient)**: Word-of-mouth influence factor
- **M (Market potential)**: Maximum market size

---

## Troubleshooting

### Common Issues

**Issue**: File not found error
- **Solution**: Verify the file path matches your folder structure
- Update the path in the notebook: `file_path = 'data/AR_Revenue_Dataset.xlsx'`

**Issue**: Missing libraries
- **Solution**: Install required packages using pip:

pip install pandas numpy scipy matplotlib openpyxl

**Issue**: Plots not displaying
- **Solution**: Ensure matplotlib backend is configured correctly
- Add at the beginning: `%matplotlib inline`

**Issue**: Images not saving
- **Solution**: Create the `img/` folder manually if it doesn't exist
- Or update save paths in the notebook

---

## Notes

- The notebook uses data from 2017-2025 for model training
- Forecasts extend beyond 2025 based on Bass Model predictions
- All monetary values are in USD (billions)
- The analysis focuses on worldwide AR market trends

---
.

## License

This project is submitted as homework for the Marketing Analytics course.



