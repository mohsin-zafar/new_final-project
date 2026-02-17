# 📓 Google Colab Training Instructions

## Step-by-Step Guide to Train the Model in Google Colab

---

## Step 1: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click "New Notebook"

---

## Step 2: Upload Files

### Upload the Dataset

Run this cell to upload your CSV file:

```python
from google.colab import files
uploaded = files.upload()
```

Select `manufacturing_dataset_1000_samples.csv` from your computer.

---

## Step 3: Install Required Packages

Run this cell:

```python
!pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## Step 4: Copy and Run Training Code

Copy the entire content from `training/train.py` into a new cell and run it.

Or upload the `train.py` file and run:

```python
%run train.py
```

---

## Step 5: Download Model Files

After training completes, download the model files:

```python
from google.colab import files

# Download model file
files.download('model.pkl')

# Download scaler file
files.download('scaler.pkl')

# Download label encoders
files.download('label_encoders.pkl')

# Download feature names
files.download('feature_names.pkl')

# Optional: Download training visualization
files.download('training_results.png')
```

---

## Step 6: Place Files in Project

1. Create the `model/` folder in your project if it doesn't exist
2. Move all downloaded `.pkl` files to the `model/` folder:
   - `model.pkl`
   - `scaler.pkl`
   - `label_encoders.pkl`
   - `feature_names.pkl`

---

## Expected Output

When training completes, you should see:

```
============================================================
TRAINING COMPLETE!
============================================================

Saved Files:
  ✓ model.pkl - Trained Linear Regression model
  ✓ scaler.pkl - StandardScaler for feature scaling
  ✓ label_encoders.pkl - Label encoders for categorical variables
  ✓ feature_names.pkl - List of feature names
  ✓ training_results.png - Visualization plots

Model Performance:
  • Test R² Score: ~0.93
  • Test RMSE: ~4.2
```

---

## Troubleshooting

### Issue: FileNotFoundError for dataset

**Solution:** Make sure you uploaded the CSV file with the exact name `manufacturing_dataset_1000_samples.csv`

### Issue: Module not found error

**Solution:** Run the pip install cell again:
```python
!pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Issue: Memory error

**Solution:** 
- Use the free GPU/TPU runtime: Runtime → Change runtime type → GPU
- Or restart the runtime: Runtime → Restart runtime

---

## Quick Reference Commands

```python
# Upload files
from google.colab import files
uploaded = files.upload()

# List files in current directory
!ls -la

# Check file exists
import os
os.path.exists('manufacturing_dataset_1000_samples.csv')

# Download all model files at once
!zip models.zip model.pkl scaler.pkl label_encoders.pkl feature_names.pkl
files.download('models.zip')
```

---

## Complete Colab Notebook Code

Here's the complete code to run in a single Colab notebook:

```python
# Cell 1: Install packages
!pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Cell 2: Upload dataset
from google.colab import files
print("Please upload manufacturing_dataset_1000_samples.csv")
uploaded = files.upload()

# Cell 3: Run training (paste train.py content here or upload and run)
# ... training code ...

# Cell 4: Download results
files.download('model.pkl')
files.download('scaler.pkl')
files.download('label_encoders.pkl')
files.download('feature_names.pkl')
```

---

**Happy Training! 🚀**
