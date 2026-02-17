# Model Files Directory

This folder should contain the trained model files:

## Required Files

After training, place these files here:

1. **model.pkl** - Trained Linear Regression model
2. **scaler.pkl** - StandardScaler for feature scaling
3. **label_encoders.pkl** - Label encoders for categorical variables
4. **feature_names.pkl** - List of feature names in correct order

## How to Get These Files

### Option 1: Train in Google Colab

1. Upload `training/train.py` to Google Colab
2. Upload `dataset/manufacturing_dataset_1000_samples.csv`
3. Run the training script
4. Download the generated `.pkl` files
5. Place them in this `model/` folder

### Option 2: Train Locally

```bash
cd training
python train.py
mv *.pkl ../model/
```

## Important Notes

- All 4 `.pkl` files are required for the backend to work
- Do not modify the pickle files manually
- If you retrain the model, replace all files together
