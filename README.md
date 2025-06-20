# 🏡 House Price Prediction using Linear Regression

This beginner-level Machine Learning project predicts house prices using a **Linear Regression** model. It uses key housing features like:
- **GrLivArea** (Above ground living area in square feet)
- **BedroomAbvGr** (Number of bedrooms)
- **FullBath** (Number of full bathrooms)

We’ll train and test the model on the popular **Kaggle: House Prices - Advanced Regression Techniques** dataset.

---

## 📁 Dataset

- **Source**: [Kaggle Competition Page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- You need to join the competition to download the dataset.
- File used:
  - `train.csv` — contains training data with input features and target (`SalePrice`)

---

## 🧰 Project Structure

```
prodigy_task1/
│
├── train.csv                  # Dataset file
├── house_price_prediction.ipynb  # Main Jupyter notebook
├── requirements.txt           # (Optional) List of dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

---

## 🚀 Installation Instructions

### 🐍 1. Clone this repository

```bash
git clone https://github.com/JissaAanJuby/house-price-prediction-ML.git
cd house-price-prediction-ML
```

---

### 💻 2. Create and activate virtual environment (optional but recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment:
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

---

### 📦 3. Install dependencies

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or manually install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ How to Run

### Option 1: Use Jupyter Notebook

```bash
jupyter notebook
```
- Open `house_price_prediction.ipynb`
- Run the cells one by one

### Option 2: Use VS Code
- Open VS Code
- Open the folder `prodigy_task1`
- Select your Python interpreter from the bottom-left status bar
- Open the `.ipynb` notebook or create a `.py` script and run

---

## 📊 Step-by-Step Breakdown of the Code

```python
# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

```python
# 2. Load the dataset
df = pd.read_csv('train.csv')
df.head()
```

```python
# 3. Select useful features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = df[features]
y = df[target]
```

```python
# 4. Optional: Visualize data
sns.pairplot(df[features + [target]])
plt.show()
```

```python
# 5. Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
# 6. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)
```

```python
# 7. Make predictions
y_pred = model.predict(X_test)
```

```python
# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
```

```python
# 9. Visualize predictions
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()
```

---

## 📈 Sample Output

```
Mean Absolute Error: 25401.10
Mean Squared Error: 1999804587.44
R^2 Score: 0.73
```

---

## 📚 Learning Resources

Short & Beginner-friendly videos:
- 🎥 [What is Linear Regression?](https://www.youtube.com/watch?v=ZkjP5RJLQF4)
- 🎥 [Simple Linear Regression in Python](https://www.youtube.com/watch?v=E5RjzSK0fvY)
- 🎥 [Train/Test Split & Evaluation](https://www.youtube.com/watch?v=Q81RR3yKn30)

---

## 🧾 License

This project is for educational use only. Not affiliated with Kaggle or any organization.
