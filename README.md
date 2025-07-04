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

---

## Explanation for the code for myself :

import pandas as pd
df = pd.read_csv("train.csv") 
 we imported the pandas library and then we are going to read the training dataset

df.head()
this displays the five rows of the dataset

df.info()
this shows the summary of the dataframe like the number of rows etc

df.describe()
this provides the descriptive status like the mean median etc

df.columns
this lists all column names of the dataframe

features = ["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]
data = df[features]
data = data.dropna()
the first line selects three features as input and an output from the dataset
then we are assigning the data as the features and dropping the missing values in it

import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(data)
plt.show()
then we are importing the libararies seaborn and matplotlib for the visualisation and paiplot is used for creating scatter plot the last line displays the plot

X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]
in x axis the features which are used to predict the prices are given that is the independent variables
in y axis is the feature which is the output

