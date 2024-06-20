# SalaryPredictor

Predicts salaries based on years of experience, test scores, and interview scores using an AI model.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
This project includes an AI model that estimates the salaries of individuals based on their years of experience, test scores, and interview scores. The model is built using a linear regression algorithm.

## Features
- Predicts salary for different experience levels and scores
- Handles missing values in the dataset
- Provides visualizations for data exploration and correlation
- Easy to use with simple input parameters
- Includes examples and usage instructions

## Installation
To run this project, you need to have Python and the following libraries installed:
- pandas
- seaborn
- scikit-learn
- word2number
- numpy
- matplotlib

You can install the required libraries using the following commands:
```bash
pip install pandas seaborn scikit-learn matplotlib word2number
```
## Usage

1. Clone the repository
```bash
git clone https://github.com/salihfurkaan/SalaryPredictor.git
cd SalaryPredictor
```

2. Prepare your dataset in the same format as `hiring.csv`:
     Columns should include experience, test_score(out of 10), interview_score(out of 10), and salary($)
   
4. Run the following code to train the model and predict salaries:
```py
import pandas as pd
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from word2number import w2n

# Load dataset
df = pd.read_csv("hiring.csv")

# Handle missing values
df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)
test_score_median = df["test_score(out of 10)"].median()
interview_score_median = df["interview_score(out of 10)"].median()
df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(test_score_median)
df["interview_score(out of 10)"] = df["interview_score(out of 10)"].fillna(interview_score_median)

# Data visualization
sns.pairplot(df)
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()
sns.regplot(data=df, x=df["experience"], y=df["salary($)"], line_kws=dict(color="red"))
plt.show()

# Prepare data for model
X = df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
y = df["salary($)"]

# Train model
model = linear_model.LinearRegression()
model.fit(X, y)

# Make predictions
print(model.predict([[2, 9, 6]]))  # Predict salary for 2 years experience, 9 test score, 6 interview score
print(model.predict([[12, 10, 10]]))  # Predict salary for 12 years experience, 10 test score, 10 interview score
```

4. Interpret the results printed by the model for the given input values.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License
This project is licensed under the MIT License.

## Contact
For any questions or inquiries, please contact slhfurkaan@gmail.com





