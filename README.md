# Data_Science_in_Practice

PySpark mini-project for **Income Classification**.  
Developed for the **Postgraduate Diploma in Data Science (Wits University, 2025)**.

---

## 🎯 Project Objective
This project applies scalable machine learning techniques in **Apache Spark (PySpark)** to predict whether a person’s income exceeds a certain threshold based on demographic and employment attributes.

---

## 🧰 Tools & Technologies
- 🪶 **Apache Spark (PySpark)**
- 🔢 **Spark MLlib**
- 🧩 **StringIndexer**, **VectorAssembler**, **Pipeline**
- 🌲 **DecisionTreeClassifier** and **RandomForestClassifier**
- 📏 **MulticlassClassificationEvaluator**

---

## 📊 Dataset
- **File:** [`data/income.csv`](data/income.csv)
- The dataset contains demographic and employment features (e.g., age, education, occupation, workclass).
- The target variable is **`income_class`** (binary classification).

---

## 🧪 Methodology
1. Load and clean dataset (`income.csv`)
2. Drop missing values
3. Encode categorical variables using **StringIndexer**
4. Combine numerical and encoded features with **VectorAssembler**
5. Split into **training (80%)** and **testing (20%)**
6. Train and evaluate **Decision Tree** and **Random Forest** models
7. Report model performance metrics

---

## 📈 Results Summary
| Model | Accuracy |
|--------|-----------|
| Decision Tree | ~85% |
| Random Forest | ~89% |

> 🏆 The Random Forest model outperformed the Decision Tree due to ensemble averaging, leading to better generalization.

---

## 🚀 How to Run the Code
1. Ensure Spark is installed locally.
2. Place `income.csv` in the `/data` folder.
3. Run the script from the command line:
   ```bash
   spark-submit src/Assessment2_Income.py
