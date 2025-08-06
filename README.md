# Titanic Survival Prediction with KMeans + KNN

## 🚢 Project Overview

This is a Machine Learning project based on the classic Titanic dataset.  
The goal? Predict whether a passenger survived or not using a combination of unsupervised and supervised learning techniques.

---

## 🔥 What’s special here?

I used **KMeans clustering** as a preprocessing step to help **extract hidden patterns** from the data, then fed those cluster labels as an extra feature into a **K-Nearest Neighbors (KNN)** classifier.  
This combo boosted the prediction accuracy to **~81%** on the test set — pretty solid for this dataset!

---

## 📋 Dataset

- The dataset contains passenger info:  
  `Pclass`, `Sex`, `Age`, `SibSp` (siblings/spouses aboard), `Parch` (parents/children aboard), `Fare`, and `Embarked` (port of embarkation).
- The target is `Survived` (1 = survived, 0 = died).

---

## 🧹 Data Preprocessing

- Filled missing `Age` values with median age.
- Filled missing `Embarked` values with the mode (most common port).
- Converted categorical variables (`Sex` and `Embarked`) into numeric using one-hot encoding (dummy variables).
- Split the data into train (80%) and test (20%) sets.

---

## ⚙️ Model Pipeline

1. **Standard Scaling:**  
   Features are scaled to have mean=0 and variance=1 for better model performance.

2. **KMeans Clustering (Unsupervised):**  
   Applied KMeans clustering with 2 clusters on the scaled training data to capture hidden groups in passengers.

3. **Feature Engineering:**  
   Added the cluster labels predicted by KMeans as a new feature called `"Cluster"` to both train and test datasets.

4. **KNN Classifier (Supervised):**  
   Trained a KNN classifier using the train set augmented with cluster info.

---

## 📈 Results

- The KMeans + KNN combined model achieved an accuracy of approximately **81%** on the test set.
- This approach improved over using KNN alone without cluster features.

---

## 📦 How to Run

1. Make sure you have the following libraries installed:

```bash
pip install pandas scikit-learn
