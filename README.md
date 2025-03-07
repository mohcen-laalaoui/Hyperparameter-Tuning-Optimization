# 🔥 Hyperparameter Tuning & Optimization

## 🚀 Overview
This project focuses on optimizing the hyperparameters of a **Random Forest Classifier** using **Optuna** and **Scikit-Learn** to improve model performance. The goal is to explore different tuning strategies, evaluate model performance, and analyze the impact of hyperparameters.

## 📌 Features
- **Model Selection**: Uses `RandomForestClassifier` from Scikit-Learn.
- **Hyperparameter Tuning**: Implements **Bayesian Optimization** with Optuna.
- **Cross-Validation**: Evaluates model performance using `cross_val_score`.
- **Performance Metrics**: Compares accuracy, precision, recall, and F1-score before and after tuning.

## 📂 Project Structure
```
📁 hyperparameter-tuning-optimization
│-- 📄 README.md
│-- 📄 breast_cancer.csv
│-- 📄 Model Hyperparameter Tuning.ipynb  
```

## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mohcen-laalaoui/Hyperparameter-Tuning-Optimization.git
cd Hyperparameter-Tuning-Optimization
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook
```
Open `Model Hyperparameter Tuning.ipynb` and execute the cells.

## ⚡ Hyperparameter Tuning Strategy
We use **Optuna** to optimize the following hyperparameters:
- `n_estimators`: Number of trees (50-500)
- `max_depth`: Maximum tree depth (5-50)
- `min_samples_split`: Minimum samples required to split (5-20)
- `min_samples_leaf`: Minimum leaf node samples (5-15)
- `max_features`: Feature selection method (`sqrt`, `log2`, None)

### Example of Optuna Search Space
```python
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 5, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 15)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
    )
    
    return cross_val_score(rf_model, X_train, y_train, cv=10, scoring="accuracy").mean()
```

## 📊 Results & Observations
- **Before Tuning**: Accuracy = `0.9560`
- **After Tuning**: Accuracy varies due to model complexity and dataset structure.
- **Findings**: Increasing `max_depth` and `n_estimators` beyond optimal values **caused overfitting**.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests with improvements or new tuning techniques!

## 🌟 Acknowledgments
- **Scikit-Learn**: Machine learning library.
- **Optuna**: Hyperparameter optimization framework.
- **Jupyter Notebook**: Interactive computing environment.

---
🔗 **GitHub Repository:** https://github.com/mohcen-laalaoui/Hyperparameter-Tuning-Optimization

