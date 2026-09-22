# Diabetes Progression Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project uses an **Artificial Neural Network (ANN)** to model and predict diabetes disease progression based on multiple independent variables.

The project uses the **Diabetes dataset available in Scikit-learn** and follows a complete machine learning workflow including:

* Data loading
* Data preprocessing
* Missing-value checking
* Feature scaling
* Exploratory Data Analysis (EDA)
* ANN model development
* Model training
* Model evaluation
* ANN architecture improvement
* Performance comparison

The main objective is to understand how different patient-related features can be used to predict diabetes progression.

---

## 🎯 Objective

The objective of this project is to develop an ANN regression model that can predict the progression of diabetes using the available independent variables.

The project also compares a **baseline ANN model** with an **improved ANN model** to evaluate whether architectural and training changes can improve prediction performance.

---

## 📊 Dataset

The project uses the **Diabetes dataset from Scikit-learn**.

The dataset contains:

* **442 samples**
* **10 independent variables**
* **1 target variable**

### Features

| Feature | Description             |
| ------- | ----------------------- |
| `age`   | Age-related measurement |
| `sex`   | Sex-related measurement |
| `bmi`   | Body Mass Index         |
| `bp`    | Blood pressure          |
| `s1`    | Serum measurement 1     |
| `s2`    | Serum measurement 2     |
| `s3`    | Serum measurement 3     |
| `s4`    | Serum measurement 4     |
| `s5`    | Serum measurement 5     |
| `s6`    | Serum measurement 6     |

### Target

The target variable represents a quantitative measure of **diabetes disease progression**.

Since the target is continuous, this project is treated as a **regression problem**.

---

## 🛠️ Technologies and Libraries Used

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* Keras
* Matplotlib
* Seaborn

---

## 🔄 Project Workflow

```text
Load Diabetes Dataset
        ↓
Check Missing Values
        ↓
Feature Scaling
        ↓
Exploratory Data Analysis
        ↓
Train-Test Split
        ↓
Build Baseline ANN
        ↓
Train Model
        ↓
Evaluate Model
        ↓
Improve ANN Architecture
        ↓
Train Improved Model
        ↓
Evaluate Improved Model
        ↓
Compare Results
```

---

# 1. Data Loading and Preprocessing

The Diabetes dataset was loaded using:

```python
from sklearn.datasets import load_diabetes

diabetes_data = load_diabetes(as_frame=True)
```

The independent variables were separated into `X`, while the target variable was stored in `y`.

```python
X = diabetes_data.data
y = diabetes_data.target
```

### Missing Value Check

Missing values were checked using:

```python
X.isnull().sum()
```

The dataset contained **no missing values** in any of the input features.

Therefore, no missing-value imputation was required.

### Feature Scaling

The input features were standardized using `StandardScaler`:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The scaled data was then converted back into a Pandas DataFrame.

Feature scaling was performed because neural networks generally perform better when input features are on comparable scales.

---

# 2. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and relationships between the features and diabetes progression.

### Statistical Summary

The `describe()` function was used to examine:

* Count
* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

### Target Distribution

A histogram with a KDE curve was used to visualize the distribution of the diabetes progression target.

```python
sns.histplot(y, kde=True)
```

This helped visualize how the target values were distributed.

### Feature Correlation

The correlation of each feature with the target was calculated and visualized using a bar plot.

```python
correlations = dataset_df.corr()['Target'].drop('Target')
```

This helped identify the strength and direction of the linear relationship between individual features and diabetes progression.

---

# 3. Baseline ANN Model

A baseline Artificial Neural Network was created using TensorFlow/Keras.

### Architecture

```text
Input Layer
    ↓
Dense Layer - 16 neurons
    ↓
Output Layer - 1 neuron
```

The hidden layer uses the **ReLU activation function**.

The output layer uses a **linear activation function**, which is suitable for a regression problem.

### Model Code

```python
model_baseline = Sequential([
    Dense(
        16,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ),
    Dense(
        1,
        activation='linear'
    )
])
```

### Compilation

The model was compiled using:

* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error
* **Metric:** Mean Squared Error

```python
model_baseline.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_squared_error']
)
```

### Training

The baseline model was trained with:

* **Epochs:** 100
* **Batch size:** 16
* **Validation split:** 20%

---

# 4. Baseline Model Evaluation

The baseline model was evaluated using:

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted values.

A lower MSE indicates lower prediction error.

### R² Score

R² measures how much of the variation in the target variable is explained by the model.

The baseline model achieved:

| Metric   | Baseline ANN |
| -------- | -----------: |
| MSE      |  **6508.38** |
| R² Score |  **-0.2284** |

The negative R² score indicates that the baseline model did not perform well on the test data.

---

# 5. Improving the ANN Model

The baseline model was modified to improve its ability to learn more complex relationships.

The following changes were made:

### 1. Increased the number of neurons

The first hidden layer was increased from **16 neurons to 64 neurons**.

### 2. Added another hidden layer

A second hidden layer containing **32 neurons** was added.

The improved architecture therefore became:

```text
Input Layer
     ↓
Dense - 64 neurons
     ↓
LeakyReLU
     ↓
Dropout - 20%
     ↓
Dense - 32 neurons
     ↓
LeakyReLU
     ↓
Output - 1 neuron
```

### 3. LeakyReLU Activation

Instead of standard ReLU, the improved model uses **LeakyReLU**.

This allows a small gradient for negative inputs and helps avoid inactive neurons during training.

### 4. Dropout

A dropout rate of **20%** was introduced.

Dropout randomly disables a portion of neurons during training and was used to help reduce overfitting.

### 5. Learning Rate Adjustment

The improved model used the Adam optimizer with an initial learning rate of:

```text
0.01
```

A `ReduceLROnPlateau` callback was also used to automatically reduce the learning rate when validation loss stopped improving.

```python
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.0001
)
```

### 6. Increased Training Epochs

The number of training epochs was increased from:

```text
100 → 150
```

---

# 6. Improved Model Results

The improved ANN achieved:

| Metric             | Baseline Model | Improved Model |
| ------------------ | -------------: | -------------: |
| Mean Squared Error |        6508.38 |    **2694.03** |
| R² Score           |        -0.2284 |     **0.4915** |

### Performance Change

The MSE decreased from:

```text
6508.38 → 2694.03
```

This represents approximately a **58.6% reduction in MSE**.

The R² score changed from:

```text
-0.2284 → 0.4915
```

The improved model therefore explained approximately **49% of the variance** in the test-set diabetes progression values.

---

# 📈 Results Summary

```text
                    Baseline        Improved
                    --------        --------
MSE                 6508.38         2694.03
R²                  -0.2284         0.4915
```

The improved ANN showed substantially lower prediction error and a positive R² score compared with the baseline model.

---

# 🧠 Key Learning Outcomes

Through this project, the following concepts were explored:

* Loading datasets using Scikit-learn
* Separating independent and dependent variables
* Checking for missing values
* Feature standardization
* Exploratory Data Analysis
* Correlation analysis
* Train-test splitting
* Artificial Neural Networks
* Dense layers
* ReLU activation
* LeakyReLU activation
* Dropout regularization
* Adam optimizer
* Mean Squared Error
* R² Score
* Learning-rate scheduling
* Comparing baseline and improved models

---

# 📝 Conclusion

This project developed an Artificial Neural Network for predicting diabetes disease progression using the Scikit-learn Diabetes dataset.

The baseline ANN produced an MSE of **6508.38** and an R² score of **-0.2284**. The model was then improved by increasing the network size, adding a second hidden layer, using LeakyReLU activation, introducing Dropout regularization, and applying dynamic learning-rate adjustment.

The improved model achieved an MSE of **2694.03** and an R² score of **0.4915**, showing a substantial reduction in prediction error compared with the baseline model.

This project demonstrates how ANN architecture and training strategies can be experimented with to improve regression performance.
