# AI-ML-internship-tasks
# Task 1: Exploring and Visualizing the Iris Dataset

## Task Objective
The objective of this project was to explore the Iris dataset to understand the trends, distributions, and relationships among its features using data analysis and visualization techniques.

## Dataset Used
The project utilizes the classic Iris dataset, which is commonly used for pattern recognition and contains measurements of sepal length, sepal width, petal length, and petal width for three species of Iris flowers (setosa, versicolor, and virginica). The dataset was loaded directly using the `seaborn` library.

## Key Results and Findings
Through data exploration and visualization, the following key insights were gained:

- The dataset contains 150 observations and 5 features, with no missing values.
- Descriptive statistics provided an overview of the central tendency, variability, and distribution of the numerical features.
- Scatter plots revealed clear relationships between features and showed how different species cluster based on sepal and petal dimensions. Petal length and width were particularly effective in separating the species.
- Histograms illustrated the distribution of each feature, highlighting the bimodal nature of petal length and petal width distributions, which correspond to the distinct characteristics of the different Iris species.
- Box plots helped visualize the spread and potential outliers for each feature within each species, further emphasizing the separability of the species based on petal dimensions.

In summary, the exploratory analysis and visualizations demonstrated that the Iris species can be effectively differentiated based on their sepal and petal measurements, with petal dimensions being the most distinguishing features.

# Task 2: House Price Prediction Project

## 1. Project Description

The goal of this project is to build a regression model to accurately predict house prices using the House Sales in King County, USA dataset. The dataset contains various features related to residential properties, including structural information, geographical location, and condition. The project aims to explore the data, preprocess it, build and evaluate several regression models, and identify the best-performing model for predicting house prices.

## 2. Dataset

The dataset used in this project is the "House Sales in King County, USA" dataset from Kaggle. It includes the following columns:

*   `date`: Date the house was sold
*   `price`: Sale price of the house (Target Variable)
*   `bedrooms`: Number of bedrooms
*   `bathrooms`: Number of bathrooms
*   `sqft_living`: Living area in square feet
*   `sqft_lot`: Lot size in square feet
*   `floors`: Number of floors
*   `waterfront`: Whether the property is on the waterfront (binary)
*   `view`: Quality of the view
*   `condition`: Condition of the house (1 to 5)
*   `sqft_above`: Square feet above ground
*   `sqft_basement`: Square feet of the basement
*   `yr_built`: Year the house was built
*   `yr_renovated`: Year the house was renovated
*   `street`: Street address
*   `city`: City
*   `statezip`: State and Zip code
*   `country`: Country

The dataset initially contains 4600 entries with no missing values.

## 3. Methodology

The project followed a standard machine learning workflow:

### 3.1. Data Understanding and Initial Observation

*   Examined the first few rows of the dataset (`df.head()`) to understand the column names, data types, and features.
*   Used `df.info()` to check data types and identify non-null counts.
*   Generated statistical summaries (`df.describe()`) for numerical features to understand distributions, central tendencies, and potential outliers.

### 3.2. Exploratory Data Analysis (EDA)

*   Visualized the distribution of key numerical features like `price`, `bedrooms`, and `bathrooms`.
*   Analyzed the relationship between `price` and features like `condition` and `sqft_living` using box plots and scatter plots.
*   Examined the distribution of houses across different cities.
*   Generated a correlation matrix to understand the relationships between numerical features and the target variable (`price`).

### 3.3. Outlier Detection and Removal

*   Applied the IQR (Interquartile Range) method to detect and remove outliers from key numerical features (`price`, `sqft_living`, `bathrooms`, `bedrooms`, `sqft_above`, `sqft_basement`, `sqft_lot`).
*   This step reduced the dataset size from 4600 rows to 3588 rows, improving data consistency.

### 3.4. Feature Engineering

*   Extracted `year_sold` from the `date` column.
*   Created `house_age` by subtracting `yr_built` from `year_sold`.
*   Created a binary feature `has_been_renovated` based on `yr_renovated`.
*   Dropped irrelevant or redundant columns: `date`, `yr_renovated`, `yr_built`, `street`, and `country`.

### 3.5. Data Preprocessing: Encoding, Scaling & Train/Test Split

*   Separated the target variable (`price`) from the features (`X`).
*   Split the data into training (80%) and testing (20%) sets.
*   Used `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder` for categorical features (`waterfront`, `view`, `condition`, `city`, `statezip`, `has_been_renovated`).
*   Created a `Pipeline` to combine preprocessing and modeling steps.

### 3.6. Modeling

*   Trained several regression models:
    *   Linear Regression
    *   Decision Tree Regressor
    *   Random Forest Regressor
    *   Gradient Boosting Regressor
    *   XGBoost Regressor
*   Models were trained using the preprocessed training data.

### 3.7. Model Evaluation and Comparison

*   Evaluated each trained model on the test set using the following metrics:
    *   Mean Absolute Error (MAE)
    *   Root Mean Squared Error (RMSE)
    *   R² Score
*   Compared the performance of all models using a table and bar charts.

### 3.8. Feature Importance Analysis (XGBoost)

*   Analyzed the feature importance scores from the best-performing XGBoost model to understand which features had the most influence on predictions.

### 3.9. Log Transformation and Re-evaluation

*   Applied a log transformation (`np.log1p`) to the target variable (`price`) to address its skewed distribution.
*   Re-trained the XGBoost model on the log-transformed target.
*   Reversed the log transformation (`np.expm1`) on the predictions to evaluate performance on the original price scale.
*   Visualized actual vs predicted prices and residuals to analyze the model's performance with the transformed target.
*   Applied 5-Fold Cross-Validation to the XGBoost model trained on the log-transformed target to assess its generalization ability.

## 4. Results and Findings

*   Initial EDA revealed a heavily right-skewed price distribution and the presence of outliers in several numerical features.
*   Outlier removal and feature engineering steps helped in cleaning and preparing the data for modeling.
*   Among the initial models tested, **Linear Regression** showed the best performance based on MAE, RMSE, and R² scores on the initial test split. However, XGBoost was considered more suitable for real-world scenarios due to its ability to handle non-linear patterns and potential for further tuning.
*   Feature importance analysis on the XGBoost model highlighted the significant impact of **location-related features** (city, statezip) on house prices.
*   Applying log transformation to the target variable and re-training XGBoost resulted in a model with different performance characteristics. While the scatter and residual plots showed a good general fit, the cross-validation results indicated potential issues with generalization, emphasizing the importance of robust evaluation techniques.

| Model             | MAE       | RMSE       | R² Score |
| :---------------- | :-------- | :--------- | :------- |
| Linear Regression | 66894.08  | 104615.71  | 0.74     |
| XGBoost           | 71506.84  | 112382.09  | 0.70     |
| Random Forest     | 77455.14  | 120017.94  | 0.65     |
| Gradient Boosting | 81955.89  | 120997.40  | 0.65     |
| Decision Tree     | 100565.50 | 149387.33  | 0.46     |

*Note: The evaluation metrics after log transformation and cross-validation are presented separately in the notebook.*

## 5. Conclusion

The project successfully explored and preprocessed the house price dataset, built and evaluated several regression models, and identified key factors influencing house prices. While Linear Regression performed best on the initial test split, XGBoost was chosen as the preferred model due to its advanced capabilities. The application of log transformation and cross-validation provided deeper insights into model stability and generalization, highlighting areas for potential future improvement, such as hyperparameter tuning and exploring alternative feature engineering strategies.

## 6. Future Work

*   Perform hyperparameter tuning for the best-performing models (XGBoost, Linear Regression) to potentially improve performance.
*   Explore other feature engineering techniques, such as creating interaction terms or polynomial features.
*   Investigate the impact of different encoding strategies for categorical variables.
*   Consider advanced techniques for handling outliers or skewed distributions.
*   Explore other regression models or ensemble methods.
