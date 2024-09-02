# Telco-Customer-Churn-Prediction-and-Retention-Strategy


# **Customer Churn Prediction Project**

## **Overview**

This project aims to predict customer churn for a telecom company using machine learning models. The objective is to identify customers who are likely to leave the company so that targeted retention strategies can be implemented. The dataset used for this project contains information about customer demographics, services, account information, and churn status.

## **Business and Data Understanding**

### **Business Problem**

Customer churn is a critical issue for telecom companies, directly impacting revenue and growth. By accurately predicting which customers are at risk of leaving, the company can proactively implement retention strategies, optimize marketing campaigns, and enhance customer satisfaction.

### **Dataset Description**

The dataset contains 7,043 rows and 20 columns, with each row representing a customer and columns providing various attributes, such as:

- **Customer Demographic Information:**
  - `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Customer Account Information:**
  - `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services Signed Up by Customer:**
  - `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target Variable:**
  - `Churn`: Indicates whether the customer churned or not (Yes/No)

## **Data Preparation**

### **Data Cleaning**

1. Handled missing values in the dataset, particularly in the `TotalCharges` column, by dropping the missing values.
2. Converted categorical variables to numerical format using one-hot encoding to make them suitable for machine learning models.
3. Normalized numerical features to ensure that all data inputs are on a similar scale, which is important for models like Logistic Regression.

### **Feature Engineering**

1. Created new features based on domain knowledge to provide additional predictive power to the models.
2. Analyzed feature importance using SHAP (SHapley Additive exPlanations) values to understand which features contribute the most to the model’s predictions.

## **Modeling**

Three different models were used to predict customer churn:

1. **Logistic Regression:** A simple, interpretable model to serve as a baseline.
2. **Decision Tree Classifier:** A non-parametric model that can capture complex patterns in the data.
3. **Random Forest Classifier:** An ensemble model known for its robustness and ability to handle large datasets with many features.

### **Handling Imbalanced Data**

Due to the imbalanced nature of the target variable (more non-churners than churners), various resampling techniques were employed:

- **SMOTE (Synthetic Minority Over-sampling Technique):** To generate synthetic samples for the minority class.
- **Undersampling:** To balance the dataset by reducing the number of majority class samples.
- **Combination of SMOTE and Undersampling:** To further improve the model's ability to predict churners accurately.

## **Model Evaluation**

Each model was evaluated based on the following metrics:

- **Accuracy:** Overall correctness of the model.
- **Precision:** Proportion of true positives among the predicted positives.
- **Recall:** Proportion of true positives among the actual positives.
- **F1-Score:** Harmonic mean of precision and recall, which balances the two metrics.

### **Results**

#### **Logistic Regression Performance:**

- **Accuracy:** 80%
- **Precision for Churners:** 65%
- **Recall for Churners:** 55%
- **F1-Score for Churners:** 60%

#### **Decision Tree Performance:**

- **Accuracy:** 78%
- **Precision for Churners:** 58%
- **Recall for Churners:** 65%
- **F1-Score for Churners:** 61%

#### **Random Forest with Best Hyperparameters Performance:**

- **Accuracy:** 79%
- **Precision for Churners:** 63%
- **Recall for Churners:** 51%
- **F1-Score for Churners:** 56%

## **Hyperparameter Tuning**

To further enhance the model's performance, hyperparameters for the Random Forest model were fine-tuned using `GridSearchCV`:

- **Best Parameters Found:**
  - `n_estimators`: 200
  - `max_depth`: 20
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
  - `max_features`: 'sqrt'
  - `bootstrap`: True

The tuned model achieved an accuracy of 79%, with improved precision and recall metrics.


### **Conclusion:**

The customer churn prediction model developed in this project offers a valuable tool for identifying customers at risk of leaving the company. The Random Forest model, fine-tuned for optimal performance, achieved an accuracy of 79%, with good overall metrics for predicting non-churning customers. However, there is room for improvement in identifying churners, which are crucial for proactive customer retention efforts.

By leveraging this model, the company can make data-driven decisions to better understand customer behavior, reduce churn, and ultimately enhance customer retention strategies. While the model is already providing actionable insights, further refinements and enhancements can increase its effectiveness in identifying customers at risk.

### **Recommendations:**

1. **Deploy the Model for Customer Retention:**
   - Integrate the model into the company's customer relationship management (CRM) system to identify at-risk customers in real time. This will enable the customer service team to proactively engage with these customers through personalized retention offers, discounts, or enhanced support.

2. **Enhance Feature Development:**
   - Invest in creating new features that capture customer behavior more accurately. This could include developing metrics around usage patterns, service interactions, and customer feedback to provide deeper insights into factors influencing churn.

3. **Continue Model Refinement:**
   - Regularly retrain the model with new data to ensure its predictions remain accurate over time. As market conditions and customer behaviors change, updating the model will keep it aligned with current trends.

4. **Leverage Advanced Algorithms:**
   - Explore more advanced algorithms such as Gradient Boosting Machines (e.g., XGBoost, LightGBM) or ensemble methods that might yield better performance and provide more granular insights into customer churn patterns.

5. **Conduct A/B Testing:**
   - Implement A/B testing with different retention strategies (e.g., personalized offers vs. general promotions) to determine the most effective methods for reducing churn. Use the model's predictions to target the right customers and measure the impact of different interventions.

6. **Focus on High-Risk Segments:**
   - Prioritize efforts on high-risk customer segments identified by the model. Tailor specific marketing and retention strategies to these segments, such as special loyalty programs or exclusive benefits, to increase customer satisfaction and reduce churn.

7. **Monitor and Measure Impact:**
   - Develop a dashboard to continuously monitor the performance of the churn model and the effectiveness of retention strategies. Measure key performance indicators (KPIs) such as churn rate reduction, customer lifetime value (CLV), and return on investment (ROI) for retention campaigns.

By following these recommendations, the company can enhance its customer retention efforts, reduce churn rates, and ultimately improve customer loyalty and profitability.

## **Files in the Repository**

- `Customer_Churn_Prediction.ipynb`: Main notebook containing all the analysis, model training, and evaluation.
- `data/`: Folder containing the dataset used for the project.
- `images/`: Folder containing images generated during the analysis.
- `requirements.txt`: File listing all required Python packages.

