import pandas as pd
import statsmodels.api as sm

# Generate the "true" data with known relationships
observDF = pd.DataFrame({
    'Stress': [0, 0, 0, 1, 1, 1, 2, 2, 2, 8, 8, 8, 12, 12, 12],
    'StressSurvey': [0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 12, 12, 12],
    'Time': [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2.1, 2.2, 2.2, 2.2],
    'Anxiety': [0, 0.1, 0.1, 1.1, 1.1, 1.1, 2.2, 2.2, 2.2, 8.2, 8.2, 8.21, 12.22, 12.22, 12.22]
})

# 1. Bivariate Regression Analysis with StressSurvey
print("--- Bivariate Regression Analysis with StressSurvey ---")
X_survey = observDF['StressSurvey']
y = observDF['Anxiety']
X_survey_const = sm.add_constant(X_survey)
model_survey = sm.OLS(y, X_survey_const).fit()
print(model_survey.summary())

# 3. Bivariate Regression Analysis with Time
print("\n--- Bivariate Regression Analysis with Time ---")
X_time = observDF['Time']
X_time_const = sm.add_constant(X_time)
model_time = sm.OLS(y, X_time_const).fit()
print(model_time.summary())

# 5. Multiple Regression Analysis with StressSurvey and Time
print("\n--- Multiple Regression Analysis with StressSurvey and Time ---")
X_multi_survey = observDF[['StressSurvey', 'Time']]
X_multi_survey_const = sm.add_constant(X_multi_survey)
model_multi_survey = sm.OLS(y, X_multi_survey_const).fit()
print(model_multi_survey.summary())

# 6. Multiple Regression Analysis with Stress and Time
print("\n--- Multiple Regression Analysis with Stress and Time ---")
X_multi_true = observDF[['Stress', 'Time']]
X_multi_true_const = sm.add_constant(X_multi_true)
model_multi_true = sm.OLS(y, X_multi_true_const).fit()
print(model_multi_true.summary())

# 9. Subset Analysis
print("\n--- Subset Analysis ---")
subsetDF = observDF[observDF['StressSurvey'] <= 6].copy()
X_subset = subsetDF[['StressSurvey', 'Time']]
y_subset = subsetDF['Anxiety']
X_subset_const = sm.add_constant(X_subset)
model_subset = sm.OLS(y_subset, X_subset_const).fit()
print(model_subset.summary())