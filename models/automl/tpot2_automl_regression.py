from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, make_scorer
import tpot2
import pandas as pd

# Read your data
# Assuming 'df' is your DataFrame and 'ESG_Score' is your target column
df = pd.read_csv("path/to/your/esg_data.csv")

# Separate features and target
X = df.drop(["ESG_Score"], axis=1)
y = df["ESG_Score"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a custom scorer (optional)
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Initialize TPOT2 regressor with K-Fold cross-validation
# Make sure to adjust the generations and population_size based on your computational resources and needs. 
est = tpot2.TPOTEstimator(
    generations=5,
    population_size=20,
    n_jobs=-1,
    cv=KFold(n_splits=5),  # 5-Fold cross-validation
    verbosity=2,
    classification=False,
    scorers=[scorer],
    scorers_weights=[1],
)

# Fit the model
est.fit(X_train, y_train)

# Evaluate on test data
test_score = est.score(X_test, y_test)

# Print test score
print(f"Test Score (Negative MSE): {test_score}")

# Get the best pipeline
print(f"Optimized Pipeline: {est.fitted_pipeline_}")

# Get top models in the Pareto front
pareto_front = est.evaluated_individuals[est.evaluated_individuals['Pareto_Front'] == 1]
print(f"Top models in the Pareto front: {pareto_front}")

# Optionally, you can save the pipeline to a file
# with open("optimized_pipeline_regression.py", "w") as f:
#     f.write(est.export())

# Optionally, you can save the pipeline to a file
# with open("optimized_pipeline_regression.py", "w") as f:
#     f.write(est.export())
