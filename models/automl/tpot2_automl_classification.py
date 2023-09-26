# The ESG Risk Ratings are categorized across five risk levels:
# negligible (0-10), low (10-20), medium (20-30), high (30-40) and severe (40+).


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
import tpot2
import pandas as pd


# Function to categorize ESG risk scores
def categorize_esg_score(score):
    if score >= 0 and score < 10:
        return "Negligible"
    elif score >= 10 and score < 20:
        return "Low"
    elif score >= 20 and score < 30:
        return "Medium"
    elif score >= 30 and score < 40:
        return "High"
    else:
        return "Severe"


# Read your data
# Assuming 'df' is your DataFrame and 'ESG_Score' is your ESG score column
df = pd.read_csv("/mnt/data/your_esg_data.csv")

# Convert ESG scores to categories
df["ESG_Category"] = df["ESG_Score"].apply(categorize_esg_score)

# Separate features and target
X = df.drop(["ESG_Score", "ESG_Category"], axis=1)
y = df["ESG_Category"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create a custom scorer (optional)
scorer = make_scorer(roc_auc_score, multi_class="ovr", needs_proba=True)

# Initialize TPOT2 classifier with Stratified K-Fold cross-validation
est = tpot2.TPOTEstimator(
    generations=5,
    population_size=20,
    n_jobs=-1,
    cv=StratifiedKFold(n_splits=5),  # 5-Fold cross-validation
    verbosity=2,
    classification=True,
    scorers=[scorer],
    scorers_weights=[1],
)

# Fit the model
est.fit(X_train, y_train)

# Evaluate on test data
test_score = est.score(X_test, y_test)

# Print test score
print(f"Test Score (AUC-ROC): {test_score}")

# Get the best pipeline
print(f"Optimized Pipeline: {est.fitted_pipeline_}")

# Optionally, you can save the pipeline to a file
# with open("optimized_pipeline_classification.py", "w") as f:
#     f.write(est.export())
