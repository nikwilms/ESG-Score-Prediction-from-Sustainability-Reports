import optuna
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from models.LDA_optuna_tuning.lda_optuna_tune import lda_optuna_tune  # Assuming this is your objective function

def call_optuna_tune(df, n_trials=50):
    """
    Tune hyperparameters for LDA model using Optuna.
    
    Args:
        df (DataFrame): DataFrame containing the 'preprocessed_content' column.
        n_trials (int): Number of trials for Optuna optimization. Default is 50.
        
    Returns:
        dict: Dictionary containing best parameters and best coherence score.
    """
    
    # Create a study object and specify the direction is 'maximize'
    study = optuna.create_study(direction='maximize')
    
    # Optimize the study
    study.optimize(lambda trial: lda_optuna_tune(trial, df), n_trials=n_trials)
    
    # Get the best parameters and best score
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"The best parameters are {best_params} with a coherence score of {best_score}")
    
    return {'best_params': best_params, 'best_score': best_score}

# Use the function
results = call_optuna_tune(lda_test_df)  # Replace lda_test_df with your DataFrame
