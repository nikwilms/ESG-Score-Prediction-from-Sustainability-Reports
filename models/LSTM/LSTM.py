from models.LSTM.LSTM_tune import run_tuner
from models.LSTM.LSTM_tune import build_model
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
from helpers.topic_modelling.get_embeddings import get_embeddings
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
import os
from helpers.topic_modelling.get_embeddings import get_embeddings
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch



def train_lstm_model(ready_to_model_df, df_with_target, target_column, embedding_dim, num_epochs=50, batch_size=32):
    """
    Train an LSTM model using precomputed BERT embeddings.

    Parameters:
    - ready_to_model_df: DataFrame containing BERT embeddings.
    - df_with_target: DataFrame containing target values.
    - target_column: Name of the column containing target values.
    - embedding_dim: Size of the BERT embeddings.
    - num_epochs: Number of epochs to train for.
    - batch_size: Training batch size.

    Returns:
    - model: Trained LSTM model.
    - rmse_train: Root Mean Squared Error on the training set.
    - rmse_test: Root Mean Squared Error on the test set.
    """
    
    # Merge DataFrames on the 'ticker' and 'year' columns
    merged_df = pd.merge(ready_to_model_df, df_with_target, left_on=['ticker', 'year'], right_on=['Company_Symbol', 'year'], how='inner')
    merged_df.columns = [col.lower().replace('-', '_') for col in merged_df.columns]

    # Ensure embeddings are not already present in the dataframe to avoid recomputing.
    if 'esg_bert_embeddings' not in merged_df.columns:
        # Ensure all entries are strings.
        merged_df['preprocessed_content'] = merged_df['preprocessed_content'].astype(str)

    # Check for existing embeddings file
    if os.path.exists('../data/model_data/esg_bert_embeddings.csv'):
        # Load existing embeddings
        bert_embeddings = pd.read_csv('../data/model_data/esg_bert_embeddings.csv')
    else:
        # Generate embeddings
        merged_df['preprocessed_content'] = merged_df['preprocessed_content'].astype(str)
        merged_df['esg_bert_embeddings'] = merged_df['preprocessed_content'].apply(get_embeddings)
        bert_embeddings = pd.DataFrame(merged_df['esg_bert_embeddings'].tolist(), index=merged_df.index)
        merged_df = pd.concat([merged_df, bert_embeddings], axis=1)
        merged_df.drop(columns=['esg_bert_embeddings'], inplace=True)
        
        # Save the BERT embeddings as a CSV file
        bert_embeddings.to_csv('../data/model_data/esg_bert_embeddings.csv', index=False)
    
    # Assuming the BERT embeddings are in the last `embedding_dim` columns of the dataframe.
    bert_embeddings = merged_df.iloc[:, -embedding_dim:].values
    bert_embeddings = np.expand_dims(bert_embeddings, axis=1) 
    
    # Target values.
    y = merged_df[target_column].values
    
    # Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(bert_embeddings, y, test_size=0.2, random_state=42)
    
    run_tuner(X_train, y_train, X_test, y_test, embedding_dim)

    # Define a model-building function for the tuner, taking into account the embedding_dim
    def build_model(hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                       input_shape=(1, embedding_dim), return_sequences=True))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    # Initialize the tuner
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='helloworld'
    )

    # Define early stopping
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    # Start the tuner search
    tuner.search(X_train, y_train,
                 epochs=num_epochs,
                 validation_data=(X_test, y_test),
                 callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), batch_size=batch_size, verbose=1)

    # Predictions and Performance Evaluation.
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    
    rmse_train = sqrt(mean_squared_error(y_train, predictions_train))
    rmse_test = sqrt(mean_squared_error(y_test, predictions_test))
    
    print(f"Train RMSE: {rmse_train:.3f}")
    print(f"Test RMSE: {rmse_test:.3f}")
    
    return model, rmse_train, rmse_test
