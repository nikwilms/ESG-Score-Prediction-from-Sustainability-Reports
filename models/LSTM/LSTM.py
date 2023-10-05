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
from models.LSTM.LSTM_eval import LSTM_eval
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold

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
    
    print(bert_embeddings.shape)

    # Target values.
    y = merged_df[target_column].values
    
    # Scaling the target variable
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    # Define KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Iterate through the folds
    for train_index, val_index in kf.split(bert_embeddings):
        X_train, X_val = bert_embeddings[train_index], bert_embeddings[val_index]
        y_train, y_val = y_scaled[train_index], y_scaled[val_index]

        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)


        # run_tuner(X_train, y_train, X_val, y_val, embedding_dim)  # Adjusted validation set

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
        
        stop_early = EarlyStopping(monitor='val_loss', patience=5)
        #checkpoint = ModelCheckpoint('/Users/neuefische/repos/ESG-Score-Prediction-from-Sustainability-Reports/models/LSTM/best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5, verbose=1)

        tuner.search(X_train, y_train,
                     epochs=num_epochs,
                     validation_data=(X_val, y_val),
                     callbacks=[stop_early, reduce_lr])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        #best_hps = run_tuner(X_train, y_train, X_val, y_val, embedding_dim)

        model = tuner.hypermodel.build(best_hps)
    
        # Manually save the model after training
        model_save_path = "Users/neuefische/repos/ESG-Score-Prediction-from-Sustainability-Reports/models/LSTM/best_model.h5"  # Replace with your desired path
        model.save(model_save_path)

        #model.load_weights('/Users/neuefische/repos/ESG-Score-Prediction-from-Sustainability-Reports/models/LSTM/best_model.h5')

        predictions_train = model.predict(X_train)
        predictions_val = model.predict(X_val)

        predictions_train = scaler_y.inverse_transform(predictions_train)
        predictions_val = scaler_y.inverse_transform(predictions_val)
        y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1))
        y_val_original = scaler_y.inverse_transform(y_val.reshape(-1, 1))

        rmse_train = sqrt(mean_squared_error(y_train_original, predictions_train))
        rmse_val = sqrt(mean_squared_error(y_val_original, predictions_val))

        #predictions_test = model.predict(X_test)
        #rmse_test = sqrt(mean_squared_error(y_test, predictions_test))

        print(f"Train RMSE: {rmse_train:.3f}")
        print(f"Validation RMSE: {rmse_val:.3f}")
        #print(f"Test RMSE: {rmse_test:.3f}")

        LSTM_eval(model, X_train, y_train, X_val, y_val)
    
    return model, rmse_train #, rmse_test
