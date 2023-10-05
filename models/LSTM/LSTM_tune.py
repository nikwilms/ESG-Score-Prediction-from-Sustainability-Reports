from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from kerastuner.tuners import RandomSearch

def build_model(hp, embedding_dim):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   input_shape=(1, embedding_dim), return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run_tuner(X_train, y_train, X_val, y_val, embedding_dim):
    tuner = RandomSearch(
        lambda hp: build_model(hp, embedding_dim),
        objective='val_loss',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='helloworld'
    )

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_hps
