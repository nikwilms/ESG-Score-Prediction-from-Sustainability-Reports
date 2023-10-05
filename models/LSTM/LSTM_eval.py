import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

def LSTM_eval(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model and print RMSE, MAE, R^2 for both training and test sets.
    Plot the actual vs predicted values.

    Parameters:
    - model: Trained model.
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Test data and labels.
    """
    
    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    # Print metrics
    print(f'Training RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')
    print(f'\nTraining MAE: {train_mae:.2f}')
    print(f'Test MAE: {test_mae:.2f}')
    print(f'\nTraining R^2: {train_r2:.2f}')
    print(f'Test R^2: {test_r2:.2f}')
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, train_preds, alpha=0.5)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
    plt.title('Training set: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, test_preds, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.title('Test set: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.tight_layout()
    plt.show()
