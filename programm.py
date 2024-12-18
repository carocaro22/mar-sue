import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Model creation and training
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense

# Utilities for TensorBoard
from keras.callbacks import TensorBoard
import datetime
import os
import shutil

class RockPaperScissors: 
    # Recommended sequence_length = 3
    def read_data(self, sequence_length=3, column=1, verbose=False):
        # read data
        data = pd.read_csv('/home/carol/Documents/Master/3_Semester/Anwendung der KI/Project/data.csv')
        if verbose:
            print('Raw Data: ')
            print(data.head())

        # transform into numeric data
        label_encoder = LabelEncoder()
        for col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])
        if verbose: 
            print('Numeric Data: ')
            print(data.head()) 

        # create sequences 
        X = []
        y = []
        for i in range(len(data) - sequence_length):
            X.append(data[[f'H{column}', f'C{column}']].iloc[i:i+sequence_length].values)
            y.append(data[f'H{column}'].iloc[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        if verbose:
            print(X[0])
            print(y[0])
            print(X[1])
            print(y[1])
            print(X[2])
            print(y[2])

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # hot encode y
        output_dim = len(label_encoder.classes_)
        y_train = to_categorical(y_train, num_classes=output_dim)
        y_test = to_categorical(y_test, num_classes=output_dim)

        if verbose:
            print(y_train[0])

        return X_train, X_test, y_train, y_test, output_dim
    
    def train_model(self, hidden_units=10, stackLSTM=True, dropout=0.2, retrain=False, verbose=False): 
        # Define the log directory
        log_dir = './logs/'
        models_dir = './models/'

        if retrain:
            # Clean the log directory if it exists
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            
            # Clean the models directory is it exists
            if os.path.exists(models_dir):
                shutil.rmtree(models_dir)
            
            # Create a new log directory
            os.makedirs(log_dir)
            # Create a new models directory
            os.makedirs(models_dir)

        accuracies = []
        losses = [] 
        labels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
        
        for i in range(1, 7):
            model_path = f'.{models_dir}model_H{i}.h5'
            if not retrain and os.path.exists(model_path):
                # Load the existing model
                model = load_model(model_path)
                print(f'Loaded model from {model_path}')
            else:
            # read data
                X_train, X_test, y_train, y_test, output_dim = self.read_data(column=i, verbose=verbose)

                # get data dimensions
                input_dim = X_train.shape[2] # Human and Computer
                timestep_length = X_train.shape[1] # 5 last rounds
                if not verbose: 
                    print(f'input_dim: {input_dim}')
                    print(f'timestep_length: {timestep_length}')

                # create model
                model = Sequential()
                model.add(LSTM(hidden_units, return_sequences=stackLSTM, input_shape=(timestep_length, input_dim)))
                if stackLSTM:
                    model.add(Dropout(dropout))
                    model.add(LSTM(hidden_units))
                model.add(Dropout(dropout))
                model.add(Dense(output_dim, activation='softmax'))

                # train model
                now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tbCallBack = TensorBoard(log_dir='./logs/' + now, histogram_freq=1, write_graph=True, write_images=False)

                showProgress = 0
                if verbose: 
                    showProgress = 2

                model.compile(
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'], 
                    optimizer='adam'
                    )

                model.fit(
                    X_train,
                    y_train,
                    epochs=30,
                    batch_size=20,
                    verbose=showProgress,
                    validation_split=0.2,
                    callbacks=[tbCallBack]
                )

                # Save the model
                model.save(model_path)
                print(f'Saved model to {model_path}') 

            # Evaluate model
            X_train, X_test, y_train, y_test, output_dim = self.read_data(column=i, verbose=verbose)
            results = model.evaluate(X_test, y_test)
            accuracies.append(results[1])
            losses.append(results[0])

        # Create a bar plot using Matplotlib
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Loss
        ax[0].bar(labels, losses, color='red')
        ax[0].set_title('Loss per Model')
        ax[0].set_ylabel('Loss')
        ax[0].set_ylim(0, 1)

        # Plot Accuracy
        ax[1].bar(labels, accuracies, color='blue')
        ax[1].set_title('Accuracy per Model')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig('model_performance.png')
        plt.show()

obj = RockPaperScissors()
obj.train_model(verbose=True, retrain=True)