import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
import os
os.environ["CUDA_VISIBLE_DEVICES"]=0


# Function to create the CNN model
def create_model(learning_rate, momentum, num_kernels_1, num_kernels_2, reg_param):
    model = Sequential()
    
    # First convolutional layer: kernel size equals full sequence length
    model.add(Conv2D(
        filters=num_kernels_1,
        kernel_size=[1,sequence_length'],
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(reg_param),
        input_shape=(sequence_length, num_features)
    ))

    model.add(layers.BatchNormalization())
    
    # Second convolutional layer: smaller kernel size
    model.add(Conv2D(
        filters=num_kernels_2,
        kernel_size=[16,1],
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(reg_param)
    ))
    
    # Flatten and fully connected layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Compile the model
    optimizer = SGD(learning_rate=learning_rate, decay=1e-6, momentum=momentum)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])
    
    return model


# Hyperparameter ranges
learning_rates = [0.001, 0.01, 0.1]
momentums = [0.1, 0.5, 0.9]
num_kernels_1_options = [10, 30, 50]
num_kernels_2_options = [10, 30, 50]
reg_params = [0.0001, 0.001, 0.01]

# Function for hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    best_accuracy = 0
    best_params = {}
    
    for lr, mom, nk1, nk2, reg in product(learning_rates, momentums, num_kernels_1_options, num_kernels_2_options, reg_params):
        print(f"Testing combination: lr={lr}, momentum={mom}, nk1={nk1}, nk2={nk2}, reg_param={reg}")
        model = create_model(lr, mom, nk1, nk2, reg)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Shorter epochs for testing
        
        # Evaluate on the test set
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'lr': lr, 'momentum': mom, 'num_kernels_1': nk1, 'num_kernels_2': nk2, 'reg_param': reg}
    
    return best_accuracy, best_params

# Perform hyperparameter tuning

dataset = np.load('data.npz')

X_train, y_train = dataset['train_x'], to_categorical(dataset['train_y'])
X_vali, y_vali = dataset['vali_x'], to_categorical(dataset['vali_y'])
X_test, y_test = dataset['test_x'], to_categorical(dataset['test_y'])
# Learning
model_cb = keras.callbacks.ModelCheckpoint(MODEL+'_rrrs.best.h5', verbose=1, save_best_only=True)
stop_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=(X_vali, y_vali), callbacks=[model_cb, stop_cb])


best_accuracy, best_params = hyperparameter_tuning(X_train, y_train, X_test, y_test)

# Print the best parameters and accuracy
print(f"Best Accuracy: {best_accuracy}")
print(f"Best Parameters: {best_params}")

# Train the final model with the best hyperparameters
final_model = create_model(
    best_params['lr'], 
    best_params['momentum'], 
    best_params['num_kernels_1'], 
    best_params['num_kernels_2'], 
    best_params['reg_param']
)

final_model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1) 