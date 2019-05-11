import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from my_util import plot_history,full_multiclass_report
# Load data
seed = 1000
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=seed)
print("X train shape: " ,x_train.shape,"Y train shape: ",y_train.shape )
print("X val shape: " ,x_val.shape,"Y val shape: ",y_val.shape )
print("X test shape: " ,x_val.shape,"Y test shape: ",y_val.shape )

# Create model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Criterion
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary model
model.summary()

# Train model
history = model.fit(x_train, y_train, epochs=5,batch_size=512,validation_data=(x_val,y_val))

# Evaluate model
loss,acc = model.evaluate(x_test, y_test)
print("Loss : %f    Accuracy: %f" %(loss,acc) )

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# if we want to save best model
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
# model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=0, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

# or this code
# checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# model.fit(X, y, epochs=15, validation_split=0.4, callbacks=[checkpoint], verbose=False)

# Restore the model's state,
# this requires a model with the same architecture.
# model.load_weights('./weights/my_model')

# Report history
plot_history(history)

# Report full report
num_of_class= 10
target_names = ['class 0','class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']
le = LabelEncoder()
encoded_labels = le.fit_transform(target_names)
y_val_one_hot = np_utils.to_categorical(y_val, num_of_class) # 1 hot coding
full_multiclass_report(model,
                       x_val,
                       y_val_one_hot,
                       le.inverse_transform(np.arange(10)))
