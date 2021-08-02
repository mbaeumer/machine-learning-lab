import data_handler as data
from tensorflow.keras.utils import to_categorical
#from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import boundary


X_train = data.X_train
X_validation = data.X_validation
Y_train = to_categorical(data.Y_train)
Y_validation = to_categorical(data.Y_validation)

model = Sequential()
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=30000, batch_size=25)

boundary.show(model, data.X_train, data.Y_train)

# results from ex 17:
# Epoch 5000/5000
# 1/12 [=>............................] - ETA: 0s - loss: 0.4201 - accuracy: 0.8400
# 12/12 [==============================] - 0s 3ms/step - loss: 0.3681 - accuracy: 0.8447 - val_loss: 0.3330 - val_accuracy: 0.8281
# results from ex 18:
# Epoch 5000/5000
# 1/12 [=>............................] - ETA: 0s - loss: 0.1581 - accuracy: 1.0000
# 12/12 [==============================] - 0s 3ms/step - loss: 0.1586 - accuracy: 0.9572 - val_loss: 0.2218 - val_accuracy: 0.9123

# results from ex 17, with 30000 epochs:
# Epoch 30000/30000
# 1/12 [=>............................] - ETA: 0s - loss: 0.1516 - accuracy: 0.9200
# 12/12 [==============================] - 0s 3ms/step - loss: 0.1373 - accuracy: 0.9414 - val_loss: 0.1747 - val_accuracy: 0.9228
# results from ex 18, 30000 epochs:
# Epoch 30000/30000
# 1/12 [=>............................] - ETA: 0s - loss: 5.2170e-04 - accuracy: 1.0000
# 12/12 [==============================] - 0s 3ms/step - loss: 0.0125 - accuracy: 0.9966 - val_loss: 0.4554 - val_accuracy: 0.9158





#print(X_train)
#print(data.Y_train)
#print(Y_train)