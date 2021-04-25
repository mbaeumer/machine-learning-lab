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
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=5000, batch_size=25)

boundary.show(model, data.X_train, data.Y_train)




#print(X_train)
#print(data.Y_train)
#print(Y_train)