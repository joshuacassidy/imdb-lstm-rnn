from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.datasets import imdb
from keras.preprocessing import sequence

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

input_train = sequence.pad_sequences(x_train, maxlen=500)
input_test = sequence.pad_sequences(x_test, maxlen=500)

model = Sequential()

model.add(Embedding(10000, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['acc'])

model.fit(
    input_train, 
    y_train, 
    epochs=10,
    batch_size=128, 
    validation_split=0.2
)

print(model.evaluate(input_test, y_test))
