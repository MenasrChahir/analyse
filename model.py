import numpy as np
import pandas as pd
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
import matplotlib.pyplot as plt




data = pd.read_csv('/content/dataset.csv')


with open('/content/algerian_arabic_stopwords.txt', 'r') as f:
    stopwords = set(f.read().split())

from typing import List

def clean_text(text: str, algerian_arabic_stopwords: List[str]) -> str:
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if not w in stopwords]
    text = " ".join(words)
    return text


data['text'] = data['text'].apply(lambda x: clean_text(x, stopwords))


tokenizer = Tokenizer(num_words=50000, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, maxlen=250)

# Split data into training and testing sets
Y = pd.get_dummies(data['label']).values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced',
                                      classes=np.unique(y_train.argmax(axis=1)),
                                      y=y_train.argmax(axis=1))

# Model architecture
nb_lstm = 1
lstm_units = 16
lstm_dropout = 0.8
epochs = 10
dropout_rate = 0.5
batch_size = 100

model = Sequential()
model.add(Embedding(50000, 256, input_length=X.shape[1]))
for i in range(nb_lstm):
    model.add(LSTM(lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, return_sequences=True if i < nb_lstm-1 else False))
model.add(Dropout(dropout_rate))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Define class weights
class_weights = {0: 1 / len(y_train[y_train[:,0] == 1]), 1: 1 / len(y_train[y_train[:,1] == 1])}

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=epochs, batch_size=batch_size, callbacks=[early_stop],
                    class_weight=class_weights)

# Print validation loss and accuracy
# Evaluate model on train data
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print("Train loss: {:.3f}, Train accuracy: {:.3f}".format(train_loss, train_acc))

# Evaluate model on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}, Test accuracy: {:.3f}".format(test_loss, test_acc))

val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(val_loss, val_acc))

# Print the final loss and accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Save test data
np.savetxt("test_data.csv", X_test, delimiter=",")
np.savetxt("test_labels.csv", y_test, delimiter=",")

# Save train data
np.savetxt("train_data.csv", X_train, delimiter=",")
np.savetxt("train_labels.csv", y_train, delimiter=",")


# Save the tokenizer and model
model.save('FINALsentiment_modelss.h5')
import pickle

# Save the tokenizer
with open('FINALsentiment_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)


