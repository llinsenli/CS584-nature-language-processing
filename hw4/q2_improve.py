# Using Pre-Trained GloVe Embedding
from keras.utils import to_categorical
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Dropout, Embedding, Input
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.layers.merge import concatenate
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.models import Model
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import numpy as np
import pandas as pd

# Read the txt file and store them as list variable
with open('./a4-data/q2/negative_review.txt', encoding="utf8", errors='ignore') as f:
    negative = [i.strip() for i in f.readlines()]
with open('./a4-data/q2/positive_review.txt', encoding="utf8", errors='ignore') as f:
    positive = [i.strip() for i in f.readlines()]

# Define a function to extract the review text from <review_text>


def review_text_extract(text_list):
    reviews = list()
    for i in range(len(text_list)):
        if text_list[i] == '<review_text>':
            k = 1
            while text_list[i+k] != '</review_text>':
                k += 1
            review = ''.join(text_list[i+1:i+k])
            reviews.append(review)
    return reviews

# Define a function to build the dataframe object
# 1 represent positive, 0 represent negative


def q2_to_df(p, n):
    cate_array_positive = np.ones((len(p),))
    cate_array_negative = np.zeros((len(n),))
    df_positive = pd.DataFrame({'reviews': p, 'category': cate_array_positive})
    df_negative = pd.DataFrame({'reviews': n, 'category': cate_array_negative})
    df = pd.concat([df_positive, df_negative])
    return df

# This one_hot encoding function is to tramsform the y to one-hot vector


def one_hot(y):
    return to_categorical(y)


positive_reviews = review_text_extract(positive)
negative_reviews = review_text_extract(negative)
df2 = q2_to_df(positive_reviews, negative_reviews)
docs = df2['reviews']
labels = array(df2['category'])

sentences = docs.values
y = labels

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.1, random_state=5)
maxlen = 100
t = Tokenizer()
t.fit_on_texts(sentences_train)

x_train = t.texts_to_sequences(sentences_train)
x_test = t.texts_to_sequences(sentences_test)

vocab_size = len(t.word_index) + 1

# pad documents to a max length of 100 words
max_length = 100
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
y_train = one_hot(y_train)
y_test_c = one_hot(y_test)

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


inputs = Input(shape=(maxlen,))
# channel 1
embedding1 = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(inputs)
conv1 = Conv1D(filters=6, kernel_size=3, activation='relu')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)
# channel 2
embedding2 = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(inputs)
conv2 = Conv1D(filters=6, kernel_size=2, activation='relu')(embedding2)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)
# channel 3
embedding3 = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(inputs)
conv3 = Conv1D(filters=6, kernel_size=4, activation='relu')(embedding3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
drop4 = Dropout(0.8)(merged)
dense1 = Dense(10, activation='relu')(drop4)
outputs = Dense(2, activation='softmax')(dense1)
model = Model(inputs=[inputs], outputs=outputs)
# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize
print(model.summary())


# fit the model
# I have save the model into a json file
# So for checking the homework you can just load it and not have to train it again
# model.fit(x_train, y_train, epochs=100, validation_split=0.2,verbose=1)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate the model
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
loss, accuracy = loaded_model.evaluate(x_test, y_test_c, verbose=0)
print('\n\nThe test accuracy is:')
print('Accuracy: %f' % (accuracy*100))

# Report the recall and precision for each category on the test sets
print('\n\nReport the recall and precision for each category on the test sets:')
y_pred_NN = loaded_model.predict([x_test], batch_size=16, verbose=0)
y_pred_bool = np.argmax(y_pred_NN, axis=1)
print(classification_report(y_test, y_pred_bool))
