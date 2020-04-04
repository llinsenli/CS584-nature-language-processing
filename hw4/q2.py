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

# Receive the positive reviews
positive_reviews = review_text_extract(positive)
# Receive the negative reviews
negative_reviews = review_text_extract(negative)

# Define a function to build the dataframe object
# 1 represent positive, 0 represent negative
def q2_to_df(p,n):
    cate_array_positive = np.ones((len(p),))
    cate_array_negative = np.zeros((len(n),))
    df_positive = pd.DataFrame({'reviews': p, 'category':cate_array_positive})
    df_negative = pd.DataFrame({'reviews': n, 'category':cate_array_negative})
    df = pd.concat([df_positive,df_negative])
    return df

import numpy as np
import pandas as pd
# Show the dataframe
df2 = q2_to_df(positive_reviews,negative_reviews)

# Define a function to pre-processing the df
# including build the vocabulary, tokenize, word to sequence
# input:  df
# output: x_train, y_train, x_val, y_val,x_test, y_test, (vocab_size, maxlen)
import numpy as np
from keras.utils import to_categorical
def text_preprocessing(df):
    X = df.iloc[:,0]
    y = df.iloc[:,1]
    sentences = X.values
    y = y
    sentences_train,sentences_test,y_train,y_test = train_test_split(sentences, y, test_size=0.1, random_state=5)
    X_train, X_val, y_train, y_val = train_test_split(sentences_train, y_train, test_size=0.1, random_state=2)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_validation = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_validation = pad_sequences(X_validation, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    print('Shape of x_train: ' + str(X_train.shape))
    print('Shape of y_train: ' + str(y_train.shape))
    print('Shape of x_validation: ' + str(X_validation.shape))
    print('Shape of y_validation: ' + str(y_val.shape))
    print('Shape of x_test: ' + str(X_test.shape))
    print('Shape of y_test: ' + str(y_test.shape))
    return X_train,y_train, X_validation,y_val,X_test, y_test, (vocab_size, maxlen)

# This one_hot encoding function is to tramsform the y to one-hot vector
def one_hot(y):
    return to_categorical(y)

# Define a function to built the convoluted channel
def convolution(vocab_size, maxlen):
    inputs = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, 50)(inputs)
    # channel 1
    conv1 = Conv1D(filters=8, kernel_size=3,activation='relu')(embedding)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    #conv2 = Conv1D(filters=8, kernel_size=2,activation='relu')(embedding)
    #drop2 = Dropout(0.5)(conv2)
    #pool2 = MaxPooling1D(pool_size=2)(drop2)
    #flat2 = Flatten()(pool2)
    # channel 3
    conv3 = Conv1D(filters=16, kernel_size=4,activation='relu')(embedding)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1,flat3])
    # interpretation
    drop4 = Dropout(0.8)(merged)
    dense1 = Dense(10, activation='relu')(drop4)
    outputs = Dense(2, activation='softmax')(dense1)
    model = Model(inputs=[inputs], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    return model

# Build the model
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Activation, BatchNormalization,Flatten,Dropout,Embedding,Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras import optimizers
x_train, y_train, x_val, y_val,x_test, y_test, (vocab_size, maxlen) = text_preprocessing(df2)
y_train = one_hot(y_train)
y_val = one_hot(y_val)
y_test_c = one_hot(y_test)
model = convolution(vocab_size, maxlen)

# Train the model
history = model.fit([x_train], y_train, epochs=15,validation_data=([x_val],y_val), batch_size=16)

# Print out the model evalution
loss_and_acc = model.evaluate([x_test], y_test_c)
print('\n\nThe test accuracy is:')
print('loss = ' + str(loss_and_acc[0]))
print('accuracy = ' + str(loss_and_acc[1]))
# Report the recall and precision for each category on the test sets
from sklearn.metrics import classification_report
y_pred_NN = model.predict([x_test], batch_size=16, verbose=1)
y_pred_bool = np.argmax(y_pred_NN, axis=1)
print('Report the recall and precision for each category on the test sets:')
print(classification_report(y_test, y_pred_bool))

# Plot training loss and validation loss every epoches
# summarize history for loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss for task 2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training_loss', 'validation_loss'], loc='upper right')
plt.show()
