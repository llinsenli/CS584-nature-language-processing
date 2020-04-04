# Read the txt file and store them as list variable
# each element in the list is a row in the previous txt file
# the txt file should be in the same dictionary as /data/file.txt
with open('./a4-data/q1/28054-0.txt', 'r') as f:
    x1 = f.readlines()

with open('./a4-data/q1/pg1661.txt', 'r') as f:
    x2 = f.readlines()

with open('./a4-data/q1/pg31100.txt', 'r') as f:
    x3 = f.readlines()

import re
import numpy as np
import pandas as pd
# Data pre-processing function
# Input:
#     text: the document
#     category: the category of the document, we assign 0, 1, 2 to this variable
#     n: the length of the paragraph, use to control the sample size,have to tune
# Return:
#     tol_num: the total number of examples for each category
#     df: the Dataframe variable after assign each paragraphs a category
def prepare(text, category, n):
    # Divide each document into multiple paragraphs.
    s = ''.join(text)
    l = s.split('\n\n')
    l1 = [i.strip() for i in l if len(i) > n]
    # Remove punctuation, irrelevant symbols, urls, and numbers
    l2 = [re.sub('[^A-Za-z0-9]+', ' ', i).lower().strip() for i in l1]
    tol_num = len(l2)
    # Create the dataframe object
    cate_array = np.ones((tol_num,)) * category
    df = pd.DataFrame({'paragraph': l2, 'category':cate_array })
    print('The total number of examples for category ' + str(category)+ ' is: ' + str(tol_num))
    return df, tol_num

df1, num1 = prepare(x1, 0, 600)
df2, num2 = prepare(x2, 1, 200)
df3, num3 = prepare(x3, 2, 1000)
df = pd.concat([df1,df2,df3])
X = df['paragraph']
y = df['category']


# Now we transform our data to be used for neural network
# Including data split, tokenizer, word to sequence
# At last printout the shape of each part
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
sentences = X.values
y = y

sentences_train,sentences_test,y_train,y_test = train_test_split(sentences, y, test_size=0.25, random_state=3)
X_train, X_val, y_train, y_val = train_test_split(sentences_train, y_train, test_size=0.05, random_state=1)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_validation = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(sentences_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# maxlen = max([len(s.split()) for s in sentences])
maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_validation = pad_sequences(X_validation, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Use One hot encoding to deal with the test set
# Since test set contain 3 categories
# then [1,0,0] represent category 0
#      [0,1,0] represent category 1
#      [0,0,1] represent category 2
import numpy as np
from keras.utils import to_categorical
label_train = to_categorical(y_train)
label_validation = to_categorical(y_val)
label_test = to_categorical(y_test)
print('Shape of x_train: ' + str(X_train.shape))
print('Shape of y_train: ' + str(label_train.shape))
print('Shape of x_validation: ' + str(X_validation.shape))
print('Shape of y_validation: ' + str(label_validation.shape))
print('Shape of x_test: ' + str(X_test.shape))
print('Shape of y_test: ' + str(label_test.shape))


# Bulid the neural network
from keras.models import Model
from keras.layers import Dense, Activation, BatchNormalization,Flatten,Dropout,Embedding,Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras import optimizers
inputs = Input(shape=(maxlen,))
embedding = Embedding(vocab_size, 100)(inputs)
# channel 1
conv1 = Conv1D(filters=12, kernel_size=3,activation='relu')(embedding)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)
# channel 2
conv2 = Conv1D(filters=12, kernel_size=2,activation='relu')(embedding)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)
# channel 3
conv3 = Conv1D(filters=12, kernel_size=4,activation='relu')(embedding)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
#drop4 = Dropout(0.5)(merged)
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(3, activation='softmax')(dense1)
model = Model(inputs=[inputs], outputs=outputs)
# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize
print('The summary of the model:')
print(model.summary())


# Train the model
history = model.fit([X_train], label_train, epochs=10,validation_data=([X_validation], label_validation), batch_size=16)

# Print out the model evalution
loss_and_acc = model.evaluate([X_test], label_test)
print('\n\nThe test accuracy is:')
print('loss = ' + str(loss_and_acc[0]))
print('accuracy = ' + str(loss_and_acc[1]))
# Report the recall and precision for each category on the test sets
from sklearn.metrics import classification_report
y_pred_NN = model.predict([X_test], batch_size=16, verbose=1)
y_pred_bool = np.argmax(y_pred_NN, axis=1)
print('Report the recall and precision for each category on the test sets:')
print(classification_report(y_test, y_pred_bool))

# Plot training loss and validation loss every epoches
# summarize history for loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss for task1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training_loss', 'validation_loss'], loc='upper right')
plt.show()
