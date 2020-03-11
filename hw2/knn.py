import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from utils.treebank import StanfordSentiment
import random
import matplotlib
matplotlib.use('agg')

dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
# inverse the dictionary
token_inverse = dict(zip(tokens.values(), tokens.keys()))


def k_closest(target_vector, wordvectors, k):
    # input:
    #   target_vector: a vector
    #   wordvectors: a matrix
    #   k: an integer
    # output:
    #   top_k_index: k indices of the matrix’s rows that are closest to the vector
    df = pd.DataFrame(wordvectors)
    score = [1 - distance.cosine(target_vector, df.iloc[i, :]) for i in range(len(df))]
    df['score'] = score
    df_sort = df.sort_values(by=['score'], ascending=False)
    target_index = df_sort.index.values[0]
    target_word = token_inverse[target_index]
    top_k_index = df_sort.index.values[1:k+1]
    top_k_word = [token_inverse[i] for i in top_k_index]
    print('The target word is :' + target_word)
    print('The top %d neighbors words are:' % k)
    print(top_k_word)
    print('\n\n')
    '''
    # Save each as a png
    all_words_index = df_sort.index.values[0:k+1]
    visualizeWords = [token_inverse[i] for i in all_words_index]
    visualizeIdx = all_words_index
    visualizeVecs = wordvectors[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])

    for i in range(len(visualizeWords)):
        plt.text(coord[i, 0], coord[i, 1], visualizeWords[i],
                 bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    plt.savefig('./example/knn_example_%s.png' % target_word)
    plt.show()
    plt.close()
    '''
    return top_k_index


# Show the result of knn
# Pick 10 words randomly from the training result
# Use knn to find the k near neighbor for the picked word
# Activate the the second part code in k_closest
# Plot the result each
def test_knn():
    wordvectors = np.load('saved_params_40000.npy')
    target_list = [random.randint(0, nWords) for _ in range(10)]  # Generate 10 example
    k = 10  # Set k = 10 in KNN
    result_index = []
    for target_index in target_list:
        target_vector = wordvectors[target_index, :]
        result = k_closest(target_vector, wordvectors, k)
        result_index .append(result)


if __name__ == "__main__":
    test_knn()
