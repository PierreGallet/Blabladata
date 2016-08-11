from sklearn.cluster import KMeans
import numpy as np
# np.set_printoptions(suppress=True)
# np.set_printoptions(threshold='nan')

with open('./sfr/input/tfidf/sentences.npy', 'rb') as sentences_npy:
    sentences = np.array(np.load(sentences_npy))

model = KMeans(n_clusters=8, max_iter=300)
y_pred = model.fit_predict(sentences)
print y_pred, sentences

with open('./sfr/input/sentences.txt', 'r+') as sentences:
    sentences_txt = sentences.readlines()

# print np.where(y_pred == 5)[0]
for i in range(8):
    for k in list(np.where(y_pred == i)[0])[:10]:
        print i, sentences_txt[k]
