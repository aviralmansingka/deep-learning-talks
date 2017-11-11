from collections import Counter
import string

import numpy as np
from nltk.tokenize import word_tokenize
import simplejson as json

from nlp import build_model


def process_str(s):
    s = s.translate(None, string.punctuation)
    return word_tokenize(s.lower())

def read_dataset(sentences, labels):
    dataset = []
    with open(sentences) as sentences, open(labels) as labels:
        import ipdb; ipdb.set_trace()
        for s, l in zip(sentences, labels):
            try:
                words = process_str(s)
                dataset.append( (int(l), set(words)) )
            except ValueError:
                pass
    return dataset

def get_most_commons(dataset, skip=10, total=4000):
    my_list = []
    for item in dataset:
        my_list += list(item[1])

    counter = Counter(my_list)

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

def generate_vectors(dataset, common_words, ternary, bias_term):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    for item in dataset:
        vector = [0] * len(common_words)
        for word in item[1]:
            if word in d:
                vector[d[word]] += 1

        if bias_term:
            vector.append(1.0)

        vectors.append( (item[0], np.array(vector)) )

    return vectors

def random_sample(data, proportion):

    count = int(len(data) * proportion)
    np.random.shuffle(data)
    return data[:count]


def evaluate(preds, golds):
    tp, pp, cp = 0.0, 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    precision = tp / pp
    recall = tp / cp
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)


def build_dataset(sentences, labels, nb_words=500):
    with open(sentences) as f:
        data = f.readlines()
    with open(labels) as g:
        labels = [int(label) for label in g.read()[:-1].split("\n")]

    ternary = True

    train_data = [set(process_str(s)) for s in data]
    train_data = [(label, data) for label, data in zip(labels, train_data)]

    common_words = get_most_commons(train_data, total=nb_words)

    train_vectors = generate_vectors(train_data, common_words, ternary, False)

    X = np.array([e[1] for e in train_vectors])
    Y = np.array([e[0] for e in train_vectors]).reshape(-1, 1)

    data = np.append(X, Y, axis=1)
    np.random.shuffle(data)

    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    return X, Y, common_words


if __name__ == '__main__':

    batch_size = 128
    nb_classes = 2
    nb_epoch = 12
    nb_words = 4000

    X, Y, common_words = build_dataset('sentences.txt', 'labels.txt', nb_words=nb_words)

    model = build_model(nb_words)
    model.fit(X[:7000], Y[:7000], nb_epoch=nb_epoch, batch_size=batch_size)
    score = model.evaluate(X[7000:], Y[7000:])
    print('Loss: ', score[0], ' Accuracy: ', score[1])

    with open('parsed_chat.json') as f:
        chat_file = json.load(f)

    messages = [message['message'] for message in chat_file['threads'][0]['messages'] if message['sender'] == 'Abhinav Mansingka']
    message = [(0, ' '.join(messages))]
    import ipdb; ipdb.set_trace()
    message_vector = generate_vectors(message, common_words, False, False)
    X = np.array([message_vector[0][1]])
    print(model.predict(X))
