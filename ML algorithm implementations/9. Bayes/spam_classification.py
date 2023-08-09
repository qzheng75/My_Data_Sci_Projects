import numpy as np
import re


def text_parse(input_txt):
    tokens = re.split(r'\W+', input_txt)
    return [word.lower() for word in tokens if len(word) > 2]


def create_vocab_list(doc_list):
    vocab_set = set()
    for document in doc_list:
        vocab_set |= set(document)
    return list(vocab_set)


def train_test_split_indices(n_samples, test_ratio):
    test_samples = int(n_samples * test_ratio)
    all_indices = np.arange(n_samples)
    test_indices = np.random.choice(all_indices, size=test_samples, replace=False)
    train_indices = np.setdiff1d(all_indices, test_indices)
    return train_indices, test_indices


def word2vec_set(vocab_set, input_set):
    return_vector = [0] * len(vocab_set)
    for word in input_set:
        if word in vocab_set:
            return_vector[vocab_set.index(word)] += 1
    return return_vector


def train_naive_bayes(training_mat, training_class):
    num_samples = len(training_mat)
    num_words = len(training_mat[0])
    # p1: probability of spam
    p1 = np.sum(training_class) / float(num_samples)
    p0_num = np.ones((num_words,)) # Use ones in case division by 0
    p1_num = np.ones((num_words,))
    p0_denom = 2
    p1_denom = 2
    for i in range(num_samples):
        if training_class[i] == 1:
            p1_num += training_mat[i]
            p1_denom += np.sum(training_mat[i])
        else:
            p0_num += training_mat[i]
            p0_denom += np.sum(training_mat[i])
    p1_vec = np.log(p1_num / p1_denom) # Use np.log to prevent too small values
    p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, p1


def classify(word_test_mat, p0_vec, p1_vec, spam_prob):
    # log(P(h+|D)) = log(P(h+)) * log(P(D|h+)) = log(P(h+) + P(D|h+))
    # P(D|h+) = P(d_1|h+) * P(d_2|h+) * ... * P(d_n|h+)
    # log(P(h+) * P(D|h+)) = log(P(h+)) + log(sum(P(d_i|h+)))

    # word_test_mat * p_vec means the frequency of each word in spam/not spam emails
    p1 = np.log(spam_prob) + np.sum(word_test_mat * p1_vec)
    p0 = np.log(1 - spam_prob) + np.sum(word_test_mat * p0_vec)
    return 0 if p0 > p1 else 1


def spam():
    doc_list, class_list = [], []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        class_list.append(1) # 1 for spam email

        word_list = text_parse(open('email/normal/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        class_list.append(0)  # 0 for normal email

    vocab_list = create_vocab_list(doc_list)
    training_indices, test_indices = train_test_split_indices(50, 0.3)
    training_mat = []
    training_class = []
    for doc_idx in training_indices:
        training_mat.append(word2vec_set(vocab_list, doc_list[doc_idx]))
        training_class.append(class_list[doc_idx])
    p0_vec, p1_vec, p1 = train_naive_bayes(np.array(training_mat), np.array(training_class))
    error_count = 0
    for doc_idx in test_indices:
        word_test_mat = word2vec_set(vocab_list, doc_list[doc_idx])
        if classify(np.array(word_test_mat), p0_vec, p1_vec, p1) != class_list[doc_idx]:
            error_count += 1
    print(f"{error_count} samples are classified wrong in a test set with {len(test_indices)} samples.")


if __name__ == '__main__':
    spam()
