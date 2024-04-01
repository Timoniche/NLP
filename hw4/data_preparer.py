import pickle
from random import shuffle

from datasets import load_dataset
from sentence_transformers import InputExample


def trainval_shuffled_data(from_file=True):
    datafile = 'trainval.pkl'

    if from_file:
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
        return data

    good, bad = load_trainval_samples()
    data = good + bad
    shuffle(data)

    with open(datafile, 'wb') as f:
        pickle.dump(data, f)

    return data


def trainval_examples():
    data = trainval_shuffled_data(from_file=True)

    trainval = []
    for question, document, cos in data:
        trainval.append(
            InputExample(texts=[question, document], label=cos)
        )

    return trainval


def load_trainval_samples():
    train, val, _ = _load_sberquad_dataset()

    questions = train['question'].tolist() + val['question'].tolist()
    contexts = train['context'].tolist() + val['context'].tolist()

    good_q_c_cos = list(zip(questions, contexts, [1.0] * len(questions)))

    bad_q_c_cos = []
    n = len(good_q_c_cos)
    delta = 30  # in sberquad dataset same contexts are close
    for i in range(n):
        cur_q, cur_c, _ = good_q_c_cos[i]
        next_q, next_c, _ = good_q_c_cos[(i + delta) % n]
        if next_c != cur_c:
            bad_q_c_cos.append((cur_q, next_c, 0.0))

    return good_q_c_cos, bad_q_c_cos


def _load_raw_sberquad_dataset(from_file=True):
    sberquad = 'sberquad.pkl'
    if from_file:
        with open(sberquad, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_dataset('sberquad')
        with open(sberquad, 'wb') as f:
            pickle.dump(df, f)

    return df


def _load_sberquad_dataset():
    df = _load_raw_sberquad_dataset()
    train = df['train'].to_pandas()
    validation = df['validation'].to_pandas()
    test = df['test'].to_pandas()

    return train, validation, test


def main():
    trainval_examples()


if __name__ == '__main__':
    main()
