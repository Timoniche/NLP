import pickle

from datasets import load_dataset


def load_question_context_similarity():
    train, _, _ = _load_sberquad_dataset()

    questions = train['question'].tolist()
    contexts = train['context'].tolist()

    good_q_c_cos = list(zip(questions, contexts, [1.0] * len(questions)))

    bad_q_c_cos = []
    n = len(good_q_c_cos)
    delta = 30  # in sberquad dataset same contexts are close
    for i in range(n):
        cur_q, cur_c, _ = good_q_c_cos[i]
        next_q, next_c, _ = good_q_c_cos[(i + delta) % n]
        if next_c != cur_c:
            bad_q_c_cos.append((cur_q, next_c, 0.0))

    print('good train samples: ', len(good_q_c_cos))
    print('bad train samples: ', len(bad_q_c_cos))


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
    load_question_context_similarity()


if __name__ == '__main__':
    main()
