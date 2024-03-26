import pickle

import pandas as pd
from datasets import load_dataset

ANSWER_PREFIX = 'ответь: '


def load_matreshka_fort5():
    train, valid = _parse()
    t5train = _df_to_t5_format(train)
    t5valid = _df_to_t5_format(valid)
    df_train = pd.DataFrame(t5train, columns=['source_text', 'target_text'])
    df_val = pd.DataFrame(t5valid, columns=['source_text', 'target_text'])

    df_train['source_text'] = ANSWER_PREFIX + df_train['source_text']
    df_val['source_text'] = ANSWER_PREFIX + df_val['source_text']

    return df_train, df_val


def _load_raw_matreshka_dataset(from_file=True):
    matreshka = 'matreshka.pkl'
    if from_file:
        with open(matreshka, 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_dataset("zjkarina/matreshka")
        with open(matreshka, 'wb') as f:
            pickle.dump(df, f)

    return df


def _df_to_t5_format(df):
    user_bot = []
    for id, row in df.iterrows():
        raw_role = row['role']
        raw_dialog = row['dialog']
        if raw_role is None or raw_dialog is None:
            continue

        assert len(raw_role) == len(raw_dialog)

        raw_role = [r.lower() for r in raw_role]
        if any(r not in ['user', 'bot'] for r in raw_role):
            continue

        role, dialog = _shrink_roles_and_dialogue(raw_role, raw_dialog)

        if len(role) <= 1:
            continue

        assert role[0] == 'user'
        assert role[-1] == 'bot'

        i = 0
        while i < len(role):
            user_bot.append((dialog[i], dialog[i + 1]))
            i += 2

    return user_bot


def _shrink_roles_and_dialogue(raw_role, raw_dialog):
    shrinked_role = []  # [user, bot, bot, user, user] -> [user, bot, user]
    shrinked_dialog = []
    prev_role = ''
    for i in range(len(raw_dialog)):
        if raw_role[i] != prev_role:
            shrinked_role.append(raw_role[i])
            shrinked_dialog.append(raw_dialog[i])
        else:
            shrinked_dialog[-1] += ' ' + raw_dialog[i]
        prev_role = raw_role[i]

    assert len(shrinked_dialog) == len(shrinked_role)

    shrinked_role, shrinked_dialog = _drop_first_bot_last_user_utterings(shrinked_role, shrinked_dialog)

    return shrinked_role, shrinked_dialog


# role: [user, bot, user] -> [user, bot]
# role: [bot, ...] -> [...]
def _drop_first_bot_last_user_utterings(role, dialog):
    if role[-1] == 'user':
        role, dialog = role[:-1], dialog[:-1]
    if role[0] == 'bot':
        role, dialog = role[1:], dialog[1:]

    return role, dialog


def _parse():
    df = _load_raw_matreshka_dataset(from_file=True)
    train = df['train'].to_pandas()
    valid = df['validation'].to_pandas()

    return train, valid


def main():
    train, val = load_matreshka_fort5()
    print(train, val)


if __name__ == '__main__':
    main()
