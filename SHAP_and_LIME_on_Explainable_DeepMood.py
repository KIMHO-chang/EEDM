import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import pickle as pk
import random
from Explainable_DeepMood import explainable_deep_mood
from Traditional_DeepMood import traditional_deep_mood
import shap
import lime.lime_tabular

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_mode_data(mode, subject_id, params):
    data = {}
    filename = "mood_py3/%s_%d.pickle" % (mode, subject_id)
    df = pd.read_pickle(open(filename, "rb"))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.date
    if mode == 'alphanum':
        df['x'] = df['x'].apply(lambda x: x / 2276)
        df['y'] = df['y'].apply(lambda x: x / 959)
        df['dt'] = df['dt'].fillna(0).apply(lambda x: min(x, 5)).apply(lambda x: np.log(x + 1) / np.log(6))
        df['dr'] = df['dr'].apply(lambda x: min(max(50, x), 1000)).apply(lambda x: np.log(x - 49) / np.log(951))
        if params['level'] == 0:
            for i in df['session_number'].unique():
                data[i] = df[lambda x: x.session_number == i][['x', 'y', 'dt', 'dr']].values
        if params['level'] == 1:
            for i in df['timestamp'].unique():
                data[i] = df[lambda x: x.timestamp == i][['x', 'y', 'dt', 'dr']].values
    if mode == 'special':
        if params['level'] == 0:
            for i in df['session_number'].unique():
                data[i] = df[lambda x: x.session_number == i].drop(['timestamp', 'session_number'], 1).values
        if params['level'] == 1:
            for i in df['timestamp'].unique():
                data[i] = df[lambda x: x.timestamp == i].drop(['timestamp', 'session_number'], 1).values
    if mode == 'accel':
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].apply(lambda x: x / 39.2266)
        if params['level'] == 0:
            for i in df['session_number'].unique():
                data[i] = df[lambda x: x.session_number == i][['x', 'y', 'z']].values
        if params['level'] == 1:
            for i in df['timestamp'].unique():
                data[i] = df[lambda x: x.timestamp == i][['x', 'y', 'z']].values
    return data


def load_subject_data(modes, subject_id, params):
    level_name = ['session', 'day']
    filename = "mood_py3/%s_%d.pickle" % (level_name[params['level']], subject_id)
    if os.path.isfile(filename):
        return pd.read_pickle(open(filename, "rb"))
    all_data = {}
    selected_data = {}
    label = []
    timestamp = []
    for mode in modes:
        all_data[mode] = load_mode_data(mode, subject_id, params)
        selected_data[mode] = []
    ratingfile = "mood_py3/weekly_ratings_%d.pickle" % subject_id
    df_rating = pd.read_pickle(open(ratingfile, "rb"))
    datefile = "mood_py3/%s_%d.pickle" % (modes[0], subject_id)
    df_date = pd.read_pickle(open(datefile, "rb"))
    for i in all_data[modes[0]].keys():
        if all(i in all_data[mode].keys() and len(all_data[mode][i]) >= params['seq_min_len'] for mode in modes):
            for mode in modes:
                lists = np.ndarray.tolist(
                    np.transpose(all_data[mode][i]))
                pad_matrix = tf.keras.preprocessing.sequence.pad_sequences(lists, maxlen=params['seq_max_len'], dtype='float32')
                pad_matrix = np.transpose(pad_matrix)
                selected_data[mode].append(pad_matrix)
                if params['level'] == 0:
                    df_date['timestamp'] = pd.to_datetime(df_date['timestamp'])
                    rating_date = df_date[lambda x: x.session_number == i].iloc[0].timestamp.date()
                if params['level'] == 1:
                    rating_date = i
                ratings = df_rating[lambda x: x.date == rating_date.strftime('%Y-%m-%d')].iloc[0]
                label.append([ratings.sighd_17item, ratings.ymrs_total])
                timestamp.append(rating_date)
    pk.dump((selected_data, label, timestamp), open(filename, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return selected_data, label, timestamp


def convert_label(labels, params):
    if params['is_clf']:
        for i in range(len(labels)):
            labels[i] = labels[i][0]
            if labels[i] >= 8:
                labels[i] = 1
            else:
                labels[i] = 0
        return labels
    else:
        for i in range(len(labels)):
            labels[i] = labels[i][1]
        return np.array(labels)


def load_data(subject_ids, ratio, params):
    data = {}
    label = {}
    timestamp = {}
    n = 0
    for i in subject_ids:
        data[i], label[i], timestamp[i] = load_subject_data(modes, i, params)
        m = len(label[i])
        if ratio > 0:
            idx = range(int(m * ratio))
        else:
            idx = range(int(m * (ratio + 1)), m)
        for mode in modes:
            data[i][mode] = [data[i][mode][j] for j in idx]
        label[i] = [label[i][j] for j in idx]
        timestamp[i] = [timestamp[i][j] for j in idx]
        n += len(idx)
    idx = range(n)
    random.seed(0)
    idx = random.sample(idx, k=len(idx))
    datas = {}
    for mode in modes:
        datas[mode] = np.concatenate([np.asarray(data[i][mode]) for i in subject_ids])
        datas[mode] = np.asarray([datas[mode][i] for i in idx])
        np.asarray(data[i][mode])
    labels = np.concatenate([label[i] for i in subject_ids], axis=0)
    labels = [labels[i] for i in idx]
    labels = convert_label(labels, params)
    timestamps = np.concatenate([timestamp[i] for i in subject_ids], axis=0)
    timestamps = [timestamps[i] for i in idx]
    whose = np.concatenate([[i] * len(label[i]) for i in subject_ids], axis=0)
    whose = [whose[i] for i in idx]
    return datas, labels, timestamps, whose


def split_data(subject_ids, params):
    if params['test_subject'] == -1:
        train_data, train_label, _, _ = load_data(subject_ids, 0.8, params)
        test_data, test_label, _, _ = load_data(subject_ids, -0.2, params)
    else:
        train_subjects = subject_ids
        train_subjects.remove(params['test_subject'])
        train_data, train_label, _, _ = load_data(train_subjects, 1, params)
        test_data, test_label, _, _ = load_data([params['test_subject']], 1, params)
    return train_data, train_label, test_data, test_label


def acc(y_true, y_pred):
    return K.mean(K.equal(y_true > 0.5, y_pred > 0.5))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def iid(train_label, test_label, params):
    dict_users, all_idxs = {}, [i for i in range(len(train_label))]
    np.random.seed(0)
    for i in range(params['num_users']):
        dict_users[i] = set(
            np.random.choice(all_idxs, params['data'], replace=False))
        if params['data'] <= (11968 / params['num_users']):
            all_idxs = list(set(all_idxs) - dict_users[i])
    for i in range(params['num_users']):
        dict_users[i] = list(dict_users[i])
    num_items_test = int(3003 * params['data'] / (11968 / params['num_users']) / params['num_users'])
    dict_users_test, all_idxs_test = {}, [i for i in range(len(test_label))]
    for i in range(params['num_users']):
        dict_users_test[i] = set(
            np.random.choice(all_idxs_test, num_items_test, replace=False))
        if params['data'] <= (11968 / params['num_users']):
            all_idxs_test = list(set(all_idxs_test) - dict_users_test[i])
    for i in range(params['num_users']):
        dict_users_test[i] = list(dict_users_test[i])
    return dict_users, dict_users_test


def run_model(train_data, test_data, y_train, y_test, params):
    X_train = []
    X_test = []
    for mode in modes:
        X_train.append(train_data[mode])
        X_test.append(test_data[mode])
    model = explainable_deep_mood(modes, params)
    dict_users, dict_users_test = iid(y_train, y_test, params)
    X_train_merge = []
    y_train_merge = []
    for j in range(params['num_users']):
        X_train_fed, y_train_fed = [], []
        for k in range(3):
            a = X_train[k][dict_users[j][0]]
            for i in dict_users[j]:
                if i == dict_users[j][0]:
                    continue
                else:
                    a = np.concatenate((a, X_train[k][i]))
            if k == 0:
                a = a.reshape(params['data'], 100, 4)
            elif k == 1:
                a = a.reshape(params['data'], 100, 6)
            else:
                a = a.reshape(params['data'], 100, 3)
            X_train_fed.append(a)
        for i in dict_users[j]:
            y_train_fed.append(y_train[i])
        y_train_merge.extend(y_train_fed)
        if j == 0:
            X_train_merge.append(X_train_fed)
        else:
            X_train_merge[0][0] = np.concatenate((X_train_merge[0][0], X_train_fed[0]), axis=0)
            X_train_merge[0][1] = np.concatenate((X_train_merge[0][1], X_train_fed[1]), axis=0)
            X_train_merge[0][2] = np.concatenate((X_train_merge[0][2], X_train_fed[2]), axis=0)
    X_train_last = []
    for i in range(3):
        X_train_last.append(X_train_merge[0][i])
    y_train_merge = np.array(y_train_merge)
    objective = 'binary_crossentropy' if params['is_clf'] else 'mean_squared_error'
    metric = [acc] if params['is_clf'] else [rmse]
    model.compile(loss=objective, optimizer=optimizers.RMSprop(learning_rate=params['lr']), metrics=metric, run_eagerly=True)
    model.fit(X_train_last, y_train_merge, batch_size=params['batch_size'], verbose=2, epochs=params['n_epochs'], validation_split=0.2, validation_freq=1, )
    model.summary()
    y_score = model.predict(X_test, batch_size=params['batch_size'], verbose=0)
    y_pred = (np.ravel(y_score) > 0.5).astype('int32') if params['is_clf'] else np.ravel(y_score)
    return y_pred, model, X_train_last, X_test


def evaluate(y_test, y_pred, params):
    res = {}
    if params['is_clf']:
        res['accuracy'] = float(sum(y_test == y_pred)) / len(y_test)
        res['precision'] = float(sum(y_test & y_pred) + 1) / (sum(y_pred) + 1)
        res['recall'] = float(sum(y_test & y_pred) + 1) / (sum(y_test) + 1)
        res['f_score'] = 2.0 * res['precision'] * res['recall'] / (res['precision'] + res['recall'])
    else:
        res['rmse'] = np.sqrt(np.mean(np.square(y_test - y_pred)))
        res['mae'] = np.mean(np.abs(y_test - y_pred))
        res['explained_variance_score'] = 1 - np.square(np.std(y_test - y_pred)) / np.square(np.std(y_test))
        res['r2_score'] = 1 - np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test)))
    print(' '.join(["%s: %.4f" % (i, res[i]) for i in res]))
    return res


modes = ['alphanum', 'special', 'accel']
params = {'seq_min_len': 10,
          'seq_max_len': 100,
          'batch_size': 256,
          'lr': 0.001,
          'dropout': 0.1,
          'n_epochs': 400,
          'n_hidden': 8,
          'n_latent': 8,
          'n_classes': 1,
          'bias': 1,
          'is_clf': 1,
          'idx': 1,
          'test_subject': -1,
          'level': 0,
          'num_users': 8,
          'frac': 0.1,
          'data': 500,
          'round': 1,
          'includes_alphanum': 1,
          'includes_special': 1,
          'includes_accel': 1,
          'alphanum_feature_num': 4,
          'special_feature_num': 6,
          'accel_feature_num': 3,
          'flag': 1}
subject_ids = [7, 13, 15, 16, 17, 19, 20, 21, 24, 25, 27, 30, 31, 32, 33, 36, 37, 38, 39, 40]


train_data, y_train, test_data, y_test = split_data(subject_ids, params)
y_pred, model, X_train, X_test = run_model(train_data, test_data, y_train, y_test, params)
res = evaluate(y_test, y_pred, params)


def SHAP(model, X_train, X_test):
    g_al = model.get_layer(index=3)
    g_sp = model.get_layer(index=4)
    g_ac = model.get_layer(index=5)
    f = model.get_layer(index=7)
    x = [X_train[i]+X_test[i] for i in range(len(modes))]
    g_al_output = g_al.predict(x[0])
    g_sp_output = g_sp.predict(x[1])
    g_ac_output = g_ac.predict(x[2])
    f_input = np.concatenate((g_al_output, g_sp_output, g_ac_output), axis=1)
    explainer = shap.DeepExplainer(f, f_input)
    shap_values = explainer.shap_values(f_input)
    global_explanation = np.mean(np.absolute(shap_values[0]), axis=0)
    return global_explanation


def LIME(model, X_train, X_test):
    g_al = model.get_layer(index=3)
    g_sp = model.get_layer(index=4)
    g_ac = model.get_layer(index=5)
    f = model.get_layer(index=7)
    x = [X_train[i] + X_test[i] for i in range(len(modes))]
    g_al_output = g_al.predict(x[0])
    g_sp_output = g_sp.predict(x[1])
    g_ac_output = g_ac.predict(x[2])
    f_input = np.concatenate((g_al_output, g_sp_output, g_ac_output), axis=1)
    lime_values = []
    for k in range(len(f_input)):
        explainer = lime.lime_tabular.LimeTabularExplainer(f_input, feature_names=modes, class_names=['Depression'])
        exp = explainer.explain_instance(f_input[k], f.predict, num_features=3, labels=(0,))
        exp = exp.as_list(label=0)
        exp_value = [0.0, 0.0, 0.0]
        for i in exp:
            for j in range(len(modes)):
                if modes[j] in i[0]:
                    exp_value[j] = float(i[1])
        lime_values.append(exp_value)
    global_explanation = np.mean(np.absolute(lime_values), axis=0)
    return global_explanation


SHAP(model, X_train, X_test)
LIME(model, X_train, X_test)
