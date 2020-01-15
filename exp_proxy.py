import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
from tqdm import tqdm

from util import my_eval, get_accuracy_profile, get_latency_profile

def get_field(n_gpu, n_patients):
    """
    return V and c
    """

    base_filters_list = [8, 16, 32, 64, 128]
    n_block_list = [2, 4, 8, 16]
    n_fields = 3
    n_model = len(base_filters_list) * len(n_block_list)
    V = []
    for base_filters in base_filters_list:
        for n_block in n_block_list:
            accuracy = np.random.rand()
            latency = 1e-4*np.random.rand()
            tmp = [base_filters, n_block, accuracy, latency]
            V.append(tmp)
    V = np.array(V)

    n_gpu = n_gpu
    n_patients = n_patients
    c = np.array([n_gpu, n_patients])

    return V, c

def random_sample(n_model, n_samples=1000):
    X_train = []
    i = 0
    while i < n_samples:
        # get a random probability of 1s and 0s
        pp = np.random.rand()
        # get random binary vector
        tmp = np.random.choice([0, 1], size=n_model, p=(pp,1-pp))
        X_train.append(tmp)
        i += 1
    return np.array(X_train)

if __name__ == "__main__":

    # get fields
    V, c = get_field(n_gpu=4, n_patients=100)
    n_model = V.shape[0]
    b = np.zeros(n_model)

    # random sample B
    B = random_sample(n_model=n_model, n_samples=1000)

    # profile
    Y_accuracy = []
    all_latency = []
    Y_latency = []
    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency, 'all_latency':all_latency}
    for b in tqdm(B):
        # get_accuracy_profile
        Y_accuracy.append(get_accuracy_profile(V, b))
        # get_latency_profile
        tmp_latency = get_latency_profile(V, c, b)
        all_latency.append(tmp_latency)
        Y_latency.append(np.percentile(tmp_latency, 95))
        with open('res.pkl','wb') as fout:
            pickle.dump(res, fout)

    # solve
    accuracy_predictor = RF()
    latency_predictor = RF()
    accuracy_predictor.fit(B, Y_accuracy)
    latency_predictor.fit(B, Y_latency)

    # train perf
    pred_accuracy = accuracy_predictor.predict(B)
    pred_latency = latency_predictor.predict(B)
    print(my_eval(Y_accuracy, pred_accuracy))
    print(my_eval(Y_latency, pred_latency))


