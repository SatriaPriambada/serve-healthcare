import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
import random
from tqdm import tqdm

from util import my_eval, get_accuracy_profile, get_latency_profile

def get_description(n_gpu, n_patients):
    """
    return V and c

    V: base_filters, n_block, accuracy, latency
    c: n_gpu, n_patients
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

def random_sample(n_model, B, n_samples=1000):
    """
    Input:
        n_model: number of models, n
        n_samples: 

    Output:
        B \in \{0,1\}^{n_samples \times n_model}
    """
    out = []
    i = 0
    while i < n_samples:
        # get a random probability of 1s and 0s
        pp = np.random.rand()
        # get random binary vector
        tmp = np.random.choice([0, 1], size=n_model, p=(pp,1-pp))
        # dedup
        for b in B:
            if np.array_equal(tmp, b):
                break
        out.append(tmp)
        i += 1
    return out

def get_obj(accuracy, latency, lamda, L):
    return accuracy + lamda * (L - latency)

def save_checkpoint(res):
    with open('res.pkl','wb') as fout:
        pickle.dump(res, fout)

if __name__ == "__main__":

    # --------------------- hyper parameters ---------------------
    L = 0.1 # maximum latency
    lamda = 10
    N1 = 10 # warm start
    N2 = 1000 # proxy
    N3 = 10 # profile
    epoches = 10

    # --------------------- initialization ---------------------
    V, c = get_description(n_gpu=4, n_patients=1)
    n_model = V.shape[0]
    opt_b = np.zeros(n_model)
    B = []
    Y_accuracy = []
    all_latency = []
    Y_latency = []
    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency, 'all_latency':all_latency}
    accuracy_predictor = RF()
    latency_predictor = RF()

    # --------------------- (1) warm start ---------------------
    B = random_sample(n_model=n_model, B=B, n_samples=N1)
    # profile
    for b in tqdm(B):
        Y_accuracy.append(get_accuracy_profile(V, b))
        tmp_latency = get_latency_profile(V, c, b)
        all_latency.append(tmp_latency)
        Y_latency.append(np.percentile(tmp_latency, 95))
        save_checkpoint(res)

    # --------------------- (2) choose B ---------------------
    for i_epoches in tqdm(range(epoches)):

        # fit proxy
        accuracy_predictor.fit(B, Y_accuracy)
        latency_predictor.fit(B, Y_latency)

        pred_accuracy = accuracy_predictor.predict(B)
        pred_latency = latency_predictor.predict(B)
        print(my_eval(Y_accuracy, pred_accuracy))
        print(my_eval(Y_latency, pred_latency))

        # search
        # random sample a large
        B_bar = random_sample(n_model=n_model, B=B, n_samples=N2)
        pred_accuracy = accuracy_predictor.predict(B_bar)
        pred_latency = latency_predictor.predict(B_bar)
        all_obj = []
        for i in range(len(B_bar)):
            all_obj.append(get_obj(pred_accuracy[i], pred_latency[i], lamda, L))
        top_idx = np.argsort(all_obj)[::-1][:N3]
        B_0 = list(np.array(B_bar)[top_idx])

        # profile
        for b in tqdm(B_0):
            # get_accuracy_profile
            Y_accuracy.append(get_accuracy_profile(V, b))
            # get_latency_profile
            tmp_latency = get_latency_profile(V, c, b)
            all_latency.append(tmp_latency)
            Y_latency.append(np.percentile(tmp_latency, 95))
            save_checkpoint(res)

        B = B + B_0
        print(np.array(B).shape)

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L))
    opt_idx = np.argmax(all_obj)
    print('found best b is: {}'.format(B[opt_idx]))


