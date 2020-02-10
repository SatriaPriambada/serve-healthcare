from resnet1d.resnet1d import ResNet1D
import ensemble_profiler as profiler
from pathlib import Path
import ray.experimental.serve as serve
import ray
import pandas as pd 
import jsonlines

# ECG
n_channel = 1
base_filters = 128
kernel_size = 16
n_classes = 2
n_block = 8
model = ResNet1D(in_channels=n_channel,
                 base_filters=base_filters,
                 kernel_size=kernel_size,
                 stride=2,
                 n_block=n_block,
                 groups=base_filters,
                 n_classes=n_classes,
                 downsample_gap=max(n_block//8, 1),
                 increasefilter_gap=max(n_block//4, 1),
                 verbose=False)

all_tq_lat = {}
all_ts_lat = {}
for i in range(1,11):
    obs_w_30sec = i
    npatient = 1
    print("find all latency profile for {} patients".format(npatient))
    filename = "profile_results_{}patient_{}obs_w_30sec.jsonl".format(npatient, obs_w_30sec)
    file_path = Path(filename)
    constraint = {"gpu":1, "npatient":npatient}
    profile_lat_tq = []
    profile_lat_ts = []
    exp_count = 20
    for i in range(exp_count):
        tq, ts = profiler.profile_ensemble([model], file_path, fire_clients=True, with_data_collector=False,obs_w_30sec=obs_w_30sec)
        profile_lat_tq.append(tq)
        profile_lat_ts.append(ts)
        print("latency_95th_profile-{} profiled tq: {}s, ts: {}s".format(i, tq, ts))
    
    key_tq = "tq_obs_w_30sec_{}".format(obs_w_30sec)
    all_tq_lat.update({key_tq : profile_lat_tq})
    df_tq = pd.DataFrame.from_dict(all_tq_lat, orient='index').transpose()
    df_tq.to_csv("tq_1patient_obs_w_30sec_lat.csv",index=False)
    
    key_ts = "ts_obs_w_30sec_{}".format(obs_w_30sec)
    all_ts_lat.update({key_ts : profile_lat_ts})
    df_ts = pd.DataFrame.from_dict(all_ts_lat, orient='index').transpose()
    df_ts.to_csv("ts_1patient_obs_w_30sec_lat.csv",index=False)
