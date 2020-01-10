import argparse
import json
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', nargs='+', type=str, default='profile_results.jsonl', help='start system with test config')
args = parser.parse_args()

def visualize():
    print("start visualizing {} {}".format(type(args.filename),args.filename))
    list_files = args.filename
    plt.clf()
    labels = []
    for input_file in list_files:
        fname = input_file.split(".")[0]
        labels.append(fname)
        json_list = []
        df = pd.DataFrame(columns=['request_arrival','latency(ms)'])
        with open(input_file) as f:
            for line in f:
                json_list.append(json.loads(line))
        #print(json_list)
        for i, item in enumerate(json_list):
            latency = item["end"] - item["start"]
            df = df.append({'queue_id': i,'latency':latency}, ignore_index=True)

        #print("sort list {}".format(sorted(df["latency"])))
        latency = sorted(df["latency"])
        ax2 = sns.kdeplot(latency ,cumulative=True, label='test')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,labels=labels)
    ax2.set(xlabel="latency(s)", ylabel="cdf(%)")
    fa = ax2.get_figure()
    fa.savefig('cdf_latency.pdf', ext='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

    print('draw latency')
    plt.clf()
    for input_file in list_files:
        fname = input_file.split(".")[0]

        json_list = []
        df = pd.DataFrame(columns=['request_arrival','latency(ms)'])
        with open(input_file) as f:
            for line in f:
                json_list.append(json.loads(line))
        #print(json_list)
        for i, item in enumerate(json_list):
            latency = item["end"] - item["start"]
            df = df.append({'queue_id': i,'latency':latency}, ignore_index=True)

        #print("sort list {}".format(sorted(df["latency"])))
        latency = sorted(df["latency"])
        ax = sns.lineplot(x="queue_id", y="latency", data=df)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,labels=labels)
    ax.set(xlabel="latency(s)", ylabel="cdf(%)")
    f = ax.get_figure()
    f.savefig('queue_id_to_latency_time.pdf', ext='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    visualize()