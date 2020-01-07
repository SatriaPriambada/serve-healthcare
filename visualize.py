import argparse
import json
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='Resnet1d_ray_serve.jsonl', help='start system with test config')
args = parser.parse_args()

def visualize():
    print("start visualizing {}".format(args.filename))
    fname = args.filename.split(".")[0]
    json_list = []
    df = pd.DataFrame(columns=['request_arrival','latency(ms)'])
    with open(args.filename) as f:
        for line in f:
            json_list.append(json.loads(line))
    #print(json_list)
    for i, item in enumerate(json_list):
        delta_t = item["end"] - item["start"]
        #print("iter {}, delta_t {}".format(i, delta_t))
        df = df.append({'iter': i,'delta_t':delta_t}, ignore_index=True)

    #print("sort list {}".format(sorted(df["delta_t"])))
    latency = sorted(df["delta_t"])
    ax2 = sns.kdeplot(latency ,cumulative=True)
    ax2.set(xlabel="latency(s)", ylabel="cdf(%)")
    fa = ax2.get_figure()
    fa.savefig('cdf_latency_{}.pdf'.format(fname), ext='pdf', bbox_inches='tight')

    plt.clf()
    ax = sns.lineplot(x="iter", y="delta_t", data=df)
    f = ax.get_figure()
    f.savefig('iter_to_delta_time_{}.pdf'.format(fname), ext='pdf', bbox_inches='tight')


if __name__ == '__main__':
    visualize()