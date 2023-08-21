from collections import defaultdict
import matplotlib.pyplot as plt
import os
import numpy as np


class RunningMean:
    def __init__(self):
        self.mean = 0.0
        self.n = 0
        self.means = []

    def add(self, value):
        self.mean = (float(value) + self.mean * self.n)/(self.n + 1)
        self.n += 1
        return self

    def reset(self):
        self.mean = 0.0
        self.n = 0

    def get_means(self):
        self.means += [self.mean]
        self.reset()
        return self.means


class LossTracker:
    def __init__(self, output_folder):
        self.tracks = defaultdict(RunningMean)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def update(self, d):
        for k, v in d.items():
            self.tracks[k].add(v.item())

    def plot(self, name_view):
        plt.figure(figsize=(12, 8))
        color = ['bo-', 'rD-', 'g^-']
        i = 0
        for key in self.tracks.keys():
            plot = self.tracks[key].get_means()
            # plot_np = np.array(plot)
            # max_idx = np.argmax(plot_np)
            # plt.plot(max_idx, plot_np[max_idx], 'ks')
            # show_max = '[' + str(max_idx) + ' ' + str(plot_np[max_idx]) + ']'
            # plt.annotate(show_max, xytext=(max_idx, plot_np[max_idx]), xy=(max_idx, plot_np[max_idx]))
            plt.plot(range(len(plot)), plot, color[i], label=key)
            i += 1

        plt.xlabel('steps', {'family': 'Times New Roman', 'size': 30})
        plt.ylabel('Loss', {'family': 'Times New Roman', 'size': 30})

        # plt.legend(loc=4, prop={'family': 'Times New Roman', 'size': 20})
        # plt.grid(True)
        plt.tight_layout()
        # plt.tick_params(labelsize=15)

        plt.savefig(os.path.join(self.output_folder, f'plot_{name_view}.png'))
        plt.close()
