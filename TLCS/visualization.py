import matplotlib.pyplot as plt
import os
import pydot
import graphviz

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi  #像素为dpi

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        在session中生成agent的性能图，并将相关数据保存为txt
        """
        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})  # 设置更大的字体

        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
    