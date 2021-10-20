import pandas as pd
import  os
import matplotlib.pyplot as plt

import time

def get_uniq_filename(filename):
    uniq_filename = time.strftime(f"%Y-%m-%dT%H:%S_{filename}")
    return uniq_filename

def save_plot(dataframe, plot_name, plot_dir):
    pd.DataFrame(dataframe.history).plot(figsize=(10, 10))
    plt.grid(True)

    uniq_filename = get_uniq_filename(plot_name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir,uniq_filename)

    plt.savefig(plot_path)
    plt.show()

