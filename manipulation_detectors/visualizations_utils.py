import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
sns.set_theme("paper", "white")

detectors_names = ['ConstantMetadata', 'FrameID', 'Timestamp', 'MSE', 'Histogram', 'OpticalFlow']
results_plot_transformation = {'ConstantMetadata' : lambda x: 1-x, #TODO: delete for new version after result changed
                                'MSE' : lambda x: max(np.log10(x), 0),
                                'Timestamp': lambda x: max(np.log10(x), 0)}

def plot_results(results_df: pd.DataFrame):
    num_graphs = len(results_df.columns)
    fig, axs = plt.subplots(num_graphs, 1, sharex=True)
    ax_id = 0
    for name, res in results_df.iteritems():
        if num_graphs > 1:
            ax = axs[ax_id]
        else:
            ax = axs
        res = res[1:]

        if name in results_plot_transformation:
            res = res.apply(results_plot_transformation[name])
        res.plot(ax=ax)
        ax.grid(True)
        ax.set_ylabel(name, rotation=0)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax_id +=1
    plt.show()

if __name__ == "__main__":
    pkl_path = r'C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\OUTPUT\faking_matlab_rec_3.pkl'
    pkl_path = Path(pkl_path)

    df = pd.read_pickle(pkl_path.as_posix())

    scores_df = df[[col for col in df if not col.startswith('time_')]]
    # times_df = df[[col for col in df if col.startswith('time_')]]

    plot_results(scores_df)


