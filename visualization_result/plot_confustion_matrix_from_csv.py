"""
    Save confusion matrix as .png from WandB generated .csv file.

    Usage: $python plot_confusion_matrix_from_csv.py PATH_TO_CSV_FILE PATH_TO_SAVE_PNG(optional)
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


ACTION_CLASSES = [
    'beam_tighten_unfolded',
    'unfolding_beam',
    'tighten_beam',
    'tighten_blade',
    'checking_blade',
    'checking_body_structure_firmness',
    'checking_chemicals_leak_in_hoses',
    'checking_cleannes_chemicals_tank',
    'filling_chemicals_tank',
    'checking_chemicals_leak_under_drone',
    'chemicals_tank_cap_interaction',
    'none'
]


def get_matrix_from_csv_file(path_to_csv: str,
                             normalize: bool = True) -> np.ndarray:
    df = pd.read_csv(path_to_csv)
    df = df.replace(to_replace=ACTION_CLASSES,
                    value=[idx for idx in range(len(ACTION_CLASSES))])  # class_name to label conversion
    df = df.filter(['Actual', 'Predicted', 'nPredictions'])  # drop unnecessary columns

    matrix = np.zeros((len(ACTION_CLASSES), len(ACTION_CLASSES)))

    for idx, (_true, _pred,  _n) in df.iterrows():
        matrix[_true, _pred] = _n

    if normalize:
        matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]

    return matrix


def plot_confusion_matrix(matrix: np.ndarray,
                          path_to_save: str) -> None:
    disp = ConfusionMatrixDisplay(matrix, display_labels=ACTION_CLASSES)
    disp.plot(cmap=plt.cm.Blues, colorbar=False)

    fig = disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(8)
    fig.set_dpi(170)
    disp.ax_.set_xticklabels(ACTION_CLASSES, rotation=25, ha='right', rotation_mode='anchor')

    fig.savefig(path_to_save, bbox_inches='tight')


if __name__ == "__main__":
    path_to_csv_file = sys.argv[1]
    csv_path, csv_filename = os.path.split(path_to_csv_file)
    confusion_matrix = get_matrix_from_csv_file(path_to_csv_file)

    if len(sys.argv) == 2:
        path_to_save_figure = os.path.join(csv_path, f"{csv_filename[:-4]}.png")
    else:
        path_to_save_figure = sys.argv[2]

    plot_confusion_matrix(confusion_matrix, path_to_save_figure)
