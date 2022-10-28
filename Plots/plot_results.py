import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import heatmap


def create_subplot_of_alpha_and_test(file, save_file, title):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        file_name = file
        usecols = ["sigma_values", "epsilon", "test_acc", "alpha_value"]
        result_df = pd.read_csv(file_name, usecols=usecols)

        sigma_values = np.unique(list(result_df["sigma_values"]))
        epsilon = np.unique(list(result_df["epsilon"]))
        test_acc = list(result_df["test_acc"])
        alpha_value = list(result_df["alpha_value"])

        test_acc = np.reshape(test_acc, (len(sigma_values), len(epsilon)))
        alpha_value = np.reshape(alpha_value, (len(sigma_values), len(epsilon)))

        sigma_values = [str(round(x, 2)) for x in sigma_values]
        epsilon = [str(round(x, 2)) for x in epsilon]

        new_result_df_test = pd.DataFrame(data=test_acc, index=sigma_values, columns=epsilon)
        new_result_df_alpha = pd.DataFrame(data=alpha_value, index=sigma_values, columns=epsilon)

        heatmap(new_result_df_test, ax=ax1, vmin=0.4, vmax=1, center=0.5, cmap="PiYG", annot=True, fmt=".2f",
                cbar_kws={"ticks": [0.4,0.5,0.6,0.7,0.8,0.9,1]}, annot_kws={"size": 8})
        heatmap(new_result_df_alpha, ax=ax2, vmin=0, vmax=2.5, center=1, cmap="PiYG", annot=True, fmt=".2f"
                , annot_kws={"size": 8})
        ax1.title.set_text("Test")
        ax1.set_ylabel("σ")
        ax1.set_xlabel("ε")
        ax1.collections[0].colorbar.set_label("Test Auc")

        ax2.title.set_text("Alpha")
        ax2.set_ylabel("σ")
        ax2.set_xlabel("ε")
        ax2.collections[0].colorbar.set_label("Alpha value")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()


if __name__ == "__main__":
        file = os.path.join("GVM_normalize_False_corr_True_toy_model_grid_search_05_09_2022_05_55_54.csv")
        save_file = "GVM_normalize_False_corr_True_toy_model_grid_search_05_09_2022_05_55_54"
        title = ""
        create_subplot_of_alpha_and_test(file, save_file, title)
