import os
import re
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

matplotlib.use('macOSX')


def plot_results(results_folder_path):
    models = os.listdir(results_folder_path)

    pattern = r"^(not-pretrained-)?(trialCV|subjCV)(-partition\d+)?(-.*)?|^train-only(-.*)?$"
    models = [s for s in models if re.match(pattern, s)]

    models.sort()

    model_groups = defaultdict(list)
    for m in models:
        # Replace any single digit with a placeholder (e.g., '#')
        # Remove "not-pretrained-" if it exists
        normalized = re.sub(r"^not-pretrained-", "", m)
        template = re.sub(r'(?<=partition)\d', '#', normalized)  # Replace only the first digit occurrence
        model_groups[template].append(m)

    for group_name, mg in model_groups.items():
        for fn in ['time_dependent_training_metrics',
                   'time_dependent_test_metrics',
                   'time_dependent_test_large_metrics',
                   'time_dependent_test_large_metrics_only_hc',
                   'time_dependent_test_large_metrics_only_scz']:

            dfs = []
            for m in mg:
                csv = os.path.join(results_folder_path, m, f'{fn}.csv')
                if os.path.isfile(csv):
                    df = pd.read_csv(csv)
                    df['Pretrained'] = 'Not Pretrained' if 'not-pretrained' in m else 'Pretrained'
                    dfs.append(df)
            if len(dfs) == 0:
                continue

            combined_df = pd.concat(dfs)  # Merge all data into one DataFrame

            sns.set_context("paper", font_scale=1.5)

            # Create a seaborn lineplot, passing the matrix directly to seaborn
            plt.figure(figsize=(10, 5))  # Optional: Set the figure size

            sns.set_palette(sns.color_palette("deep"))

            # Create the lineplot, seaborn will automatically calculate confidence intervals
            sns.lineplot(data=combined_df, x=combined_df['chunk_position'] - 500, y='accuracy',
                         errorbar='ci', hue='Pretrained')
            sns.despine()

            plt.axhline(y=0.5, xmin=0, color='orange', linestyle='dashdot', linewidth=1, label='Random Chance')
            plt.axvline(x=0, ymin=0, ymax=0.05, color='black', linewidth=1, label='Stimulus Onset')

            # Set plot labels and title
            plt.xlabel('Time (ms)')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()

            if len(mg) > 1:
                run = group_name.split('-')[-3]
                if group_name.startswith('not-pretrained'):
                    run = 'not-pretrained-' + run
            else:
                run = group_name[:-2]

            # plt.title(run)

            plt.savefig(os.path.join('../results', 'plots', f'{run}_{fn[15:]}.png'))
            plt.close()


def plot_umap(results_folder_path):
    directories = os.listdir(results_folder_path)

    for direc in filter(lambda s: not s.startswith('.'), directories):
        plot_dir = os.path.join('../results', 'plots', direc)
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        for file in os.listdir(os.path.join(results_folder_path, direc)):
            df = pd.read_csv(os.path.join(results_folder_path, direc, file))

            sid_md, sid_det1, sid_det2 = compute_stats(df, 'Subject ID')
            lab_md, lab_det1, lab_det2 = compute_stats(df, 'Label')

            # Create a seaborn lineplot, passing the matrix directly to seaborn
            fig, ax = plt.subplots(figsize=(12, 8))
            custom_palette = {df["Subject ID"].min(): sns.color_palette()[0],
                              df["Subject ID"].max(): sns.color_palette()[1]}
            sns.scatterplot(data=df, x='x_embed', y='y_embed', hue='Subject ID',
                            style='Label', alpha=0.75, s=60,
                            palette=custom_palette, ax=ax)
            sns.despine()

            plt.legend()

            caption = (
                    r"$\mathbf{With\ respect\ to\ subjects:}$" + "\n" +
                    r"$\quad \| m_1 - m_0 \| = $" + rf"{sid_md}" + "\n" +
                    r"$\quad | \Sigma_{0} | = $" + rf"{sid_det1}" + "\n" +
                    r"$\quad | \Sigma_{1} | = $" + rf"{sid_det2}" + "\n\n" +
                    r"$\mathbf{With\ respect\ to\ labels:}$" + "\n" +
                    r"$\quad \| m_1 - m_0 \| = $" + rf"{lab_md}" + "\n" +
                    r"$\quad | \Sigma_{0} | = $" + rf"{lab_det1}" + "\n" +
                    r"$\quad | \Sigma_{1} | = $" + rf"{lab_det2}"
            )

            plt.subplots_adjust(right=0.7)
            fig.text(
                0.95, 0.5, caption, ha='right', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'),
                multialignment='left'
            )

            plt.savefig(os.path.join(plot_dir, f'{file[:-4]}.png'))
            plt.close()


def compute_stats(df, var):
    v1 = df[var].min()
    v2 = df[var].max()

    # Mean Diff
    var_mean = df.groupby(var)[['x_embed', 'y_embed']].mean()
    var_mean_diff = np.array(var_mean.loc[v2] - var_mean.loc[v1])
    var_mean_diff_magnitude = np.linalg.norm(var_mean_diff)

    # Within-Class variance
    var_cov = df.groupby(var)[['x_embed', 'y_embed']].cov()
    cov1 = np.array(var_cov.loc[v1])
    cov2 = np.array(var_cov.loc[v2])

    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)

    return round(var_mean_diff_magnitude, 3), round(det1, 3), round(det2, 3)


def combine_hc_and_scz_data(results_folder):
    for directory in os.listdir(results_folder):
        if directory == '.DS_Store':
            continue

        content = os.listdir(os.path.join(results_folder, directory))

        if 'time_dependent_test_large_metrics_only_scz.csv' in content:
            scz_csv = os.path.join(results_folder, directory, 'time_dependent_test_large_metrics_only_scz.csv')
            hc_csv = os.path.join(results_folder, directory, 'time_dependent_test_large_metrics_only_hc.csv')

            scz_df = pd.read_csv(scz_csv)
            hc_df = pd.read_csv(hc_csv)

            df = pd.DataFrame({'chunk_position': scz_df['chunk_position'],
                               'accuracy': scz_df['accuracy'] * scz_df['n_samples'] + hc_df['accuracy'] * hc_df['n_samples'],
                               'n_samples': scz_df['n_samples'] + hc_df['n_samples']})

            df['accuracy'] = df['accuracy'] / df['n_samples']

            df.to_csv(os.path.join(results_folder, directory, 'time_dependent_test_large_metrics.csv'), index=False)


if __name__ == '__main__':
    results_folder = '../results/models/upstream'
    plots_folder = '../results/plots'

    # combine_hc_and_scz_data(results_folder=results_folder)

    if not os.path.isdir(plots_folder):
        os.mkdir(plots_folder)

    plot_results(results_folder_path=results_folder)
    # plot_umap(results_folder_path=os.path.join(results_folder, 'umap-0', 'umap'))
