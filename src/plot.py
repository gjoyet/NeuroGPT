import os
import re
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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
                   'time_dependent_test_metrics']:  # later add 'scz', 'hc'

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

            # Create a seaborn lineplot, passing the matrix directly to seaborn
            plt.figure(figsize=(10, 6))  # Optional: Set the figure size

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

            if len(mg) > 1:
                run = group_name.split('-')[-3]
                if group_name.startswith('not-pretrained'):
                    run = 'not-pretrained-' + run
            else:
                run = group_name[:-2]

            plt.title(run)

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

            # Create a seaborn lineplot, passing the matrix directly to seaborn
            plt.figure(figsize=(10, 6))  # Optional: Set the figure size

            custom_palette = {df["Subject ID"].min(): sns.color_palette()[0],
                              df["Subject ID"].max(): sns.color_palette()[1]}
            sns.scatterplot(data=df, x='x_embed', y='y_embed', hue='Subject ID',
                            style='Label', alpha=0.75, palette=custom_palette)
            sns.despine()

            # Set plot labels and title
            plt.legend()

            plt.savefig(os.path.join(plot_dir, f'{file[:-4]}.png'))
            plt.close()


if __name__ == '__main__':
    results_folder = '../results/models/upstream'
    plots_folder = '../results/plots'
    if not os.path.isdir(plots_folder):
        os.mkdir(plots_folder)
    plot_results(results_folder_path=results_folder)
    plot_umap(results_folder_path=os.path.join(results_folder, 'umap-0', 'umap'))
