import os
import re
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

matplotlib.use('macOSX')


# TODO: test this (locally!)
def plot_results(results_folder_path):
    models = os.listdir(results_folder_path)

    pattern = r"^(not-pretrained-)?(trialCV|subjCV)(-partition\d+)?(-.*)?$"
    models = [s for s in models if re.match(pattern, s)]

    models.sort()

    model_groups = defaultdict(list)

    for m in models:
        # Replace any single digit with a placeholder (e.g., '#')
        template = re.sub(r'(?<=partition)\d', '#', m)  # Replace only the first digit occurrence
        model_groups[template].append(m)

    for group_name, mg in model_groups.items():
        for fn in ['time_dependent_training_metrics',
                   'time_dependent_test_metrics']:  # later add 'scz', 'hc'
            dfs = [pd.read_csv(os.path.join(results_folder_path, m, f'{fn}.csv')) for m in mg]

            combined_df = pd.concat(dfs)  # Merge all data into one DataFrame

            # Create a seaborn lineplot, passing the matrix directly to seaborn
            plt.figure(figsize=(10, 6))  # Optional: Set the figure size

            # Create the lineplot, seaborn will automatically calculate confidence intervals
            sns.lineplot(data=combined_df, x=combined_df['chunk_position'] - 500, y='accuracy',
                         errorbar='ci', label='Accuracy')
            sns.despine()

            plt.axhline(y=0.5, xmin=0, color='orange', linestyle='dashdot', linewidth=1, label='Random Chance')
            plt.axvline(x=0, ymin=0, ymax=0.05, color='black', linewidth=1, label='Stimulus Onset')

            # Set plot labels and title
            plt.xlabel('Time (ms)')
            plt.ylabel('Accuracy')
            plt.legend()

            run = group_name.split('-')[-3]
            if group_name.startswith('not-pretrained'):
                run = 'not-pretrained-' + run
            plt.title(run)

            plt.savefig(os.path.join('../results/', 'plots', f'{run}_{fn[15:]}.png'))

            # TODO: joint plot
            # add columns indicating 'group' and 'training/test' in combined_df, append it to all_data
            # plot again, with hue (or whatever) set to 'group', one for training, one for test


def plot_umap(results_folder_path):
    filenames = os.listdir(results_folder_path)
    umap_files = [f for f in filenames if 'umap' in f]

    for uf in umap_files:
        df = pd.read_csv(os.path.join(results_folder_path, uf))

        for colouring in ['Subject ID', 'Label']:
            # Create a seaborn lineplot, passing the matrix directly to seaborn
            plt.figure(figsize=(10, 6))  # Optional: Set the figure size

            # Create the lineplot, seaborn will automatically calculate confidence intervals
            sns.scatterplot(data=df, x='x_embed', y='y_embed',
                            hue=df[colouring].map({df[colouring].min(): 0, df[colouring].max(): 1}))
            sns.despine()

            # Set plot labels and title
            plt.legend(title=colouring)

            c = 'subjID' if colouring == 'Subject ID' else 'label'
            plt.savefig(os.path.join('../results/', 'plots', f'{uf[:-4]}_{c}.png'))

        # Create a seaborn lineplot, passing the matrix directly to seaborn
        plt.figure(figsize=(10, 6))  # Optional: Set the figure size

        # Create the lineplot, seaborn will automatically calculate confidence intervals
        sns.scatterplot(data=df, x='x_embed', y='y_embed', hue='Label', style='Subject ID')
        sns.despine()

        # Set plot labels and title
        plt.legend()

        c = 'subjID' if colouring == 'Subject ID' else 'label'
        plt.savefig(os.path.join('../results/', 'plots', f'{uf[:-4]}_combined.png'))


if __name__ == '__main__':
    results_folder = '/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master ' \
                     'Thesis/Code/NeuroGPT/results/models/upstream'
    if not os.path.isdir('/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master '
                         'Thesis/Code/NeuroGPT/results/plots'):
        os.mkdir('/Users/joyet/Documents/Documents - Guillaume’s MacBook Pro/UniBasel/MSc_Data_Science/Master '
                 'Thesis/Code/NeuroGPT/results/plots')
    plot_results(results_folder_path=results_folder)
    plot_umap(results_folder_path=os.path.join(results_folder, 'test-0'))
