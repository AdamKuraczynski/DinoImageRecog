import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('results/results.csv')

if not os.path.exists('results'):
    os.makedirs('results')

def plot_parameter(df, parameter, ax=None):
    if ax is None:
        plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x=parameter, y='Train Accuracy', ci=None, ax=ax)
    ax.set_title(f'Train Accuracy vs {parameter}')
    ax.set_xlabel(parameter)
    ax.set_ylabel('Train Accuracy')
    ax.grid(True)
    
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

plot_parameter(df, 'Conv Layers', ax=axes[0, 0])

plot_parameter(df, 'Pool Layers', ax=axes[0, 1])

plot_parameter(df, 'Epochs', ax=axes[0, 2])

sns.boxplot(data=df, x='Optimizer', y='Train Accuracy', ax=axes[1, 0])
axes[1, 0].set_title('Train Accuracy vs Optimizer')
axes[1, 0].set_xlabel('Optimizer')
axes[1, 0].set_ylabel('Train Accuracy')
axes[1, 0].grid(True)

plot_parameter(df, 'Batch Size', ax=axes[1, 1])

axes[1, 2].axis('off')

plt.tight_layout()

plt.savefig('results/combined_plots.png')
plt.show()

optimal_configurations = {}
parameters = ['Conv Layers', 'Pool Layers', 'Epochs', 'Optimizer', 'Batch Size']
for param in parameters:
    optimal_configurations[param] = df.loc[df.groupby(param)['Train Accuracy'].idxmax()][param].values[0]

print("\nOptimal configurations for each parameter:")
for param, value in optimal_configurations.items():
    print(f"{param}: {value}")

print("\nFeature Importance (Correlation with Train Accuracy):")

relevant_columns = ['Train Accuracy', 'Conv Layers', 'Pool Layers', 'Epochs', 'Batch Size']
numeric_df = df[relevant_columns]
correlation_matrix = numeric_df.corr()
correlation_with_target = correlation_matrix['Train Accuracy'].drop('Train Accuracy')
print(correlation_with_target)

overall_best_config = df.loc[df['Train Accuracy'].idxmax()][parameters].to_dict()
print("\nOverall best configuration:")
for param, value in overall_best_config.items():
    print(f"{param}: {value}")