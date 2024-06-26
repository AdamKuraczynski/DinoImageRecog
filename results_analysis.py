import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv('results/results_best_parameters.csv')

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def plot_parameter(df, parameter, ax=None):
    if ax is None:
        plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x=parameter, y='Test Accuracy', ci=None, ax=ax)
    ax.set_title(f'Test Accuracy vs {parameter}')
    ax.set_xlabel(parameter)
    ax.set_ylabel('Test Accuracy')
    ax.grid(True)

fig, axes = plt.subplots(3, 3, figsize=(24, 18))

plot_parameter(df, 'Filters', ax=axes[0, 0])
plot_parameter(df, 'Kernel Size', ax=axes[0, 1])
plot_parameter(df, 'Activation', ax=axes[0, 2])
plot_parameter(df, 'Padding', ax=axes[1, 0])
plot_parameter(df, 'Pool Size', ax=axes[1, 1])
plot_parameter(df, 'Stride', ax=axes[1, 2])
plot_parameter(df, 'Kernel Initializer', ax=axes[2, 0])

sns.boxplot(data=df, x='Activation', y='Test Accuracy', ax=axes[2, 1])
axes[2, 1].set_title('Test Accuracy vs Activation')
axes[2, 1].set_xlabel('Activation')
axes[2, 1].set_ylabel('Test Accuracy')
axes[2, 1].grid(True)

axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig('results/combined_plots_best_parameters.png')
plt.show()

# Optimal configurations
optimal_configurations = {}
parameters = ['Filters', 'Kernel Size', 'Activation', 'Padding', 'Pool Size', 'Stride', 'Kernel Initializer']
for param in parameters:
    optimal_configurations[param] = df.loc[df.groupby(param)['Test Accuracy'].idxmax()][param].values[0]

print("\nOptimal configurations for each parameter:")
for param, value in optimal_configurations.items():
    print(f"{param}: {value}")

print("\nFeature Importance (Correlation with Test Accuracy):")

relevant_columns = ['Test Accuracy', 'Filters', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Train Time (s)']
numeric_df = df[relevant_columns]
correlation_matrix = numeric_df.corr()
correlation_with_target = correlation_matrix['Test Accuracy'].drop('Test Accuracy')
print(correlation_with_target)

overall_best_config = df.loc[df['Test Accuracy'].idxmax()][parameters].to_dict()
print("\nOverall best configuration:")
for param, value in overall_best_config.items():
    print(f"{param}: {value}")
