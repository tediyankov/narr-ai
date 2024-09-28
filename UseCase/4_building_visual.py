
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
guardian_corpus = pd.read_csv('/data/exet5975/thesis/NarrAI/expanded_guardian_corpus.csv')
narrative_labels = pd.read_csv('/data/exet5975/thesis/NarrAI/narrative_labels.csv')

# Convert the 'Date' column to datetime
guardian_corpus['Date'] = pd.to_datetime(guardian_corpus['Date'])

# Ensure all narrative columns are treated as numeric
for label in narrative_labels['Label']:
    guardian_corpus[label] = pd.to_numeric(guardian_corpus[label], errors='coerce')

# Check for any non-numeric values and print them
for label in narrative_labels['Label']:
    non_numeric = guardian_corpus[guardian_corpus[label].isna()][label]
    if not non_numeric.empty:
        print(f"Non-numeric values found in {label}: {non_numeric}")

# Resample the data by month and count the total number of articles per month
monthly_total_counts = guardian_corpus.resample('MS', on='Date').size()

# Resample the data by month and count the number of 1s for each narrative
monthly_counts = guardian_corpus.resample('MS', on='Date').sum(numeric_only=True)

# Ensure monthly_total_counts is a DataFrame for division
monthly_total_counts = monthly_total_counts.to_frame(name='Total')

# Calculate the percentage of articles for each narrative
monthly_percentage = monthly_counts.div(monthly_total_counts['Total'], axis=0) * 100

# Set the font size
plt.rcParams.update({'font.size': 14})

# Create a single plot
fig, ax = plt.subplots(figsize=(18, 6))
colors = sns.color_palette('Set2', len(narrative_labels['Label']))

# Plot the percentage of 1s for each narrative
for i, label in enumerate(narrative_labels['Label']):
    narrative = narrative_labels[narrative_labels['Label'] == label]['Narrative'].values[0]
    
    # Add a newline character in the middle of the title for the top-most graph
    if i == 0:
        split_index = len(narrative) // 2
        narrative = narrative[:split_index] + '\n' + narrative[split_index:]
    
    sns.lineplot(x=monthly_percentage.index, y=monthly_percentage[label], linewidth=2.5, color=colors[i], label=narrative)

# Set the title, x-label, and y-label
ax.set_title('Narrative Percentages Over Time')
ax.set_xlabel('Month')
ax.set_ylabel('Narrative Percentage')

# Adjust the layout and show the legend
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)

# Save the plot
plt.savefig('/data/exet5975/thesis/NarrAI/use_case/UseCase/narrative_percentage2.png', dpi=300, bbox_inches='tight')