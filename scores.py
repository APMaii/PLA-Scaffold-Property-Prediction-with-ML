#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:24:09 2025

@author: apm
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    "models": [
        "Linear Regression", 
        "K nearest neighbour", 
        "Decision Tree", 
        "Random Forest", 
        "Support vector machine", 
        "Multi layer perceptron"
    ],
    "Pore size": [0.158652442, 0.268955162, 0.319800902, 0.43597526, 0.08023138, 0.128668836],
    "Porosity": [0.044395676, 0.085531105, 0.048091858, 0.051724124, 0.045244129, 0.043121911],
    "Modulus": [0.320663728, 0.318125737, 0.426592314, 0.456775472, 0.205187312, 0.221108271],
    "Strength": [0.203296345, 0.364799884, 0.276992076, 0.362580206, 0.194847856, 0.165679671],
    "Specific strength": [0.226808932, 0.254803866, 0.217396279, 0.29594383, 0.192131523, 0.194573189]
}

df = pd.DataFrame(data)

# Set style
sns.set(style="whitegrid", context="talk")

# Grouped bar chart
plt.figure(figsize=(12, 7))
df_melted = df.melt(id_vars="models", var_name="Variable", value_name="Value")

sns.barplot(
    data=df_melted,
    x="Variable", y="Value", hue="models",
    palette="tab10"
)

plt.title("Model Comparison Across Variables", fontsize=18, weight="bold")
plt.ylabel("MAPE Error", fontsize=18, weight="bold")

plt.xticks(rotation=30, ha="right")
plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# Radar / Spider chart
import numpy as np

# Variables
categories = list(df.columns[1:])
N = len(categories)

# Angles for radar
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Plot
plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)

# Make grid more clear
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# One axis per variable + labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight="bold")

# Draw y-labels
ax.set_rlabel_position(0)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["0.1", "0.2", "0.3", "0.4", "0.5"], color="gray", size=10)
plt.ylim(0, 0.5)

# Plot each model
colors = plt.cm.tab10.colors  # distinct colors
for i, row in df.iterrows():
    values = row.drop("models").tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row["models"], color=colors[i % 10])
    ax.fill(angles, values, alpha=0.15, color=colors[i % 10])

# Title and legend
plt.title("Radar Chart - Model Comparison", size=16, weight="bold", pad=30)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05), fontsize=10, frameon=False)

plt.show()
