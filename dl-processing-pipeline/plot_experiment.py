# -*- coding: utf-8 -*-
"""plot_experiment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BtNkwsUs0rsKYpnwbvSiSPHuDuwUXdLh
"""

from google.colab import drive
drive.mount('/content/drive')

import json
import matplotlib.pyplot as plt

# Load the JSON data
with open('/content/drive/MyDrive/Colab Notebooks/cs598_ccc/train_epoch_log_no_compression_nooffloading.json', 'r') as f:
    data_0 = json.load(f)
with open('/content/drive/MyDrive/Colab Notebooks/cs598_ccc/train_epoch_log_compression_nooffloading.json', 'r') as f:
    data = json.load(f)

# Extract metrics for plotting
epochs = [entry['epoch'] for entry in data]
accuracy_top1 = [entry['accuracy_top1'] for entry in data]
accuracy_top5 = [entry['accuracy_top5'] for entry in data]
loss = [entry['loss'] for entry in data]
epoch_time = [entry['batch_time'] for entry in data]
data_time = [entry['data_time'] for entry in data]

epochs_0 = [entry['epoch'] for entry in data_0]
accuracy_top1_0 = [entry['accuracy_top1'] for entry in data_0]
accuracy_top5_0 = [entry['accuracy_top5'] for entry in data_0]
loss_0 = [entry['loss'] for entry in data_0]
epoch_time_0 = [entry['batch_time'] for entry in data_0]
data_time_0 = [entry['data_time'] for entry in data_0]

# Plot Accuracy vs Epoch
plt.figure(figsize=(8, 5))
plt.plot(epochs_0, accuracy_top1_0, label='Accuracy Top1: no compression + no offload', marker='o', color='purple')
plt.plot(epochs_0, accuracy_top5_0, label='Accuracy Top5: no compression + no offload', marker='x', color='orange')
plt.plot(epochs, accuracy_top1, label='Accuracy Top1: compression + no offload', marker='o', color='blue')
plt.plot(epochs, accuracy_top5, label='Accuracy Top5: compression + no offload', marker='x', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss vs Epoch
plt.figure(figsize=(8, 5))
plt.plot(epochs_0, loss_0, label='Loss: no compression + no offload', marker='s', color='skyblue')
plt.plot(epochs, loss, label='Loss: compression + no offload', marker='s', color='salmon')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot Average Batch Time vs. Epoch
plt.figure(figsize=(5, 3))
plt.plot(epochs, epoch_time, label='Batch Time', marker='o', color='b')
plt.plot(epochs, data_time, label='Data Time', marker='o', color='g')
plt.xlabel('Epoch')
plt.ylabel('Average Time (seconds)')
plt.title('Time vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot Average Data Time vs. Epoch
plt.figure(figsize=(5, 3))
plt.plot(epochs, data_time, label='Data Time', marker='o', color='g')
plt.xlabel('Epoch')
plt.ylabel('Data Time (seconds)')
plt.title('Data Time vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Plot Average Batch Time and Data Time for Two Settings vs. Epoch
plt.figure(figsize=(8, 5))

# Setting 1 (data_0)
plt.plot(epochs_0, epoch_time_0, label='Batch Time: no compression + no offload', marker='o', color='purple')
plt.plot(epochs_0, data_time_0, label='Data Time: no compression + no offload', marker='x', color='orange')


# Setting 2 (main data)
plt.plot(epochs, epoch_time, label='Batch Time: compression + no offload', marker='o', color='blue')
plt.plot(epochs, data_time, label='Data Time: compression + no offload', marker='x', color='green')


# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Average Time (seconds)')
plt.title('Average Batch Time and Data Time vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot Data Time for Two Settings vs. Epoch (Separate Plot)
plt.figure(figsize=(8, 5))

# Setting 1 (data_0)
plt.plot(epochs_0, data_time_0, label='Data Time: no compression + no offload', marker='o', color='orange')


# Setting 2 (main data)
plt.plot(epochs, data_time, label='Data Time: compression + no offload', marker='o', color='green')


# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Data Time (seconds)')
plt.title('Data Time vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt


# Calculate the averages
average_epoch_time = sum(epoch_time) / len(epoch_time)
average_data_time = sum(data_time) / len(data_time)


# Define colors for each bar
colors = ['skyblue', 'salmon']

# Plot the averages as a bar plot with different colors

plt.figure(figsize=(6, 4))
plt.bar(['Data Processing + Training', 'Data Processing'], [average_epoch_time, average_data_time], color=colors)
plt.ylabel('Time (seconds)')
plt.title('Average Time Per Epoch')
plt.yscale('log', base=2)
plt.show()

import matplotlib.pyplot as plt

# Assume these are the average times for two settings
# Setting 1
average_epoch_time_1 = sum(epoch_time_0) / len(epoch_time_0)
average_data_time_1 = sum(data_time_0) / len(data_time_0)

# Setting 2
average_epoch_time_2 = sum(epoch_time) / len(epoch_time)
average_data_time_2 = sum(data_time) / len(data_time)

# Define labels and colors
labels = ['Data Processing + Training', 'Data Processing']
colors = ['skyblue', 'salmon']  # Colors for Setting 1 and Setting 2

# Bar width and positions
bar_width = 0.35
x = range(len(labels))

# Plot the bar chart
plt.figure(figsize=(8, 6))
plt.bar(x, [average_epoch_time_1, average_data_time_1], width=bar_width, color=colors[0], label='no compression + no offload')
plt.bar([i + bar_width for i in x], [average_epoch_time_2, average_data_time_2], width=bar_width, color=colors[1], label='compression + no offload')

# Adding labels and legend
plt.xlabel('Process')
plt.ylabel('Time (seconds)')
plt.title('Average Time Per Epoch')
plt.yscale('log', base=2)  # Set y-axis to log2 scale
plt.xticks([i + bar_width / 2 for i in x], labels)  # Center the labels
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

data

