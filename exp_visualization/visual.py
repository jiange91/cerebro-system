import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os


def read_json_files(path_prefix):
    json_files = filter(lambda path: path.split(".")[-1] == "json", os.listdir(path_prefix))
    json_files = [os.path.join(path_prefix, path) for path in json_files]

    trials = []
    for json_file in json_files:
        with open(json_file, "r") as file:
            content = json.load(file)
            timestamp = []
            epoch = []
            accuracy = []
            for line in content:
                timestamp.append(line[0])
                epoch.append(line[1])
                accuracy.append(line[2])
            trials.append([timestamp, epoch, accuracy])

    return trials


def read_csv_files(path_prefix):
    csv_files = filter(lambda path: path.split(".")[-1] == "csv", os.listdir(path_prefix))
    csv_files = [os.path.join(path_prefix, path) for path in csv_files]

    trials = []
    for csv_file in csv_files:
        single_df = pd.read_csv(csv_file)
        timestamp = single_df["Wall time"].values.tolist()
        epoch = single_df["Step"].values.tolist()
        accuracy = single_df["Value"].values.tolist()
        trials.append([timestamp, epoch, accuracy])

    return trials


def read_single_text_file(path):
    train_trials, valid_trials = [], []
    with open(path, "r") as file:
        trial_dict = eval(file.readline())
        for trial_id in trial_dict:
            trial_line = trial_dict[trial_id]
            train_trials.append([list(range(1, len(trial_line["train_loss"]) + 1)), trial_line["train_accuracy"]])
            valid_trials.append([list(range(1, len(trial_line["train_loss"]) + 1)), trial_line["val_accuracy"]])
    return train_trials, valid_trials


def draw_two_plots(train_trials, valid_trials, plot1_path=None, plot2_path=None, save=False,
                   fix_y=True):
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    max_epoch = 0
    for trial in train_trials:
        if plot1_path == "plots/criteo_nas_greedy_accuracy_epoch.jpg":
            print(trial)
        plt.plot(trial[1], trial[2], linewidth=3, marker="o", alpha=0.8)
        if len(trial[1]) > max_epoch:
            max_epoch = len(trial[1])
    plt.ylabel("Train Accuracy", fontsize=20)
    plt.xlabel("Epoch", fontsize=20)
    plt.title("Accuracy on Training set", fontsize=20)
    plt.tick_params(labelsize=16)
    plt.xticks([epoch for epoch in range(1, max_epoch+1)], [epoch for epoch in range(1, max_epoch+1)], fontsize=18)

    plt.subplot(122)
    max_epoch = 0
    for trial in valid_trials:
        plt.plot(trial[1], trial[2], linewidth=3, marker="o", alpha=0.8)
        if len(trial[1]) > max_epoch:
            max_epoch = len(trial[1])
    plt.ylabel("Validation Accuracy", fontsize=20)
    plt.xlabel("Epoch", fontsize=20)
    plt.title("Accuracy on Validation set", fontsize=20)
    plt.tick_params(labelsize=16)
    plt.xticks([epoch for epoch in range(1, max_epoch+1)], [epoch for epoch in range(1, max_epoch+1)], fontsize=18)
    if save and plot1_path is not None and plot2_path is not None:
        plt.savefig(plot1_path, dpi=150)
    else:
        plt.show()

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    max_timestamp = 0
    min_timestamp = 99999999999999999999999999999999999999999
    for trial in train_trials:
        plt.plot(trial[0], trial[2], linewidth=3, marker="o", alpha=0.8)
        if max(trial[0]) > max_timestamp:
            max_timestamp = max(trial[0])
        if min(trial[0]) < min_timestamp:
            min_timestamp = min(trial[0])
    plt.ylabel("Train Accuracy", fontsize=20)
    plt.xlabel("Timestamp", fontsize=20)
    plt.title("Accuracy on Training set", fontsize=20)
    timestamp = [int(num) for num in np.linspace(min_timestamp, max_timestamp, num=5)]
    plt.xticks(timestamp, timestamp, fontsize=14)
    plt.tick_params(labelsize=16)

    plt.subplot(122)
    max_timestamp = 0
    min_timestamp = 99999999999999999999999999999999999999999
    for trial in valid_trials:
        plt.plot(trial[0], trial[2], linewidth=3, marker="o", alpha=0.7)
        if max(trial[0]) > max_timestamp:
            max_timestamp = max(trial[0])
        if min(trial[0]) < min_timestamp:
            min_timestamp = min(trial[0])
    plt.ylabel("Validation Accuracy", fontsize=20)
    plt.xlabel("Timestamp", fontsize=20)
    plt.title("Accuracy on Validation set", fontsize=20)
    timestamp = [int(num) for num in np.linspace(min_timestamp, max_timestamp, num=5)]
    plt.xticks(timestamp, timestamp, fontsize=14)
    plt.tick_params(labelsize=16)
    if save and plot1_path is not None and plot2_path is not None:
        plt.savefig(plot2_path, dpi=150)
    else:
        plt.show()


def draw_single_plots(train_trials, valid_trials, plot1_path=None, save=False):
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    max_epoch = 0
    for trial in train_trials:
        plt.plot(trial[0], trial[1], linewidth=3, marker="o", alpha=0.8)
        if len(trial[0]) > max_epoch:
            max_epoch = len(trial[0])
    plt.ylabel("Train Accuracy", fontsize=20)
    plt.xlabel("Epoch", fontsize=20)
    plt.title("Accuracy on Training set", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.xticks([epoch for epoch in range(1, max_epoch+1)], [epoch for epoch in range(1, max_epoch+1)], fontsize=18)

    plt.subplot(122)
    max_epoch = 0
    for trial in valid_trials:
        plt.plot(trial[0], trial[1], linewidth=3, marker="o", alpha=0.8)
        if len(trial[0]) > max_epoch:
            max_epoch = len(trial[0])
    plt.ylabel("Validation Accuracy", fontsize=20)
    plt.xlabel("Epoch", fontsize=20)
    plt.title("Accuracy on Validation set", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.xticks([epoch for epoch in range(1, max_epoch+1)], [epoch for epoch in range(1, max_epoch+1)], fontsize=18)
    if save and plot1_path is not None:
        plt.savefig(plot1_path, dpi=150)
    else:
        plt.show()


plt.style.use("ggplot")
mnist_fixarch_train_accuracy = read_json_files("mnist_fixarch_tb/json_data/train_accuracy")
mnist_fixarch_valid_accuracy = read_json_files("mnist_fixarch_tb/json_data/val_accuracy")
mnist_nas_random_train_accuracy = read_csv_files("mnist_random_nas_log/train")
mnist_nas_random_valid_accuracy = read_csv_files("mnist_random_nas_log/valid")
mnist_nas_greedy_train_accuracy, \
mnist_nas_greedy_valid_accuracy = read_single_text_file("mnist_greedy_nas_logs.txt")
criteo_nas_greedy_train_accuracy = read_json_files("criteo_tb_log/json_data/greedy/train_accuracy")
criteo_nas_greedy_valid_accuracy = read_json_files("criteo_tb_log/json_data/greedy/val_accuracy")
criteo_nas_random_train_accuracy = read_json_files("criteo_tb_log/json_data/random/train_accuracy")
criteo_nas_random_valid_accuracy = read_json_files("criteo_tb_log/json_data/random/val_accuracy")
cifar_nas_greedy_train_accuracy = read_json_files("cifar10_tb_log/json_files/train")
cifar_nas_greedy_valid_accuracy = read_json_files("cifar10_tb_log/json_files/valid")

draw_two_plots(mnist_fixarch_train_accuracy, mnist_fixarch_valid_accuracy,
               plot1_path="plots/mnist_fixarch_accuracy_epoch.jpg",
               plot2_path="plots/mnist_fixarch_accuracy_time.jpg",
               save=True)

draw_two_plots(mnist_nas_random_train_accuracy, mnist_nas_random_valid_accuracy,
               plot1_path="plots/mnist_nas_random_accuracy_epoch.jpg",
               plot2_path="plots/mnist_nas_random_accuracy_time.jpg",
               save=True)

criteo_nas_greedy_train_accuracy.pop(1)
criteo_nas_greedy_valid_accuracy.pop(1)
draw_two_plots(criteo_nas_greedy_train_accuracy, criteo_nas_greedy_valid_accuracy,
               plot1_path="plots/criteo_nas_greedy_accuracy_epoch.jpg",
               plot2_path="plots/criteo_nas_greedy_accuracy_time.jpg",
               save=True)

draw_two_plots(criteo_nas_random_train_accuracy, criteo_nas_random_valid_accuracy,
               plot1_path="plots/criteo_nas_random_accuracy_epoch.jpg",
               plot2_path="plots/criteo_nas_random_accuracy_time.jpg",
               save=True)

cifar_10_train = [[trial[0][::3], trial[1][::3], trial[2][::3]] for trial in cifar_nas_greedy_train_accuracy]
cifar_10_valid = [[trial[0][::3], trial[1][::3], trial[2][::3]] for trial in cifar_nas_greedy_valid_accuracy]

draw_single_plots(mnist_nas_greedy_valid_accuracy, mnist_nas_greedy_valid_accuracy,
                  plot1_path="plots/mnist_nas_greedy_accuracy_time.jpg",
                  save=True)

# CIFAR-10

plt.figure(figsize=(20, 10))
plt.subplot(121)
max_epoch = 0
for trial in cifar_10_train:
    plt.plot(trial[1], trial[2], linewidth=3, marker="o", alpha=0.8)
    if len(trial[1]) > max_epoch:
        max_epoch = len(trial[1])
plt.ylabel("Train Accuracy", fontsize=20)
plt.xlabel("Epoch", fontsize=20)
plt.title("Accuracy on Training set", fontsize=20)
plt.tick_params(labelsize=16)

plt.subplot(122)
max_epoch = 0
for trial in cifar_10_valid:
    plt.plot(trial[1], trial[2], linewidth=3, marker="o", alpha=0.8)
    if len(trial[1]) > max_epoch:
        max_epoch = len(trial[1])
plt.ylabel("Validation Accuracy", fontsize=20)
plt.xlabel("Epoch", fontsize=20)
plt.title("Accuracy on Validation set", fontsize=20)
plt.tick_params(labelsize=16)
plt.savefig("plots/cifar10_nas_greedy_accuracy_epoch.jpg", dpi=150)
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(121)
max_timestamp = 0
min_timestamp = 99999999999999999999999999999999999999999
for trial in cifar_10_train:
    plt.plot(trial[0], trial[2], linewidth=3, marker="o", alpha=0.8)
    if max(trial[0]) > max_timestamp:
        max_timestamp = max(trial[0])
    if min(trial[0]) < min_timestamp:
        min_timestamp = min(trial[0])
plt.ylabel("Train Accuracy", fontsize=20)
plt.xlabel("Timestamp", fontsize=20)
plt.title("Accuracy on Training set", fontsize=20)
timestamp = [int(num) for num in np.linspace(min_timestamp, max_timestamp, num=5)]
plt.xticks(timestamp, timestamp, fontsize=14)
plt.tick_params(labelsize=16)

plt.subplot(122)
max_timestamp = 0
min_timestamp = 99999999999999999999999999999999999999999
for trial in cifar_10_valid:
    plt.plot(trial[0], trial[2], linewidth=3, marker="o", alpha=0.7)
    if max(trial[0]) > max_timestamp:
        max_timestamp = max(trial[0])
    if min(trial[0]) < min_timestamp:
        min_timestamp = min(trial[0])
plt.ylabel("Validation Accuracy", fontsize=20)
plt.xlabel("Timestamp", fontsize=20)
plt.title("Accuracy on Validation set", fontsize=20)
timestamp = [int(num) for num in np.linspace(min_timestamp, max_timestamp, num=5)]
plt.xticks(timestamp, timestamp, fontsize=14)
plt.tick_params(labelsize=16)
plt.savefig("plots/cifar10_nas_greedy_accuracy_time.jpg", dpi=150)

