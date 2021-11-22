import matplotlib.pyplot as plt

with open("cerebro_test.txt", "r") as file:
    data = file.readlines()

data = eval(data[0])

train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

for model in data:
    train_loss.append(data[model]["train_loss"])
    train_accuracy.append(data[model]["train_accuracy"])
    val_loss.append(data[model]["val_loss"])
    val_accuracy.append(data[model]["val_accuracy"])

epochs = [1, 2, 3, 4, 5]
epoch_labels = ["1", "2", "3", "4", "5"]

fig = plt.figure(figsize=(30, 20))
fig.add_subplot(221)
for index, line in enumerate(train_loss):
    plt.plot(epochs, line, marker=".", alpha=0.7)
plt.title("Training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(epochs, epoch_labels)

fig.add_subplot(222)
for index, line in enumerate(train_accuracy):
    plt.plot(epochs, line, marker=".", alpha=0.7)
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(epochs, epoch_labels)

fig.add_subplot(223)
for index, line in enumerate(val_loss):
    plt.plot(epochs, line, marker=".", alpha=0.7)
plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(epochs, epoch_labels)

fig.add_subplot(224)
for index, line in enumerate(val_accuracy):
    plt.plot(epochs, line, marker=".", alpha=0.7)
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(epochs, epoch_labels)

plt.savefig("plot.png", dpi=150)

