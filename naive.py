import csv
import math
import random
import statistics

def cal_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

dataset = []

# Reading tab-separated values
with open('lab5.csv') as csvfile:
    lines = csv.reader(csvfile, delimiter='\t')  # Specify tab delimiter
    for row in lines:
        try:
            dataset.append([float(attr) for attr in row])
        except ValueError:
            print(f"Skipping invalid row: {row}")  # Optional: handle bad data rows

dataset_size = len(dataset)
print("Size of dataset is:", dataset_size)

# Splitting the dataset
train_size = int(0.7 * dataset_size)
print("Training set size:", train_size)

X_train = []
X_test = dataset.copy()

training_indexes = random.sample(range(dataset_size), train_size)
for i in training_indexes:
    X_train.append(dataset[i])
    X_test.remove(dataset[i])

# Organizing training data by class
classes = {}
for sample in X_train:
    label = int(sample[-1])
    if label not in classes:
        classes[label] = []
    classes[label].append(sample)

# Calculating mean and std deviation for each class
summaries = {}
for classValue, training_data in classes.items():
    summary = [(statistics.mean(attr), statistics.stdev(attr)) for attr in zip(*training_data)]
    del summary[-1]  # Remove class label stats
    summaries[classValue] = summary

# Predicting on test data
X_prediction = []
for sample in X_test:
    probabilities = {}
    for classValue, classSummary in summaries.items():
        probabilities[classValue] = 1
        for index, attr in enumerate(classSummary):
            probabilities[classValue] *= cal_probability(sample[index], attr[0], attr[1])
    
    best_label, best_prob = None, -1
    for classValue, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = classValue
    X_prediction.append(best_label)

# Accuracy calculation
correct = 0
for index, sample in enumerate(X_test):
    if sample[-1] == X_prediction[index]:
        correct += 1

accuracy = (correct / float(len(X_test))) * 100
print("Accuracy:", accuracy)
