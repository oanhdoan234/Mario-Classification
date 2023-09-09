import matplotlib.pyplot as plt
import os

# Define the filenames of the three files
filenames = ['exp3_cnn_epoch25_lr0.001.txt', 'exp3_cnn_epoch25_lr0.0001.txt']

# Loop over the filenames and extract the test_err values from each file
for i, filename in enumerate(filenames):
    path = './output'
    dirr = os.path.join(path, filename)
    with open(dirr, 'r') as f:
        data = f.read().splitlines()
        lst = data[3].split(': ')[1][1:][:-1].split(", ")
        test_err = [float(val) for val in lst]
        plt.plot(test_err, label=f'LR = {filename.split("_")[3].replace("lr", "").split(".txt")[0]}')

# Set the labels and legend of the graph
plt.xlabel('Epoch')
plt.ylabel('Test Error')
plt.title('Exp 3: All eight classes \n Test Error vs num_epochs')
plt.legend()

# Show the graph
plt.show()
