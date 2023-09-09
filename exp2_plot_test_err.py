import matplotlib.pyplot as plt
import os

# Define the filenames of the three files
pairs = ['Pair1_Birdo_Yoshi', 
'Pair2_Bowser_MiniBowser', 
'Pair3_Luigi_Mario', 
'Pair4_Peach_Rosalina']
epoch = ['epoch25', 'epoch10']
lrs = ['lr0.01', 'lr0.001', 'lr0.0001']

# filenames = ['cnn_epoch10_lr0.01.txt', 'cnn_epoch10_lr0.001.txt', 'cnn_epoch10_lr0.0001.txt']

def plot_errors_all_pairs(num_epoch, lr):
    filenames = []
    plt.figure()

    for pair in pairs:
        filenames.append('_'.join([pair,'cnn', num_epoch,lr]) + '.txt')

    for i, filename in enumerate(filenames):
        path = './output'
        dirr = os.path.join(path, filename)
        print(dirr)
        
        with open(dirr, 'r') as f:
            data = f.read().splitlines()
            lst = data[3].split(': ')[1][1:][:-1].split(", ")
            test_err = [float(val) for val in lst]
            plt.plot(test_err, label=' '.join(filename.split("_")[1:3]).replace(" ", " vs "))

    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.title('Exp 2: Test Error by Pair')
    plt.legend()

def plot_errors_by_pair(pair, num_epoch):
    filenames = []
    plt.figure()

    for l in lrs:
        filenames.append('_'.join([pair,'cnn', num_epoch,l]) + '.txt')

    for i, filename in enumerate(filenames):
        path = './output'
        dirr = os.path.join(path, filename)
        print(dirr)

        with open(dirr, 'r') as f:
            data = f.read().splitlines()
            lst = data[3].split(': ')[1][1:][:-1].split(", ")
            test_err = [float(val) for val in lst]
            plt.plot(test_err, label=f'LR = {filename.split("_")[5].replace("lr", "").split(".txt")[0]}')

        plt_title = 'Exp 2 - ' + ' '.join(filename.split("_")[1:3]).replace(" ", " vs ") + ": Test error vs. Epoch"
    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.title(plt_title)
    plt.legend()

plot_errors_all_pairs('epoch25','lr0.001')
plot_errors_by_pair(pairs[0], 'epoch10')
plot_errors_by_pair(pairs[1], 'epoch10')
plot_errors_by_pair(pairs[2], 'epoch10')
plot_errors_by_pair(pairs[3], 'epoch10')







#Pair1_Birdo_Yoshi_cnn_epoch25_lr0.001.txt

# # Loop over the filenames and extract the test_err values from each file
# for i, filename in enumerate(filenames):
#     path = './output'
#     dirr = os.path.join(path, filename)
#     with open(dirr, 'r') as f:
#         data = f.read().splitlines()
#         lst = data[3].split(': ')[1][1:][:-1].split(", ")
#         test_err = [float(val) for val in lst]
#         plt.plot(test_err, label=f'LR = {filename.split("_")[2].replace("lr", "").split(".txt")[0]}')

# Set the labels and legend of the graph


# Show the graph
plt.show()
