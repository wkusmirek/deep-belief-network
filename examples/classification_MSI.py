import numpy as np
import winsound
import csv
np.random.seed(1337)  # for reproducibility
from sklearn.metrics.classification import accuracy_score
from dbn import SupervisedDBNClassification

classifier = SupervisedDBNClassification(hidden_layers_structure=[500, 500, 2000],
                                         learning_rate_rbm=0.1,
                                         learning_rate=0.1,
                                         n_epochs_rbm=20,
                                         n_iter_backprop=200,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)

def load_dataset(filename):
    print('Loading data from ' + filename + ' file...')
    import scipy.io
    mat = scipy.io.loadmat(filename)
    classnames = [item for sublist in mat['classnames'] for item in sublist]
    train_data = mat['train_data']
    test_data = mat['test_data']
    train_labels = [item for sublist in mat['train_labels'] for item in sublist]
    test_labels = [item for sublist in mat['test_labels'] for item in sublist]
    return classnames, train_data, train_labels, test_data, test_labels


def filter_dataset(classnames, train_data, train_labels, test_data, test_labels):
    print('Filtering dataset to 34-66 classes...')
    del_ids = list(range(1, 34, 1)) + list(range(67, 150, 1))
    classnames = classnames[34:66]
    train_labels, train_data = zip(*((id, train_data) for id, train_data in zip(train_labels, train_data) if id not in del_ids))
    test_labels, test_data = zip(*((id, test_data) for id, test_data in zip(test_labels, test_data) if id not in del_ids))
    return classnames, train_data, train_labels, test_data, test_labels


winsound.Beep(1000,440)
classnames_16, train_data_16, train_labels_16, test_data_16, test_labels_16 = load_dataset('data/caltech101_silhouettes_16_split1.mat')
classnames_28, train_data_28, train_labels_28, test_data_28, test_labels_28 = load_dataset('data/caltech101_silhouettes_28_split1.mat')
classnames_16, train_data_16, train_labels_16, test_data_16, test_labels_16 = filter_dataset(classnames_16, train_data_16, train_labels_16, test_data_16, test_labels_16)
classnames_28, train_data_28, train_labels_28, test_data_28, test_labels_28 = filter_dataset(classnames_28, train_data_28, train_labels_28, test_data_28, test_labels_28)

# classifier.fit(np.array(train_data_16).astype('int64'), train_labels_16)
# Y_pred = classifier.predict(np.array(test_data_16).astype('int64'))
# print('Done.\nAccuracy: %f' % accuracy_score(test_labels_16, Y_pred))
classifier.fit(np.array(train_data_28).astype('int64'), train_labels_28)
Y_pred = classifier.predict(np.array(test_data_28).astype('int64'))
print('Done.\nAccuracy: %f' % accuracy_score(test_labels_28, Y_pred))
data = np.zeros([len(classnames_28), len(classnames_28)])
for i in range(0,len(Y_pred)-1):
    Y_pred[i] = Y_pred[i]-34
# for i in range(0,len(test_labels_28)-1):
#     print(test_labels_28[i]-34)
#     test_labels_28[i] = test_labels_28[i]-34
for i in range(0, len(Y_pred)-1):
    data[test_labels_28[i]-34-1, Y_pred[i]-1] = data[test_labels_28[i]-34-1, Y_pred[i]-1]+1
print(data)

with open('wyniki.csv', mode='w') as wyniki:
    wyniki_writer = csv.writer(wyniki, dialect='excel')
    csvRow = ['']
    for i in range(0, len(classnames_28)-1):
        csvRow.append(classnames_28[i].astype(str))
    wyniki_writer.writerow(csvRow)
    for i in range(0, len(classnames_28)-1):
        csvRow = [classnames_28[i]]
        for j in range(0, len(classnames_28)-1):
            csvRow.append(data[i, j])
        wyniki_writer.writerow(csvRow)
