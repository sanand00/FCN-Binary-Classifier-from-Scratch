import numpy as np
import pickle
import torch
from torchvision.transforms import v2

from fcn_classifier.train import train
from fcn_classifier.validate import validate


def main():
    
    # Data Loading

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
        
    train_batch = [unpickle("cifar10/data_batch_" + str(i)) for i in range(1,6)]
    test_batch = unpickle("cifar10/test_batch")
    for key in ["batch", "labels", "data", "filenames"]:
        for i in range(5):
            train_batch[i][key] = train_batch[i].pop(list(train_batch[i].keys())[0])
        test_batch[key] = test_batch.pop(list(test_batch.keys())[0])
        
    data_list = []

    for batch in train_batch:
        data_list.append(np.hstack((np.array(batch["labels"]).reshape(-1,1), batch["data"])))    

    data = np.vstack(data_list)
    data = data[(data[:,0] == 3) | (data[:,0] == 5), :]
    data[:, 0] = (data[:,0] == 3).astype(int)
    data = data.astype(float)
    data[:, 1:] = (2 * data[:, 1:] / 255) - 1

    test = np.hstack((np.array(test_batch["labels"]).reshape(-1,1), test_batch["data"]))
    test = test[(test[:,0] == 3) | (test[:,0] == 5), :]
    test[:, 0] = (test[:,0] == 3).astype(int)
    test = test.astype(float)
    test[:, 1:] = (2 * test[:, 1:] / 255) - 1


    # Data Augmentation

    transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(32, 4)
    ])

    data_aug = data
    for i in range(data.shape[0]):
        img_tensor = torch.reshape(torch.tensor(data[i, 1:]).double(), (3, 32, 32))
        data_aug[i, 1:] = transform(img_tensor).flatten().numpy()
        
    data_full = np.vstack((data,data_aug))
    np.random.shuffle(data_full)


    # Training

    X = np.transpose(data_full[:,1:])
    y = np.transpose(data_full[:,0]).reshape(1,-1)
    X_test = np.transpose(test[:,1:])
    y_test = np.transpose(test[:,0]).reshape(1,-1)

    W_dict = {
        1: np.random.normal(scale=0.1, size = (256,3072)),
        2: np.random.normal(scale=0.1, size = (64, 256)),
        3: np.random.normal(scale=0.1, size = (16,64)),
        4: np.random.normal(scale=0.1, size = (1, 16)),
    }

    b_dict = {
        1: np.random.normal(scale=0.1, size = (256,1)),
        2: np.random.normal(scale=0.1, size = (64,1)),
        3: np.random.normal(scale=0.1, size = (16,1)),
        4: np.random.normal(scale=0.1, size = (1,1))
    }

    params = {
        'L': 4,
        'alpha': 0.009,
        'batch_size': 1024,
        'max_iter': 100
    }


    train_results = []
    accuracy = []
    auc = []
    lam_list = (0, 0.00001, 0.0001, 0.001, 0.01, 0.1)


    for lam in lam_list:
        train_temp = train(X, y, W_dict, b_dict, lam = lam, **params)
        acc_temp, auc_temp  = validate(X_test, y_test, train_temp[0], train_temp[1], params['L'])
        train_results.append(train_temp)
        accuracy.append(acc_temp)
        auc.append(auc_temp)
        

    final = np.argmax(auc)
    W_final, b_final, _ = train_results[final]
    np.savez_compressed("trained_model/W", W_final, allow_pickle = True)
    np.savez_compressed("trained_model/b", b_final, allow_pickle = True)

if __name__ == "__main__":
    main()