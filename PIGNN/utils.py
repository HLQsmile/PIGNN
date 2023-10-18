import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,f1_score
from emodel import MultiCEFocalLoss
def read_split_data(root: str, k ,k1):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    train=[]
    test=[]
    train_label=[]
    test_label=[]
    con=[]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        assert k > 1
        avg = len(images) // k
        if len(images)>500:
            for i, row in enumerate(images):
                if (i // avg) == k1:
                    test.append(row)
                    test_label.append(image_class)
                else:
                    train.append(row)
                    train_label.append(image_class)
        else:
            for i, row in enumerate(images):
                if (i // avg) == k1:
                    test.append(row)
                    test_label.append(image_class)
                else:
                    train.extend([row for i in range(5)])
                    train_label.extend([image_class for i in range(5)])
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train)))
    print("{} images for validation.".format(len(test)))
    con=Counter(train_label)
    con=list(con.values())
    return train,train_label,test,test_label,class_indices

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def test2(model, data_loader, device,predicts=False):

    model.eval()

    y_pred =[]
    y_true=[]
    y_output=[]
    for step, data in enumerate(data_loader):
        images, labels = data
        output = model(images.to(device))
        pred = output.max(dim=1)[1].cpu().data.numpy()
        y = labels.to(device).cpu().data.numpy()
        y_pred.extend(pred)
        y_true.extend(y)
        y_output.extend(output.cpu().data.numpy())

    acc = precision_score(y_true,y_pred,average='micro')
    f1= f1_score(y_true, y_pred, average='macro')
    if predicts:
          return acc,f1,y_true,np.array(y_pred),y_output
    else:
        return acc,f1







