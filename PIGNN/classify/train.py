import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from emodel import efficientnetv2_s as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, test2
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from itertools import cycle
import pickle
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
if __name__ == '__main__':
    pathjoin = os.path.join
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.05) #0.01 0.05
    parser.add_argument('--lrf', type=float, default=0.05)#0.01 0.05
    parser.add_argument('-out-dir', '--outdir', type=str, default=r"C:\Users\DELL\Desktop\PIGNN\classify\results")
    parser.add_argument('--data-path1', type=str,default=r"C:\Users\DELL\Desktop\PIGNN\data\train")
    parser.add_argument('--weights', type=str, default=r"C:\Users\DELL\Desktop\PIGNN\classify\pre_efficientnetv2-s.pth",help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False) #True为只训练最后一层
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_folder = args.outdir
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs"')
    # tb_writer = SummaryWriter()
    prob_file = pathjoin(save_folder, 'predicted_probabilities.txt')
    pred_file = pathjoin(save_folder, 'predicted_label.txt')
    true_file = pathjoin(save_folder, 'true_label.txt')

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
   
    img_size = {"s": [384, 384],  # train_size, val_size 384
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"
    acc=[]
    f1=[]
    # data_path=[args.data_path1,args.data_path2]
    # plt.figure()
    for i in range(0,5):
        train_images_path,train_images_label,val_images_path,val_images_label=read_split_data(args.data_path1,5,i)
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                    transforms.CenterCrop(img_size[num_model][1]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),}
       
        train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

        val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 10])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=True, #False
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)
    

        model = create_model(num_classes=args.num_classes).to(device)
        if args.weights != "":
            if os.path.exists(args.weights):
                weights_dict = torch.load(args.weights, map_location=device)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                      if model.state_dict()[k].numel() == v.numel()}
                print(model.load_state_dict(load_weights_dict, strict=False))
            else:
                raise FileNotFoundError("not found weights file: {}".format(args.weights))
    
        if args.freeze_layers:
            for name, para in model.named_parameters():
              
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
    
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4) #1E-4
        
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        max_metric=float(0)
        t_loss = []
        t_acc = []
        v_loss=[]
        v_acc=[]
        for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)
    
            scheduler.step()
            t_loss.append(train_loss)
            t_acc.append(train_acc)
            torch.save(model, "C:/Users/DELL/Desktop/PIGNN/classify/weights/model.pth")
        
        with open("C:/Users/DELL/Desktop/PIGNN/classify/results/train_loss.txt", 'w') as train_los:
            train_los.write(str(t_loss))
        with open("C:/Users/DELL/Desktop/PIGNN/classify/results/train_acc.txt", 'w') as train_ac:
            train_ac.write(str(t_acc))
        
        test_acc,test_f1,y_true,y_pred,y_output = test2(model,val_loader,device,predicts=True)
        print('k:%.03f,F1: %.03f,Acc: %.03f'%(i,test_f1,test_acc))
        np.savetxt(prob_file, y_output,fmt='%s' )
        np.savetxt(pred_file, y_pred,fmt='%s' )
        np.savetxt(true_file, y_true,fmt='%s' )
        acc.append(test_acc)
        f1.append(test_f1)
        print('%d finished'%i)
    acc_mean=np.mean(acc)
    f1_mean=np.mean(f1)
    print('acc_mean:%.03f,f1_mean:%.03f'%(acc_mean,f1_mean))

    
    
    
    