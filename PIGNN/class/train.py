import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from emodel import efficientnetv2_s as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, test2

if __name__ == '__main__':
    pathjoin = os.path.join
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('-out', '--outdir', type=str, default=r"...\resultsfenlei")
    parser.add_argument('--data-path1', type=str,default=r"..\train")
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_folder = args.outdir
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # tb_writer = SummaryWriter()
    prob_file = pathjoin(save_folder, 'predicted_probabilities.txt')
    pred_file = pathjoin(save_folder, 'predicted_label.txt')
    true_file = pathjoin(save_folder, 'true_label.txt')

    img_size = {"s": [384, 384],  # train_size, val_size 384
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"
    acc=[]
    f1=[]
    weights_list = []
    for i in range(0,1):
        train_images_path, train_images_label, test_images_path, test_images_label ,classs= read_split_data(args.data_path1, 5,i)
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            "test": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                    transforms.CenterCrop(img_size[num_model][1]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),}

        train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

        test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 10])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=test_dataset.collate_fn)

        model = create_model(num_classes=args.num_classes).to(device)
        # Freeze weights or not
        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
    
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        max_metric=float(0)
        model1=model
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
                                                    epoch=epoch,)
    
            scheduler.step()
            t_loss.append(train_loss)
            t_acc.append(train_acc)
            if train_acc > max_metric:
                max_metric = train_acc
                model1=model
                torch.save(model, save_folder+"/model.pth")
        test_acc,test_f1,y_true,y_pred,y_output= test2(model1,test_loader,device,predicts=True)
        print('k:%.03f,F1: %.03f,Acc: %.03f'%(i,test_acc,test_f1))
        print(classs)
        acc.append(test_f1)
        f1.append(test_f1)
    plt.show()
    acc_mean=np.mean(acc)
    f1_mean=np.mean(f1)
    print('acc_mean:%.03f,f1_mean:%.03f'%(acc_mean,f1_mean))
    
