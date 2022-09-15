import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, IEGM_DataSET
from models.model_1 import IEGMNet, IEGMNetXNOR

import logging
import time
import os

def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # Instantiating NN
    model_class = IEGMNetXNOR if args.xnor else IEGMNet
    if args.xnor:
        net = IEGMNetXNOR(in_bw=args.act_bw, out_bw=args.act_bw, weight_bw=args.weight_bw)
    else:
        net = IEGMNet()
    net.train()
    net = net.float().to(device)
    
    if args.resume:
        logging.info("Load state dict from", args.resume)
        state_dict = torch.load(args.resume)
        net.load_state_dict(state_dict, strict=False)
        for module in net.modules():
            if hasattr(module, 'weight_org'):
                module.weight_org.copy_(module.weight)

    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    logging.info("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    epoch_num = EPOCH

    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []

    logging.info("Start training")
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)

        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            running_loss += loss.item()
            i += 1

        logging.info('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))

        Train_loss.append(running_loss / i)
        Train_acc.append((accuracy / i).item())

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total =        i = 0.0
        running_loss_test = 0.0

        for data_test in testloader:
            net.eval()
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = net(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        logging.info('Test Acc: %.5f Test Loss: %.5f' % (correct / total, running_loss_test / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total).item())

    basename = os.path.join(log_dir, "IEGM_net")
    if args.xnor:
        torch.save(net, basename + '_xnor.pkl')
        torch.save(net.state_dict(), basename + '_state_dict_xnor.pkl')
        torch.onnx.export(net, torch.zeros((1, 3, 224, 224)), basename + '_xnor.onnx', verbose=True)
    else:
        torch.save(net, basename + '.pkl')
        torch.save(net.state_dict(), basename + '_state_dict.pkl')
        torch.onnx.export(net, torch.zeros((1, 3, 224, 224)), basename + '.onnx', verbose=True)

    file = open('./saved_models/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    logging.info('Finish training')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='/home/ravit/Documents/NanoCAD/TinyMLContest/data/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--xnor', action='store_true', help='train an xnor model instead of full precision')
    argparser.add_argument('--act-bw', type=int, default=4, help='activation bitwidth')
    argparser.add_argument('--weight-bw', type=int, default=4, help='weight bitwidth')
    argparser.add_argument('--resume', type=str, default=None, help='path to state_dict from which to resume training')
    argparser.add_argument('--log-path', type=str, default='/home/ravit/Documents/NanoCAD/TinyMLContest/tinyml_contest2022_demo_example/logs',
            help='directory in which to save training logs')

    args = argparser.parse_args()

    device = "cpu" #torch.device("cuda:" + str(args.cuda))

    log_dir = os.path.join(args.log_path, time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    filePath = os.path.join(log_dir, "train.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(filePath), logging.StreamHandler()])
    logging.info("device is -------------- {}".format(device))

    main()
