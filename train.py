import argparse
from lib2to3.pytree import convert
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, Normalize, Trim32, IEGM_DataSET

from models.model_1 import IEGMNet, IEGMNetXNOR
from models.network import FC_small, FC_large, CNN_medium, CNN_large, CNN_tiny

import logging
import time
import os

from compile_utils import compile_conv_block, compile_fc_block, convert_fc_act, convert_conv_act

torch.manual_seed(0)

def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epochs
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # Instantiating NN
    if args.xnor:
        net = IEGMNetXNOR(in_bw=args.act_bw, out_bw=args.act_bw, weight_bw=args.weight_bw)
    else:
        net = IEGMNet()
    net.train()
    net = net.float().to(device)
    
    if args.resume:
        logging.info("Load state dict from " + args.resume)
        state_dict = torch.load(args.resume)
        net.load_state_dict(state_dict, strict=False)
        for module in net.modules():
            if hasattr(module, 'weight_org'):
                module.weight_org.copy_(module.weight)

    if args.xnor:
        transform = transforms.Compose([ToTensor(), Normalize(), Trim32()])
    else:
        transform = transforms.Compose([ToTensor()])

    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transform)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transform)

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    logging.info("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    epoch_num = EPOCH

    if args.evaluate:
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
    elif args.train:
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

        net.eval()
        basename = os.path.join(log_dir, "IEGM_net")
        if args.xnor:
            torch.save(net, basename + '_xnor.pkl')
            torch.save(net.state_dict(), basename + '_state_dict_xnor.pkl')
            torch.onnx.export(net, torch.zeros((1, 1, 1248, 1)), basename + '_xnor.onnx', verbose=True)
        else:
            torch.save(net, basename + '.pkl')
            torch.save(net.state_dict(), basename + '_state_dict.pkl')
            torch.onnx.export(net, torch.zeros((1, 1, 1250, 1)), basename + '.onnx', verbose=True)
        net.train()

        logging.info('Finish training')
    
    if args.compile:
        net.eval()
        index = 0
        for data_test in testloader:
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)[index:index+1]
            labels_test = labels_test.to(device)[index:index+1]

        x = IEGM_test

        conv2_in = net.conv1(x)
        conv3_in = net.conv2(conv2_in)
        conv4_in = net.conv3(conv3_in)
        conv5_in = net.conv4(conv4_in)
        fc1_in = net.conv5(conv5_in).permute((0, 2, 1, 3)).reshape((conv5_in.shape[0], -1))
        fc2_in = net.fc1(fc1_in)
        y = net.fc2(fc2_in)
        y_ = net.forward(x)

        #compile_conv_block(net.conv1, x)

        """
        conv_layer = list(net.conv1.modules())[1]
        bn_layer = list(net.conv1.modules())[2]
        mu, sigma, gamma, beta = bn_layer.running_mean.detach().clone(), bn_layer.running_var.detach().clone(), bn_layer.weight.detach().clone(), bn_layer.bias.detach().clone()
        
        conv1_out = conv_layer(x) #conv

        # simulated output
        bn_th = (torch.tensor(bn_th_inf)/2).view((1, 32, 1, 1))
        #bn_th = ((-1*beta*torch.sqrt(sigma+1e-5) / torch.sqrt(torch.pow(gamma, 2) + 1e-5)) + mu).view((1, 32, 1, 1))
        conv1_out_bn_sim = (conv1_out.detach().clone() > bn_th).float() - 0.5

        # manual calculation output
        conv1_out_bn_man = ((conv1_out.detach().clone()-mu.view((1, 32, 1, 1)))/torch.sqrt(sigma.view((1, 32, 1, 1))+1e-5))*gamma.view((1, 32, 1, 1)) + beta.view((1, 32, 1, 1))

        # actual output
        conv1_out_bn_act = bn_layer(conv1_out.detach().clone()) #bn

        print(convert_conv_act(conv1_out_bn_sim)[:100])
        print(convert_conv_act(conv1_out_bn_man)[:100])
        print(convert_conv_act(conv1_out_bn_act)[:100])
        print(convert_conv_act(conv2_in)[:100])
        """

        #compile_conv_block(net.conv2, conv2_in)
        #compile_conv_block(net.conv3, conv3_in)
        #compile_conv_block(net.conv4, conv4_in)
        #compile_conv_block(net.conv5, conv5_in)
        #compile_fc_block(net.fc1, fc1_in)
        compile_fc_block(net.fc2, fc2_in, binarize_output=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, help='epoch number', default=2)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='/home/ravit/Documents/NanoCAD/TinyMLContest/data/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--xnor', action='store_true', help='train an xnor model instead of full precision')
    argparser.add_argument('--act-bw', type=int, default=1, help='activation bitwidth')
    argparser.add_argument('--weight-bw', type=int, default=1, help='weight bitwidth')
    argparser.add_argument('--resume', type=str, default=None, help='path to state_dict from which to resume training')
    argparser.add_argument('--log-path', type=str, default='/home/ravit/Documents/NanoCAD/TinyMLContest/tinyml_contest2022_demo_example/logs',
            help='directory in which to save training logs')
    argparser.add_argument('--evaluate', action="store_true", help='flag for whether to evaluate model')
    argparser.add_argument('--train', action="store_true", help='flag for whether to train model')
    argparser.add_argument('--compile', action="store_true", help="flag for whether to compile model.")

    args = argparser.parse_args()

    device = "cpu" #torch.device("cuda:" + str(args.cuda))

    if args.xnor:
        log_name = "_".join(["xnor", str(args.act_bw), str(args.weight_bw)])
    else:
        log_name = "_".join(["float"])
    log_name += "_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join(args.log_path, log_name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    filePath = os.path.join(log_dir, "train.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(filePath), logging.StreamHandler()])
    logging.info("device is -------------- {}".format(device))

    main()
