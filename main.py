import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
from utils import f1_score
from sklearn.metrics import f1_score as sk_f1_score
from matplotlib import pyplot as plt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练set
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=128,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,  # 2e-4
                        help='l2 reg')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='False',
                        help='adaptive learning rate')

    args = parser.parse_args()
    args.dataset = 'DBLP'
    args.adaptive_lr = True

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    norm = args.norm
    adaptive_lr = args.adaptive_lr
    print(args)

    # dataset
    # 导入节点的特征  node_features.shape>>>(8994,1902)
    with open('data/' + args.dataset + '/node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)
    # 导入边的特征   len(edge)>>>4  四个矩阵构成
    with open('data/' + args.dataset + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)  # 多个异构图的邻接矩阵
    # 导入label  len(labels)>>>3
    with open('data/' + args.dataset + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)  # train; val; test
    num_nodes = edges[0].shape[0]  # num_nodes>>>8994

    # 将edges组合成矩阵  --->>>合成邻接矩阵A
    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.cuda.FloatTensor).unsqueeze(-1)  # 添加末尾一位
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.cuda.FloatTensor).unsqueeze(-1)], dim=-1)
    # A.shape>>>([8994,8994,4])
    first_A = A.permute(2, 0, 1)
    print('first_A:', first_A.shape)  # >>>([4,8994,8994])

    # 添加一个单位对角阵>>>保留矩阵本身的性质
    A = torch.cat([A, torch.eye(num_nodes).type(torch.cuda.FloatTensor).unsqueeze(-1)], dim=-1)
    # A.shape>>>([8994,8994,5])

    channels = len(edges)

    # 创建对角矩阵
    diag_matrix = torch.eye(num_nodes).type(torch.cuda.FloatTensor)

    for i in range(channels):
        # 判断每一个矩阵的位置元素是否有值
        nonzero_indices = torch.nonzero(first_A[i])
        if nonzero_indices.numel() > 0:  # 如果该矩阵上有非零元素
            # 把对角矩阵diag_matrix该位置上的元素值置为1
            for index in nonzero_indices:
                diag_matrix[index[0], index[1]] = 1

    first_adj = diag_matrix.to(device)
    print('first_adj:', first_adj.shape)

    # 创建元素值全为1的nxn矩阵
    # n = 8994
    #matrix_1 = torch.ones(8994, 8994)
    # print('matrix_1',matrix_1.shape)
    # print('matrix_1',matrix_1)

    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.cuda.LongTensor)  # 训练节点和labels
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.cuda.LongTensor)  # val
    valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.cuda.LongTensor)  # test
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.cuda.LongTensor)

    print('node_features:', node_features.shape)

    # num_classes >>> 3
    num_classes = torch.max(train_target).item() + 1
    final_f1, final_micro_f1 = [], []

    for l in range(1):
        model = GTN(num_edge=A.shape[- 1],  # edge类别的数量:4; 还有一个单位阵;所以A.shape>>>5  4+1=5
                    num_channels=num_channels,  # 1X1卷积输出的channel数：1
                    w_in=node_features.shape[1],  # 节点的特征取第一位 >>>([8894,1902])
                    w_out=node_dim,  # 隐藏节点的特征>>>64
                    num_class=num_classes,  # 输出类别数>>>3
                    norm=norm)

        if adaptive_lr == 'False':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.00001)
        else:
            optimizer = torch.optim.Adam([{'params': model.weight},
                                          {'params': model.linear1.parameters()},
                                          {'params': model.linear2.parameters()},
                                          {'params': model.pyGgat.parameters(), "lr": 0.002,"weight_decay":0.001},
                                          {'params': model.TransformerEncoder.parameters(), "lr": 0.0002,"weight_decay":0.001},
                                          #{'params': model.pygGCNII.parameters(), "lr": 0.02,"weight_decay":0.001}, #ACM=0.02 DBLP=0.05 IMDB=0.005
                                          #{'params': model.pygtansformer.parameters(), "lr": 0.0002,"weight_decay": 0.001},
                                          ], lr=0.02, weight_decay=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        loss = nn.CrossEntropyLoss()
        # 做CrossEntropy包含softmax，所以返回输出时不用做softmax  CrossEntropyLoss <==> LogSoftmax + NLLLoss
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        max_val_loss = 10000
        max_test_loss = 10000
        best_train_f1, best_micro_train_f1 = 0, 0
        best_val_f1, best_micro_val_f1 = 0, 0
        best_test_f1, best_micro_test_f1 = 0, 0
        max_val_macro_f1, max_val_micro_f1 = 0, 0
        max_test_macro_f1, max_test_micro_f1 = 0, 0

        for i in range(epochs):
            # 定义optimizer里面的参数
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.1:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ', i + 1)

            model.zero_grad()
            # 1.开始训练
            model.train()  # A:[8994, 8994, 5],5个edgeType; node_features;
            loss, y_train, Ws = model(A,  node_features, first_adj, train_node, train_target)

            train_f1 = torch.mean(
                f1_score(torch.argmax(y_train.detach(), dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            sk_train_f1 = sk_f1_score(train_target.detach().cpu(), np.argmax(y_train.detach().cpu(), axis=1),
                                      average='micro')
            print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1,
                                                                        sk_train_f1))
            # 2.反向传播
            loss.backward()
            # 3.更新
            optimizer.step()
            # 4.评估
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid, _ = model.forward(A,  node_features,first_adj,  valid_node, valid_target)

                val_f1 = torch.mean(
                    f1_score(torch.argmax(y_valid, dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                sk_val_f1 = sk_f1_score(valid_target.detach().cpu(), np.argmax(y_valid.detach().cpu(), axis=1),
                                        average='micro')
                print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1,
                                                                            sk_val_f1))

                test_loss, y_test, W = model.forward(A,  node_features,first_adj,  test_node, test_target)

                test_f1 = torch.mean(
                    f1_score(torch.argmax(y_test, dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                sk_test_f1 = sk_f1_score(test_target.detach().cpu(), np.argmax(y_test.detach().cpu(), axis=1),
                                         average='micro')
                print('Test - Loss: {}, Macro_F1: {}, Micro_F1:{} \n'.format(test_loss.detach().cpu().numpy(), test_f1,
                                                                             sk_test_f1))

            # if val_f1 > best_val_f1:
            #     best_val_loss = val_loss.detach().cpu().numpy()
            #     best_test_loss = test_loss.detach().cpu().numpy()
            #     best_train_loss = loss.detach().cpu().numpy()
            #     best_train_f1 = train_f1
            #     best_val_f1 = val_f1
            #     best_test_f1 = test_f1
            # if sk_val_f1 > best_micro_val_f1:
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                best_micro_train_f1 = sk_train_f1
                best_micro_val_f1 = sk_val_f1
                best_micro_test_f1 = sk_test_f1

            # 获取最大的val_macro_f1
            if val_f1 > max_val_macro_f1:
                max_val_loss = val_loss.detach().cpu().numpy()
                max_val_macro_f1 = val_f1
            # 获取最大的val_micro_f1
            if sk_val_f1 > max_val_micro_f1:
                max_val_loss = val_loss.detach().cpu().numpy()
                max_val_micro_f1 = sk_val_f1
            # 获取最大的test_macro_f1
            if test_f1 > max_test_macro_f1:
                max_test_loss = test_loss.detach().cpu().numpy()
                max_test_macro_f1 = test_f1
            # 获取最大的test_micro_f1
            if sk_test_f1 > max_test_micro_f1:
                max_test_loss = val_loss.detach().cpu().numpy()
                max_test_micro_f1 = sk_test_f1

        print('---------------Best Results--------------------')
        print('Train - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_train_f1,
                                                                                best_micro_train_f1))
        print('Valid - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_val_loss, best_val_f1,
                                                                                best_micro_val_f1))
        print('Test - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_test_f1,
                                                                               best_micro_test_f1))
        final_f1.append(best_test_f1)
        final_micro_f1.append(best_micro_test_f1)
    print('---------------MAX Results--------------------')
    print('Valid - Loss: {:.4f}, maxMacro_F1: {:.4f}, maxMicro_F1: {:.4f}'.format(max_val_loss, max_val_macro_f1,
                                                                                  max_val_micro_f1))
    print('Test - Loss: {:.4f}, maxMacro_F1: {:.4f}, maxMicro_F1: {:.4f}'.format(max_test_loss, max_test_macro_f1,
                                                                                 max_test_micro_f1))
