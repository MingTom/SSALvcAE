'''
多视图数据的共享生成式隐特征学习研究
'''

import os
import torch
from torch.utils.data import DataLoader
import Datasets
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.compute_r1_gradient_penalty import compute_r1_gradient_penalty
from utils.tracker import LossTracker
from Block import *
from utils.DDC_func import *
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def config_init(dataloader):  # original_dim, n_centroid, batch,
    if dataloader == 'UCI2':
        return [240, 76], 10, 1000
    if dataloader == 'UCI6':
        return [240, 76, 216, 47, 64, 6], 10, 1000
    if dataloader == 'MSRC':
        return [24, 576, 512, 256, 254], 7, 150
    if dataloader == 'Yale':
        return[4096, 3304, 6750], 15, 83
    if dataloader == 'NUS5':
        return [65, 226, 145, 74, 129], 31, 5000
    if dataloader == 'Caltech101':
        return [512, 40, 254, 1984, 48, 928], 102, 1024
    if dataloader == 'ALOI':
        return [48, 40, 254, 1984, 512, 928], 1000, 2000


cnames = ['darkorange', 'forestgreen', 'blue', 'red', 'darkslategrey', 'darkorchid', 'gold']
threshold = 15000  # 避免输入维度过大，以截断保证全连接维度


def plot_embedding_2d(X, y, title=None):
    '''
    Plot an embedding X with the class label y colored by the domain d.
    '''
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=cnames[int(y[i])],
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    # if title is not None:
    #     plt.title(title)
    plt.savefig('.\\vision\\'+title)

class SSMALAE:
    def __init__(self, num_views, num_classes, input_dim, config, lambd=0.0051):
        self.num_views = num_views
        self.config = config
        self.num_classes = num_classes
        self.F_s = nn.ModuleList()
        for i in range(num_views):
            if input_dim[i] < num_classes:
                self.F_s.append(
                    MappingFromLatent(config['z_dim'], input_dim[i],
                                      int(input_dim[i]*self.config['common_coefficient'])).cuda().train())
            elif input_dim[i] > threshold:
                self.F_s.append(
                    MappingFromLatent(config['z_dim'], self.config['w_exp'],
                                      int(self.config['w_exp']*self.config['common_coefficient'])).cuda().train())
            else:
                self.F_s.append(
                    MappingFromLatent(config['z_dim'], input_dim[i]//2,
                                      int(input_dim[i]//2*self.config['common_coefficient'])).cuda().train())

        self.G_s = nn.ModuleList()
        for i in range(num_views):
            if input_dim[i] < num_classes:
                self.G_s.append(Generator(input_dim[i], input_dim[i]).cuda().train())
            elif input_dim[i] > threshold:
                self.G_s.append(Generator(self.config['w_exp'], input_dim[i]).cuda().train())
            else:
                self.G_s.append(Generator(input_dim[i]//2, input_dim[i]).cuda().train())

        self.E_s = nn.ModuleList()
        for i in range(num_views):
            if input_dim[i] < num_classes:
                self.E_s.append(Encoder(input_dim[i], input_dim[i],
                                        int(input_dim[i]*self.config['common_coefficient'])).cuda().train())
            elif input_dim[i] > threshold:
                self.E_s.append(Encoder(input_dim[i], self.config['w_exp'],
                                        int(self.config['w_exp']*self.config['common_coefficient'])).cuda().train())
            else:
                self.E_s.append(Encoder(input_dim[i], input_dim[i]//2,
                                        int(input_dim[i]//2*self.config['common_coefficient'])).cuda().train())

        self.projector_s = nn.ModuleList()
        for i in range(num_views):
            if input_dim[i] < num_classes:
                l = [int(input_dim[i]*self.config['common_coefficient']), 4*input_dim[i], config['present_dim']]
                self.projector_s.append(Project(layers_dim=l).cuda().train())
            elif input_dim[i] > threshold:
                l = [int(self.config['w_exp']*self.config['common_coefficient']), 1024, config['present_dim']]
                self.projector_s.append(Project(layers_dim=l).cuda().train())
            else:
                l = [int(input_dim[i]//2*self.config['common_coefficient']), 4*(input_dim[i]//2), config['present_dim']]
                self.projector_s.append(Project(layers_dim=l).cuda().train())

        self.D_s = nn.ModuleList()
        for i in range(num_views):
            if input_dim[i] < num_classes:
                self.D_s.append(Discriminator(input_dim[i], num_classes).cuda().train())
            elif input_dim[i] > threshold:
                self.D_s.append(Discriminator(self.config['w_exp'], num_classes).cuda().train())
            else:
                self.D_s.append(Discriminator(input_dim[i]//2, num_classes).cuda().train())

        weight = torch.ones(num_views, device='cuda', requires_grad=True)/num_views
        self.weight = torch.nn.Parameter(weight)
        self.CLU = Clulayer(common_dim=config['present_dim'], out_dim=num_classes).cuda().train()
        self.lambd = lambd
        self.criterion = nn.MSELoss().cuda()

        self.ED_optimizers = []
        self.FG_optimizers = []
        for i in range(num_views):
            self.ED_optimizers.append(torch.optim.Adam([{'params': self.D_s[i].parameters(), 'lr': self.config['discriminator_lr']},
                                      {'params': self.E_s[i].parameters(), 'lr':self.config['lr']}],
                                      betas=(0.7, 0.9), weight_decay=1e-5))
            self.FG_optimizers.append(torch.optim.Adam([{'params': self.F_s[i].parameters(), 'lr': self.config['mapping_lr']},
                                      {'params': self.G_s[i].parameters(), 'lr':self.config['lr']}],
                                      betas=(0.7, 0.9), weight_decay=1e-5))
        params = []
        for i in range(num_views):
            params.append({'params': self.projector_s[i].parameters()})
        for i in range(num_views):
            params.append({'params': self.E_s[i].parameters()})
        params.append({'params': self.CLU.parameters()})
        params.append({'params': self.weight})
        self.CLU_optimizer = torch.optim.Adam(params, betas=(0.7, 0.9), weight_decay=1e-5, lr=self.config['lr'])

        self.EDscheduler = []
        self.FGscheduler = []
        # for i in range(num_views):
        #     self.EDscheduler.append(
        #         torch.optim.lr_scheduler.CosineAnnealingLR(self.ED_optimizers[i], T_max=50, eta_min=1e-4))
        #     self.FGscheduler.append(
        #         torch.optim.lr_scheduler.CosineAnnealingLR(self.FG_optimizers[i], T_max=50, eta_min=1e-4))
        # self.cluscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.CLU_optimizer, T_max=50, eta_min=1e-4)
        for i in range(num_views):
            self.EDscheduler.append(
                torch.optim.lr_scheduler.StepLR(self.ED_optimizers[i], step_size=50, gamma=0.1))
            self.FGscheduler.append(
                torch.optim.lr_scheduler.StepLR(self.FG_optimizers[i],  step_size=50, gamma=0.1))
        self.cluscheduler = torch.optim.lr_scheduler.StepLR(self.CLU_optimizer, step_size=50, gamma=0.1)

    def train(self, train_dl, testdl, output_dir):
        trackers = []
        acc_tracker = LossTracker(output_dir)
        ddc_tracker = LossTracker(output_dir)
        for i in range(self.num_views):
            trackers.append(LossTracker(output_dir))

        for Epoch in range(self.config['epochs']):
            print(f"epoch:{Epoch + 1}")
            for batch_real_data in tqdm(train_dl):
                self.perform_train_step(batch_real_data, trackers, ddc_tracker)

            self.cluscheduler.step()
            for i in range(self.num_views):
                self.EDscheduler[i].step()
                self.FGscheduler[i].step()

            for i in range(self.num_views):
                trackers[i].plot(f'view_{i}')
            ddc_tracker.plot('dcc')

            if Epoch % 1 == 0:
                self.test(testdl, acc_tracker, Epoch+1)
                acc_tracker.plot('acc')

            if self.config['save_model']:
                self.save_train_state(os.path.join(output_dir, "last_ckp.pth"))

    def perform_train_step(self, batch_real_data, trackers=None, ddc_tracker=None):
        # Step I. Update E, and D: optimizer the discriminator D(E( * ))
        for i in range(self.num_views):
            self.ED_optimizers[i].zero_grad()
            L_adv_ED = self.get_ED_loss(batch_real_data[i], i)
            L_adv_ED.backward()
            self.ED_optimizers[i].step()
            trackers[i].update(dict(L_adv_ED=L_adv_ED))

        # Step II. Update F, and G: Optimize the generator G(F( * )) to fool D(E ( * ))
        for i in range(self.num_views):
            self.FG_optimizers[i].zero_grad()
            L_adv_FG = self.get_FG_loss(batch_real_data[i], i)
            L_adv_FG.backward()
            self.FG_optimizers[i].step()
            trackers[i].update(dict(L_adv_FG=L_adv_FG))

        # Step III. Update E, and G: Optimize the reconstruction loss in the Latent space W
        for i in range(self.num_views):
            self.ED_optimizers[i].zero_grad()
            self.FG_optimizers[i].zero_grad()
            # self.EG_optimizer.zero_grad()
            L_err_EG = self.get_EG_loss(batch_real_data[i], i)
            L_err_EG.backward()
            # self.EG_optimizer.step()
            self.ED_optimizers[i].step()
            self.FG_optimizers[i].step()
            trackers[i].update(dict(L_err_EG=L_err_EG))

        # step IV. Update E、CLU
        self.CLU_optimizer.zero_grad()
        L_clu_loss = self.get_clu_loss(batch_real_data)
        L_clu_loss.backward()
        self.CLU_optimizer.step()
        ddc_tracker.update(dict(L_DDC=L_clu_loss))

    def test(self, test_dl, acc_tracker, Epoch):
        record = [[0 for i in range(num_classes)] for j in range(num_classes)]
        batch_num = 0
        preds = None
        labels = None
        # for_vision = None #可视化用
        with torch.no_grad():
            for data in test_dl:
                label = data[-1]
                batch_num += 1
                com = []
                for i in range(self.num_views):
                    _, b = self.E_s[i](data[i])
                    b = self.projector_s[i](b)
                    com.append(b)

                # z 为用于聚类的特征，pred 表示每一类别的概率
                z = 0
                for i in range(len(com)):
                    z += self.weight[i] * com[i]
                    # z += com[i]                       # 无加权融合
                # if batch_num != 1:
                #     for_vision = np.concatenate([for_vision, z.cpu().numpy()])
                # else:
                #     for_vision = z.cpu().numpy()

                pred = self.CLU(z)
                y_pred = torch.argmax(pred, dim=1)

                # cluster = KMeans(n_clusters=num_classes, random_state=42).fit(z.cpu().numpy())
                # y_pred = cluster.labels_
                # y_pred = torch.from_numpy(y_pred)

                for i, j in zip(y_pred, label):
                    i = i.item()
                    j = j.item()
                    record[j][i] += 1

                if batch_num != 1:
                    preds = np.concatenate([preds, y_pred.cpu().numpy()])
                    labels = np.concatenate([labels, label.cpu().numpy()])
                else:
                    preds = y_pred.cpu().numpy()
                    labels = label.cpu().numpy()

            if len(labels.shape) > 1 or len(preds.shape) > 1:
                labels = labels.flatten()
                preds = preds.flatten()
            nmi = NMI(labels, preds)
            print(f"NMI:{nmi}")

            # tsne2d = TSNE(n_components=2, init='pca', random_state=0, n_iter=500)  # 可视化用
            # X_tsne_2d = tsne2d.fit_transform(for_vision)
            # plot_embedding_2d(X_tsne_2d[:, 0:2], labels, "t-SNE Epoch "+str(Epoch))

            cmat = np.array(record)
            ri, ci = linear_sum_assignment(-cmat)
            ordered = cmat[np.ix_(ri, ci)]
            acc = np.sum(np.diag(ordered)) / np.sum(ordered)
            print(f"ACC：{acc}")
            acc_tracker.update(dict(ACC=acc))

            print('Matrix')
            for i in ordered:
                for j in i:
                    print('%4d' % j, end='')
                print()

    def get_clu_loss(self, batch_real_data):
        com = []
        # present = []a
        for i in range(self.num_views):
            a, b = self.E_s[i](batch_real_data[i])
            # a = self.projector_s[i](a)
            # present.append(a)
            b = self.projector_s[i](b)
            com.append(b)

        # z 为用于聚类的特征，pred 表示每一类别的概率
        z = 0
        for i in range(len(com)):
            z += self.weight[i] * com[i]
            # z += com[i]  # 无加权融合
        pred = self.CLU(z)

        # loss = 0
        # DDC
        # 得到样本之间的特征距离
        d_matrix = cdist(z, z)

        # 得到核矩阵 K
        K = kernel_from_distance_matrix(d_matrix, rel_sigma=0.15)

        # 得到分类结果 A
        A = pred

        # DDC1
        loss = d_cs(A, K, self.num_classes)

        # DDC2
        n = A.size(0)
        loss += 2 / (n * (n - 1)) * triu(A @ torch.t(A))

        # DDC3
        eye = torch.eye(self.num_classes, device='cuda')
        m = torch.exp(-cdist(A, eye))
        loss += d_cs(m, K, self.num_classes)

        #约束共同部分
        # com_mean = sum(com)/len(com)
        # for i in range(self.num_views):
        #     loss += torch.mean((com[i]-com_mean)**2)

        z = z.detach()
        for i in range(self.num_views-1):
            c = z.T @ com[i]
            c.div_(com[0].shape[0])
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss += 0.001 * (on_diag + self.lambd * off_diag)          # MSRC：0.001

        return loss

    def get_ED_loss(self, batch_real_data, i):
        batch_real_data.requires_grad_(True)
        batch_z = torch.randn(batch_real_data.shape[0], self.config['z_dim'], dtype=torch.float32).cuda()
        with torch.no_grad():
            batch_fake_data = self.G_s[i](self.F_s[i](batch_z)[0])
        fake_images_dicriminator_outputs = self.D_s[i](self.E_s[i](batch_fake_data)[0])
        real_images_dicriminator_outputs = self.D_s[i](self.E_s[i](batch_real_data)[0])
        loss = F.softplus(fake_images_dicriminator_outputs) + F.softplus(-real_images_dicriminator_outputs)

        r1_penalty = compute_r1_gradient_penalty(real_images_dicriminator_outputs, batch_real_data)

        loss += self.config['g_penalty_coeff'] * r1_penalty
        loss = loss.mean()

        return loss

    def get_FG_loss(self, batch_real_data, i):
        batch_z = torch.randn(batch_real_data.shape[0], self.config['z_dim'], dtype=torch.float32).cuda()
        batch_fake_data = self.G_s[i](self.F_s[i](batch_z)[0])
        fake_images_dicriminator_outputs = self.D_s[i](self.E_s[i](batch_fake_data)[0])
        loss = F.softplus(-fake_images_dicriminator_outputs).mean()

        return loss

    def get_EG_loss(self, batch_real_data, i):
        batch_z = torch.randn(batch_real_data.shape[0], self.config['z_dim'], dtype=torch.float32).cuda()
        batch_w = self.F_s[i](batch_z)[0]
        batch_reconstructed_w = self.E_s[i](self.G_s[i](batch_w))[0]
        return torch.mean(((batch_reconstructed_w - batch_w.detach())**2))

    def load_train_state(self, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.F_s.load_state_dict(checkpoint['F'])
            self.G_s.load_state_dict(checkpoint['G'])
            self.E_s.load_state_dict(checkpoint['E'])
            self.D_s.load_state_dict(checkpoint['D'])
            print(f"Checkpoint {os.path.basename(checkpoint_path)} loaded.")

    def save_train_state(self, save_path):
        torch.save(
            {
                'F': self.F_s.state_dict(),
                'G': self.G_s.state_dict(),
                'E': self.E_s.state_dict(),
                'D': self.D_s.state_dict(),
            },
            save_path
        )

if __name__ == '__main__':
    output_dir = os.path.join('结果信息', f"MSRC/result")
    # input_dim, num_classes, batch_size = config_init('NUS5')
    # dataset = Datasets.get_NUS5()
    # input_dim, num_classes, batch_size = config_init('UCI6')
    # dataset = Datasets.get_UCI6()
    # input_dim, num_classes, batch_size = config_init('UCI2')
    # dataset = Datasets.get_UCI2()
    input_dim, num_classes, batch_size = config_init('MSRC')
    dataset = Datasets.get_MSRC()
    # input_dim, num_classes, batch_size = config_init('Yale')
    # dataset = Datasets.get_Yale()
    # input_dim, num_classes, batch_size = config_init('Caltech101')
    # dataset = Datasets.get_Caltech101()
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testdl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_len = len(dataset)
    config = { 'lr': 0.001,                     # Yale: 0.001, MSRC: 0.001, NUS,Cal:0.0005
               'discriminator_lr': 0.01,        # Yale: 0.001, MSRC: 0.01, NUS,Cal:0.005
               'mapping_lr': 0.001,             # Yale: 0.001, MSRC: 0.001, NUS,Cal:0.0005
               'class_dim': num_classes,
                'z_dim': 64,
                'present_dim': 512,             # Yale,NUS,Cal: 2048, MSRC: 512
                'w_exp': 2048,
                'common_coefficient': 0.5,       #Yale: 0.3, Cal: 0.8, NUS: 0.3, MSRC: 0.5
                'epochs': 200,
                'save_model': False,
                'g_penalty_coeff': 10,           # Yale: 200, MSRC,NUS,Cal: 10
                }

    # class_sample_counts = [497, 1422, 1253, 139, 327, 866, 258, 287, 289, 1005, 352, 1093, 159, 3341, 288, 697, 1054,
    #                        941, 2563, 958, 611, 414, 1578, 329, 1185, 1238, 373, 2567, 3573, 199, 144]
    # dataset = Datasets.NUS5()
    # weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # train_targets = dataset.get_classes_for_all_imgs()
    # train_targets = list(train_targets.squeeze(1))
    # samples_weights = []
    # for i in train_targets:
    #     samples_weights.append(weights[i])
    # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    # dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    model = SSMALAE(num_views=len(input_dim), num_classes=num_classes, input_dim=input_dim, config=config)
    # print(model)

    # # 定义总参数量、可训练参数量及非可训练参数量变量
    # Total_params = 0
    # Trainable_params = 0
    # NonTrainable_params = 0
    # # 遍历model.parameters()返回的全局参数列表
    # for param in model.parameters():
    #     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    #     Total_params += mulValue  # 总参数量
    #     if param.requires_grad:
    #         Trainable_params += mulValue  # 可训练参数量
    #     else:
    #         NonTrainable_params += mulValue  # 非可训练参数量

    # print(f'Total params: {Total_params}')
    # print(f'Trainable params: {Trainable_params}')
    # print(f'Non-trainable params: {NonTrainable_params}')

    model.train(dataLoader, testdl, output_dir)



