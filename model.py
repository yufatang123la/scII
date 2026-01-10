import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

import math
class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()


    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        ZX = torch.matmul(query, query.transpose(-2, -1))
        scaling_factor = math.sqrt(x.size(-1))
        AX = self.softmax(ZX / scaling_factor)
        output_X = torch.matmul(AX, value)
        output = self.attn_dropout(output_X)

        return output
class MaxOut(nn.Module):
    def __init__(self, k):
        super(MaxOut, self).__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x
    def forward(self, x):
        return x.view(x.size(0), -1, self.k).max(dim=2)[0]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=type_num, k=2,dropout_prob = 0.01):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = k
        self.dropout_prob = dropout_prob  # dropout

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            MaxOut(self.k),  # MaxOut
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm1d(self.hidden_dim // self.k),

            nn.Linear(input_dim, hidden_dim),
            #nn.ReLU(),
            MaxOut(k),
            nn.Linear(hidden_dim // k, output_dim)

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
    def forward(self, x):
        return self.classifier(x)
class Net(nn.Module):
    def __init__(self, gene_num, type_num, ce_weight, args):
        super(Net, self).__init__()
        self.type_num = type_num
        self.ce_weight = ce_weight
        self.align_loss_epoch = args.align_loss_epoch
        self.encoder = nn.Sequential(
            nn.Linear(gene_num, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64)
        )
        self.attention = SelfAttention(64, 64)  # Self-Attention


        #self.classifier = nn.Sequential(
            #nn.ReLU(),
            #nn.Linear(64, type_num),
        #)

        # MLP
        self.classifier = MLPClassifier(input_dim=64, hidden_dim=128, output_dim=type_num,k = 2)
       

        self.adj_decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def run(
        self,
        scrna_dataloader_train,
        scrna_dataloader_eval,
        scatac_dataloader_train,
        scatac_dataloader_eval,
        scatac_adj,
        args,
    ):
        optim = torch.optim.AdamW(self.parameters(), lr=args.learning_rate)
        wce_loss = nn.CrossEntropyLoss(weight=self.ce_weight)
        align_loss = AlignLoss(type_num=self.type_num, feature_dim=64, args=args)
        epochs = args.train_epoch
        scatac_iter = iter(scatac_dataloader_train)
        for epoch in range(epochs):
            wce_loss_epoch = align_loss_epoch = stc_loss_epoch = 0.0
            train_acc = train_tot = 0.0
            self.train()
            for (scrna_x, scrna_y) in scrna_dataloader_train:
                scrna_x = scrna_x.cuda()
                scrna_y = scrna_y.cuda()
                try:
                    (scatac_x, adj_index), scatac_index = next(scatac_iter)
                except StopIteration:
                    scatac_iter = iter(scatac_dataloader_train)
                    (scatac_x, adj_index), scatac_index = next(scatac_iter)
                scatac_x = scatac_x.cuda()

                scrna_h = self.encoder(scrna_x)
                scrna_pred = self.classifier(scrna_h)
                scatac_h = self.encoder(scatac_x)

                loss_wce = wce_loss(scrna_pred, scrna_y)

                wce_loss_epoch += loss_wce.item()
                train_acc += (
                    torch.argmax(
                        scrna_pred,
                        dim=-1,
                    )
                    == scrna_y
                ).sum()
                train_tot += scrna_x.shape[0]

                loss_epoch = loss_wce

                 if epoch >= self.align_loss_epoch:
                    if args.selection_mode == "dual":
                        target_reliability = prob_feature[target_index] * prob_logit[target_index]
                    elif args.selection_mode == "similarity":
                        target_reliability = prob_feature[target_index]  # only S^a
                    elif args.selection_mode == "mlp":
                        target_reliability = prob_logit[target_index]  # only MLP loss
                    else:
                        raise ValueError(f"Unknown selection_mode: {args.selection_mode}")

                    loss_align = align_loss(
                        source_h,
                        source_y,
                        target_h,
                        preds[target_index],
                        target_reliability, 
                    )
                    loss_epoch += loss_align
                    align_loss_epoch += loss_align.item()

                if args.novel_type:
                    adj = scatac_adj[adj_index, :][:, adj_index]
                    cos_sim_x = torch.from_numpy(adj).float().cuda()
                    scatac_h = F.normalize(self.adj_decoder(scatac_h), dim=-1)
                    cos_sim_h = F.relu(scatac_h @ scatac_h.T)
                    stc_loss = (cos_sim_x - cos_sim_h) * (cos_sim_x - cos_sim_h)
                    stc_loss = torch.clamp(stc_loss - 0.01, min=0).mean()
                    loss_epoch += stc_loss
                    stc_loss_epoch += stc_loss.item()

                optim.zero_grad()
                loss_epoch.backward()
                optim.step()

            train_acc /= train_tot
            wce_loss_epoch /= len(scrna_dataloader_train)
            align_loss_epoch /= len(scrna_dataloader_train)
            stc_loss_epoch /= len(scrna_dataloader_train)

            feature_vec, type_vec, omic_vec, loss_vec = self.inference(
                scrna_dataloader_eval, scatac_dataloader_eval
            )
            similarity, preds = feature_prototype_similarity(
                feature_vec[omic_vec == 0],
                type_vec,
                feature_vec[omic_vec == 1],
            )
            if epoch == self.align_loss_epoch - 1:
                align_loss.init_prototypes(
                    feature_vec[omic_vec == 0],
                    type_vec,
                    feature_vec[omic_vec == 1],
                    preds,
                )
            prob_feature = gmm(1 - similarity)
            prob_logit = gmm(loss_vec)

            preds = torch.from_numpy(preds).long().cuda()
            prob_feature = torch.from_numpy(prob_feature).float().cuda()
            prob_logit = torch.from_numpy(prob_logit).float().cuda()

            if args.novel_type:
                print(
                    "epoch [%d/%d] wce: %.4f, mse: %.4f, train_acc: %.4f"
                    % (
                        epoch,
                        epochs,
                        wce_loss_epoch,
                        align_loss_epoch,
                        train_acc,
                    )
                )
            else:
                print(
                    "epoch [%d/%d] wce: %.4f, mse: %.4f, train_acc: %.4f"
                    % (epoch, epochs, wce_loss_epoch, align_loss_epoch, train_acc)
                )

            if train_acc > args.early_stop_acc:
                print("Early Stop.")
                break
        return preds.cpu(), prob_feature.cpu(), prob_logit.cpu()

    def inference(self, scrna_dataloader, scatac_dataloader):
        self.eval()
        feature_vec, type_vec, omic_vec, loss_vec = [], [], [], []
        for (x, y) in scrna_dataloader:
            x = x.cuda()
            with torch.no_grad():
                h = self.encoder(x)
                logit = self.classifier(h)
            feature_vec.extend(h.cpu().numpy())
            type_vec.extend(y.numpy())
            omic_vec.extend(np.zeros(x.shape[0]))
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        for (x, _), _ in scatac_dataloader:
            x = x.cuda()
            with torch.no_grad():
                h = self.encoder(x)
                logit = self.classifier(h)
                pred = torch.argmax(logit, dim=-1)
                loss = ce_loss(logit, pred)
            feature_vec.extend(h.cpu().numpy())
            omic_vec.extend(np.ones(x.shape[0]))
            loss_vec.extend(loss.cpu().numpy())
        feature_vec, type_vec, omic_vec, loss_vec = (
            np.array(feature_vec),
            np.array(type_vec),
            np.array(omic_vec),
            np.array(loss_vec),
        )
        return feature_vec, type_vec, omic_vec, loss_vec

def dynamic_gmm(X, n_components_range=(2, 10), metric_weights=None):
    if X.ndim == 1:
        X = X[:, np.newaxis]

    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    if metric_weights is not None:
        metric_weights = np.array(metric_weights)
        if X.shape[1] == len(metric_weights):
            metric_weights /= metric_weights.sum()
            X = np.dot(X, metric_weights)
            X = X[:, np.newaxis]

    best_gmm = None
    best_bic = float('inf')
    best_n_components = 0
    for n_components in range(n_components_range[0], n_components_range[1] + 1):
        gmm = GaussianMixture(n_components=n_components, max_iter=200, tol=1e-4, reg_covar=5e-4, random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n_components = n_components

    prob = best_gmm.predict_proba(X)[:, best_gmm.means_.argmin()]
    return prob, best_n_components


def feature_prototype_similarity(scrna_feature, scrna_label, scatac_feature):
    type_num = scrna_label.max() + 1
    scrna_prototypes = np.zeros((type_num, scrna_feature.shape[1])).astype(float)
    for k in range(type_num):
        scrna_prototypes[k] = scrna_feature[scrna_label == k].sum(axis=0)
    similarity = cosine_similarity(scatac_feature, scrna_prototypes)
    pred = np.argmax(similarity, axis=1)
    similarity = np.max(similarity, axis=1)
    return similarity, pred


class AlignLoss(nn.Module):
    def __init__(self, type_num, feature_dim, args):
        super(AlignLoss, self).__init__()
        self.type_num = type_num
        self.feature_dim = feature_dim
        self.scrna_prototypes = torch.zeros(self.type_num, self.feature_dim).cuda()
        self.scatac_prototypes = torch.zeros(self.type_num, self.feature_dim).cuda()
        self.momentum = args.prototype_momentum
        self.criterion = nn.MSELoss()

    def init_prototypes(
        self, scrna_feature, scrna_label, scatac_feature, scatac_prediction
    ):
        scrna_feature = torch.from_numpy(scrna_feature).cuda()
        scrna_label = torch.from_numpy(scrna_label).cuda()
        scatac_feature = torch.from_numpy(scatac_feature).cuda()
        scatac_prediction = torch.from_numpy(scatac_prediction).cuda()
        for k in range(self.type_num):
            self.scrna_prototypes[k] = scrna_feature[scrna_label == k].mean(dim=0)
            scatac_index = scatac_prediction == k
            if scatac_index.sum() != 0:
                self.scatac_prototypes[k] = scatac_feature[scatac_index].mean(dim=0)

    def forward(
        self,
        scrna_feature,
        scrna_label,
        scatac_feature,
        scatac_prediction,
        scatac_reliability,
    ):
        self.scrna_prototypes.detach_()
        self.scatac_prototypes.detach_()
        for k in range(self.type_num):
            scrna_index = scrna_label == k
            if scrna_index.sum() != 0:
                self.scrna_prototypes[k] = self.momentum * self.scrna_prototypes[
                    k
                ] + (1 - self.momentum) * scrna_feature[scrna_label == k].mean(dim=0)
            scatac_index = scatac_prediction == k
            if scatac_index.sum() != 0:
                if torch.abs(self.scatac_prototypes[k]).sum() > 1e-7:
                    self.scatac_prototypes[k] = self.momentum * self.scatac_prototypes[
                        k
                    ] + (1 - self.momentum) * (
                        scatac_reliability[scatac_index].unsqueeze(1)
                        * scatac_feature[scatac_index]
                    ).mean(
                        dim=0
                    )
                else:  # Not Initialized
                    self.scatac_prototypes[k] = (
                        scatac_reliability[scatac_index].unsqueeze(1)
                        * scatac_feature[scatac_index]
                    ).mean(dim=0)
        loss = self.criterion(
            F.normalize(self.scrna_prototypes, dim=-1),
            F.normalize(self.scatac_prototypes, dim=-1),
        )
        if (torch.abs(self.scatac_prototypes).sum(dim=1) > 1e-7).sum() < self.type_num:
            loss *= 0
        return loss
