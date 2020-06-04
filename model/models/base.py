import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')

    def split_instances(self, data=None):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def get_embeddings_dict(self, embeddings, all_labels, logits=None):
        support_idx, query_idx = self.split_instances()

        support_embeddings = embeddings[support_idx.flatten()]
        query_embeddings = embeddings[query_idx.flatten()]

        support_labels = all_labels[support_idx.flatten()]
        query_labels = all_labels[query_idx.flatten()]

        way = self.args.way if self.training else self.args.eval_way
        label_onehot = torch.zeros((all_labels.size(0), way), device=all_labels.device)
        label_onehot.scatter_(1, all_labels[:, None], 1)

        support_labels_onehot = label_onehot[support_idx.flatten()]
        query_labels_onehot = label_onehot[query_idx.flatten()]
        d = {
            'support_embeddings': support_embeddings,
            'query_embeddings': query_embeddings,
            'support_labels': support_labels,
            'query_labels': query_labels,
            'support_labels_onehot': support_labels_onehot,
            'query_labels_onehot': query_labels_onehot,
            'way': way
        }

        if logits is not None:
            d['support_logits'] = logits[support_idx.flatten()]
            d['query_logits'] = logits[query_idx.flatten()]

        return d


    def forward(self, x, get_feature=False, return_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)

            if self.training:
                logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                if return_feature:
                    return instance_embs, logits, logits_reg
                else:
                    return logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                if return_feature:
                    return instance_embs, logits
                else:
                    return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')