import torch
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False, T=0.1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs
        self.T = T

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        parser.add_argument('--T', default=0.1, required=False, type=float,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for x1, x2, targets in trn_loader:
            # Forward current model
            out1, z1 = self.model(x1.to(self.device), return_features=True)
            out2, z2 = self.model(x2.to(self.device), return_features=True)

            features = torch.cat((z1, z2), dim=1).reshape((-1, 2, z1.size()[1]))
            
            # Concatenate the output of each different input augmentation
            outputs = []
            for i in range(len(out1)):
                outputs.append(torch.cat((out1[i], out2[i]), dim=0))

            labels = torch.cat((targets,targets), dim=0)

            loss = self.contrastive_loss(features, targets.to(self.device), temperature=self.T)
            
            cross_entropy = self.criterion(t, outputs, labels.to(self.device))

            loss = loss + cross_entropy
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        print('Loss:{}'.format(loss.item()))

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for x1, x2, targets in val_loader:
                # Forward current model
                outputs, z1 = self.model(x1.to(self.device), return_features=True)
                #out2, z2 = self.model(x2.to(self.device), return_features=True)

                #features = torch.cat((z1,z2), dim=0).reshape((-1, 2, z1.size()[1]))
                z1 = z1.reshape((-1,1,z1.size()[1]))

                # Concatenate the output of each different input augmentation
                #outputs = []
                #for i in range(len(out1)):
                #    outputs.append(torch.cat((out1[i], out2[i]), dim=0))             


                #labels = torch.cat((targets,targets), dim=0)

                
                loss = self.contrastive_loss(z1, targets.to(self.device), self.T, 'all')
                cross_entropy = self.criterion(t, outputs, targets.to(self.device))

                loss = loss + cross_entropy

                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])


    def contrastive_loss(self, features, targets, temperature, contrast_mode='all'):
        """Compute loss for model.  If 'labels' is None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
         
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0] 


        if targets is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif targets is not None:
            labels = targets.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        # It's possible to apply L2 norm to input vectors 
        contrast_features = torch.nn.functional.normalize(contrast_features, p=2, dim=0)

        if contrast_mode == 'one':
            anchor_feature = features [:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_features
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(
                contrast_features, contrast_features.T
            ),
            temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create tile mask.
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            -1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        sum_exp_logits = exp_logits.sum(1, keepdim=True)
        sum_exp_logits[sum_exp_logits == 0] = 1.0

        log_prob = logits - torch.log(sum_exp_logits)

        # Compute mean of log-likelihood over positive.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss
