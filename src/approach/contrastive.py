import torch
import warnings
import numpy as np
from torchvision import transforms
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform


class Appr(Inc_Learning_Appr):
    """Class implementing the Supervised Contrastive Loss and classification icarl like."""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False, T=0.1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs
        self.T = T

        self.contrastive_transforms = None

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')

        parser.add_argument('--T', default=0.7, required=False, type=float,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

        # Algorithm 1: iCaRL NCM Classify

    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1).squeeze()
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = torch.utils.data.DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                                       num_workers=trn_loader.num_workers,
                                                       pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
            # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    feats = self.model(images.to(self.device), return_features=True)[1]
                    # normalize
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        self.exemplar_means = []

        img_size = trn_loader.dataset[0][0].shape[1:2]

        # add exemplars to train_loader
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        self.contrastive_transforms = ContrastiveTransformation(
            torch.nn.Sequential(transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop(
                                    size=img_size),
                                transforms.RandomApply([
                                    transforms.ColorJitter(
                                        brightness=0.5,
                                        contrast=0.5,
                                        saturation=0.5,
                                        hue=0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.GaussianBlur(
                                    kernel_size=3),
                                transforms.Normalize((0.5,), (0.5,))
                                ), 2)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        # compute mean of exemplars
        self.compute_mean_of_exemplars(trn_loader, val_loader.dataset.transform)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            x1, x2 = self.contrastive_transforms(images)
            _, z1 = self.model(x1.to(self.device), return_features=True)
            _, z2 = self.model(x2.to(self.device), return_features=True)

            loss = self.criterion(t, z1, z2, targets.to(self.device), temperature=self.T)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:# Forward old model
                # Forward current model

                x1, x2 = self.contrastive_transforms(images)
                out1, z1 = self.model(x1.to(self.device), return_features=True)
                out2, z2 = self.model(x2.to(self.device), return_features=True)
                loss = self.criterion(t, z1, z2, targets.to(self.device), self.T)

                outputs = []
                for i in range(len(out1)):
                    outputs.append(torch.cat((out1[i], out2[i]), dim=0))

                feats = torch.cat((z1, z2), dim=0)
                transformed_targets = torch.cat((targets, targets), dim=0)
                # during training, the usual accuracy is computed on the outputs
                if not self.exemplar_means:
                    hits_taw, hits_tag = self.calculate_metrics(outputs, transformed_targets)
                else:
                    hits_taw, hits_tag = self.classify(t, feats, transformed_targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, z1, z2, targets, temperature):
        """Returns the loss value"""
        feature_vectors = torch.cat((z1, z2), dim=0)
        feature_vectors = torch.reshape(feature_vectors, shape=(-1, len(z1[0])))

        feature_vector_normalized = torch.nn.functional.normalize(feature_vectors, p=2, dim=1)

        # Make targets vector in one-hot notation.
        labels = torch.cat((targets, targets), dim=0)
        labels = torch.nn.functional.one_hot(labels).float()
        labels_T = torch.transpose(labels, 0, 1)
        mask = torch.matmul(labels, labels_T)

        logits = torch.div(
            torch.matmul(
                feature_vector_normalized, torch.transpose(feature_vector_normalized, 0, 1)
            ),
            temperature,
        )

        exp_logits = torch.exp(logits)

        # Create positive and negative mask.
        positives_mask = torch.ones(len(labels), len(labels), device=self.device)
        positives_mask = mask * positives_mask

        negatives_mask = 1 - positives_mask
        positives_mask.fill_diagonal_(0)

        num_positive_per_row = torch.sum(positives_mask, dim=1)

        # Create the denominator of the loss.
        denominator = exp_logits + torch.sum(exp_logits * negatives_mask, dim=1, keepdim=True) + torch.sum(
            exp_logits * positives_mask, dim=1, keepdim=True)

        log_probs = (logits - torch.log(denominator)) * positives_mask
        log_probs = torch.sum(log_probs, dim=1)

        # Check if the positive per row are > 0.
        log_probs = torch.div(log_probs, num_positive_per_row)
        log_probs = torch.nan_to_num(log_probs)
        loss = -log_probs

        #loss = torch.reshape(loss, shape=[2, len(targets)])

        loss = torch.mean(loss, dim=0)

        return loss


class ContrastiveTransformation(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
