import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os
import time
import pickle
import zipfile
import sys

from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()

        ## Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        print("Loader Length = ", loader.__len__())

        for num, (data, speaker_labels, phrase_labels) in enumerate(loader, start=1):
            self.zero_grad()
            speaker_labels = torch.LongTensor(speaker_labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, speaker_labels)
            nloss.backward()
            self.optim.step()

            index += len(speaker_labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(speaker_labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(speaker_labels)

    def enroll_network(self, enrollLoader, path_save_model):
        self.eval()
        print("I am in enroll method ....")
        enrollments = {}

        for data, label in enrollLoader:
            model_id = label
            audio_segments = data.cuda()
            with torch.no_grad():
                embeddings = []
                for audio in audio_segments:
                    embedding = self.speaker_encoder.forward(audio.unsqueeze(0), aug=False)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    embeddings.append(embedding)
                averaged_embedding = torch.mean(torch.stack(embeddings), dim=0)
            enrollments[model_id] = averaged_embedding

        os.makedirs(path_save_model, exist_ok=True)
        with open(os.path.join(path_save_model, "enrollments.pkl"), "wb") as f:
            pickle.dump(enrollments, f)

    def test_network(self, testLoader, path_save_model, compute_eer=True):
        self.eval()
        enrollments_path = os.path.join(path_save_model, "enrollments.pkl")
        print(f"Loading enrollments from {enrollments_path}")
        with open(enrollments_path, "rb") as f:
            enrollments = pickle.load(f)

        scores, labels = [], []

        for data, label in testLoader:
            model_id, test_file = label
            audio = data.cuda()
            with torch.no_grad():
                test_embedding = self.speaker_encoder.forward(audio, aug=False)
                test_embedding = F.normalize(test_embedding, p=2, dim=1)

            score = torch.mean(torch.matmul(test_embedding, enrollments[model_id].T)).detach().cpu().numpy()
            scores.append(score)
            labels.append(int(test_file.split('_')[1]))  # Assuming label extraction logic from filename

        if compute_eer and labels:
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        else:
            EER = None
            minDCF = None

        answer_file_path = os.path.join(path_save_model, "answer.txt")
        with open(answer_file_path, 'w') as f:
            for score in scores:
                f.write(f"{score}\n")

        submission_zip_path = os.path.join(path_save_model, "submission.zip")
        with zipfile.ZipFile(submission_zip_path, 'w') as zipf:
            zipf.write(answer_file_path, os.path.basename(answer_file_path))

        return EER, minDCF, scores

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name is name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
