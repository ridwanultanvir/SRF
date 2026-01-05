import torch
import torch.nn as nn
import torch.nn.functional as F


class VTCLoss_caption(nn.Module):
    def __init__(self, temperature=0.07):
        super(VTCLoss_caption, self).__init__()
        self.temperature = temperature

    def forward(self, src_txt, src_vid):
        # src_txt: (bs, v_dim, h_dim)
        # src_vid: (bs, v_dim, h_dim)
        bs, v_dim, h_dim = src_vid.size()

        # normalize the feature vectors
        src_txt = F.normalize(src_txt, dim=2)  # (bs, v_dim, h_dim)
        src_vid = F.normalize(src_vid, dim=2)  # (bs, v_dim, h_dim)

        # compute the similarity matrix
        sim_mat = torch.bmm(src_txt, src_vid.transpose(1, 2))  # (b,v,h)*(b,h,v)=(bs, v_dim, v_dim)

        # create the positive and negative masks
        pos_mask = torch.eye(v_dim).bool().to(sim_mat.device).unsqueeze(0).expand(bs, -1, -1)  # (bs, v_dim, v_dim)
        neg_mask = ~pos_mask  # (bs, v_dim, v_dim)

        # compute the logits and labels
        logits = sim_mat / self.temperature  # (bs, v_dim, v_dim)
        labels = torch.arange(v_dim).repeat(bs, 1).to(sim_mat.device)  # (bs, v_dim)

        # compute the cross entropy loss for text-to-video and video-to-text
        loss_t2v = F.cross_entropy(logits.view(bs * v_dim, -1), labels.view(-1))  # scalar
        loss_v2t = F.cross_entropy(logits.transpose(1, 2).contiguous().view(bs * v_dim, -1), labels.view(-1))  # scalar

        # return the average loss
        return (loss_t2v + loss_v2t) / 2
