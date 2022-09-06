if hasattr(self.args, 'add_contrastive') and self.args.add_contrastive and sentence_labels.numel() > 0:
    if hasattr(self.args, 'add_contrastive_margin') and self.args.add_contrastive_margin:
        margin = 1e-2
    else:
        margin = -float('inf')
    num_negs = 5
    temperature = 8
    sents_ten = activations[sentence_mask]
    q_tensors = [x(activations[:, 0]).unsqueeze(-1) for x in self.qa_type_matrix_q]
    a_tensors_positive = [x(sents_ten).unsqueeze(-1) for x in self.qa_type_matrix_a]
    q_tensors = torch.cat(q_tensors, dim=-1)
    a_tensors_positive = torch.cat(a_tensors_positive, dim=-1).unsqueeze(0)
    q_type_idcs = q_type_labels.repeat(q_tensors.size()[1], 1).view(-1, q_tensors.size()[1], 1)
    a_type_idcs = q_type_labels.repeat(a_tensors_positive.size()[1], a_tensors_positive.size()[2], 1) \
        .view(-1, a_tensors_positive.size()[1], a_tensors_positive.size()[2], 1)
    q_tensors = torch.gather(q_tensors, -1, q_type_idcs).squeeze(-1)
    if hasattr(self.args, 'add_contrastive_question_type') and self.args.add_contrastive_question_type:
        combined = torch.cat((q_type_labels, torch.arange(3).to(q_type_labels.device)))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        qtype_type_idcs = difference.repeat(a_tensors_positive.size()[1], a_tensors_positive.size()[2], 1) \
            .view(-1, a_tensors_positive.size()[1], a_tensors_positive.size()[2], 2)
        qtype_tensors_negative = torch.gather(a_tensors_positive, -1, qtype_type_idcs).squeeze(-1)
    a_tensors_positive = torch.gather(a_tensors_positive, -1, a_type_idcs).squeeze(-1)
    dots_pos = torch.bmm(a_tensors_positive, q_tensors.unsqueeze(-1))
    norms_pos = torch.bmm(a_tensors_positive.norm(p=1, dim=-1).unsqueeze(-1),
                          q_tensors.unsqueeze(-1).norm(p=1, dim=-2).unsqueeze(-2))
    pos_scores = temperature * torch.clamp(torch.div(dots_pos, norms_pos + 1e-5).squeeze(-1), min=margin)
    if hasattr(self.args, 'add_contrastive_question_type') and self.args.add_contrastive_question_type:
        dots_neg = torch.bmm(qtype_tensors_negative.transpose(-1, -2).reshape(1, -1, a_tensors_positive.size(-1)),
                             q_tensors.unsqueeze(-1))
        norms_qneg = torch.bmm(dots_neg.norm(p=1, dim=-1).unsqueeze(-1),
                               q_tensors.unsqueeze(-1).norm(p=1, dim=-2).unsqueeze(-2))
        qneg_scores = temperature * torch.clamp(torch.div(dots_neg, norms_qneg + 1e-5).squeeze(-1),
                                                min=margin)
        qneg_scores = qneg_scores.view(1, -1, 2)
        qneg_scores[sentence_labels == 0] = -float('inf')
        if hasattr(self.args,
                   'add_contrastive_negative_sampling') and self.args.add_contrastive_negative_sampling:
            qneg_scores = qneg_scores[sentence_labels == 0].unsqueeze(0)
            r = torch.randperm(qneg_scores.size(1))
            qneg_scores = qneg_scores[:, r]
            qneg_scores = qneg_scores[:, :num_negs]
        qneg_scores = qneg_scores.view(1, -1)
    else:
        qneg_scores = torch.FloatTensor([-float('inf')])
    neg_scores = pos_scores.clone()
    cont_sent_scores = dots_pos.squeeze(-1).clone()
    neg_scores[sentence_labels == 1] = -float('inf')
    pos_scores[sentence_labels == 0] = -float('inf')
    if hasattr(self.args, 'add_contrastive_negative_sampling') and self.args.add_contrastive_negative_sampling:
        neg_scores = neg_scores[sentence_labels == 0].unsqueeze(0)
        r = torch.randperm(neg_scores.size(1))
        neg_scores = neg_scores[:, r]
        neg_scores = neg_scores[:, :num_negs]
    cont_loss_nom = pos_scores
    if hasattr(self.args, 'add_contrastive_question_type') and self.args.add_contrastive_question_type:
        cont_loss_denom = torch.cat([neg_scores, pos_scores, qneg_scores], dim=1).logsumexp(dim=-1, keepdim=False)
    else:
        cont_loss_denom = torch.cat([neg_scores, pos_scores], dim=1).logsumexp(dim=-1, keepdim=False)
    cont_loss = -(cont_loss_nom - cont_loss_denom)
    cont_loss = cont_loss[~torch.isinf(cont_loss)].mean()
else:
    cont_loss = 0.0
