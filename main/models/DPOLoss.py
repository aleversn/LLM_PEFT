import torch.nn as nn
import torch
import torch.nn.functional as F

class DPOLoss(nn.Module):
    """
    DPO Loss
    """
    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta
        

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_rejected_logps: torch.Tensor,
            reference_chosen_logps: torch.Tensor,
            reference_rejected_logps: torch.Tensor,
    ):
        """
        policy_chosen_logps: 模型输出的对数概率。Shape: (batch_size,)
        policy_rejected_logps:   Shape: (batch_size,)
        reference_chosen_logps: Shape: (batch_size,)
        reference_rejected_logps: Shape: (batch_size,)
        """
        policy_logps = policy_chosen_logps - policy_rejected_logps
        reference_logps = reference_chosen_logps - reference_rejected_logps
        logits = policy_logps - reference_logps

        policy_rewards = policy_logps.detach()
        reference_rewards = reference_logps.detach()
        margin = logits.detach()

        loss = -F.logsigmoid(self.beta * logits)

        # 下面两个用于追踪训练的进度
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()
    

        # 对每个batch进行平均
        return loss.mean(), chosen_rewards.mean().cpu().item(), rejected_rewards.mean().cpu().item(), policy_rewards.mean().cpu().item(), reference_rewards.mean().cpu().item(), margin.mean().cpu().item()