"""
Scale loss functions so that mostly-undefined images won't be given undue weight.  Loss functions are adapted from Solaris.
"""

class ScaledTorchDiceLoss(nn.Module):
    def __init__(self, scale=True, reduce=True, logits=False):
        super().__init__()
        self.scale = scale
        self.reduce = reduce
        self.logits = logits

    def forward(self, outputs, targets):
        if self.logits:
            outputs = torch.sigmoid(outputs)

        batch_size = outputs.size()[0]
        eps = 1e-5
        dice_outputs = outputs.view(batch_size, -1)
        dice_targets = targets.view(batch_size, -1).float()
        intersection = torch.sum(dice_outputs * dice_targets, dim=1)
        integral = torch.sum(dice_outputs, dim=1) + torch.sum(dice_targets, dim=1) + eps
        loss = 1 - (2 * intersection + eps) / integral
        #print((' dice', list(loss.detach().cpu().numpy())))

        if self.scale:
            filtersrc = outputs
            filterval = 0
            usedfraction = (filtersrc != filterval).sum(2).sum(2).sum(1).float() /\
                            (outputs.size()[2] * outputs.size()[3])
            loss = loss * usedfraction
            
        if self.reduce:
            loss = loss.mean()
        return loss

class ScaledTorchFocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2, scale=True, reduce=True, logits=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale
        self.reduce = reduce
        self.logits = logits

    def forward(self, outputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        #Following two lines from https://catalyst-team.github.io/catalyst/ 
        pt = torch.exp(-BCE_loss)
        elementwiseloss = (1-pt)**self.gamma * BCE_loss
        batch_size = outputs.size()[0]
        loss = elementwiseloss.view(batch_size, -1).mean(1)
        #print(('focal', list(loss.detach().cpu().numpy())))
        
        if self.scale:
            filtersrc = outputs
            filterval = 0
            usedfraction = (filtersrc != filterval).sum(2).sum(2).sum(1).float() /\
                            (outputs.size()[2] * outputs.size()[3])
            loss = loss * usedfraction
        
        if self.reduce:
            loss = loss.mean()
        return loss
