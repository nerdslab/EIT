import torch

class Pepper:
    r"""Adds a constant to the neuron firing rate with a probability of :obj:`p`. The firing rate vector needs to have
        already been normalized.
    .. note::
        If more than one tensor is given, the same dropout pattern will be applied to all.
    Args:
        p (float, Optional): Probability of adding pepper. (default: :obj:`0.5`)
        apply_p (float, Optional): Probability of applying the transformation. (default: :obj:`1.0`)
        sigma (float, Optional): Constant to be added to neural activity. (default: :obj:`1.0`)
    """
    def __init__(self, p=0.5, sigma=1.0, apply_p=1.):
        self.p = p
        self.sigma = sigma
        self.apply_p = apply_p

    def __call__(self, x):
        keep_mask = torch.rand(x.shape) < self.p
        random_pepper = self.sigma * keep_mask
        apply_mask = torch.rand(x.shape) < self.apply_p
        random_pepper = random_pepper * apply_mask
        return x + random_pepper.to(x)


class neuron_augmentor(object):
    def __init__(self):
        self.aug_list = []
    def augment(self, x):
        for aug in self.aug_list:
            x = aug(x)
        return x
