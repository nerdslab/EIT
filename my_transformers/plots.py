import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module = "matplotlib\..*" )


def plt_tiny(original, recon, amount=3):
    """side-by-side plt, 2 cols, [amount] rows"""

    oriA = original.detach().cpu()
    reconA = recon.detach().cpu()

    fig, axes = plt.subplots(ncols=2, nrows=amount, figsize=(2, amount))
    for i in range(amount):
        oriA_i = oriA[i]
        reconA_i = reconA[i]

        oriA_i = torch.permute(oriA_i, (1, 2, 0))
        reconA_i = torch.permute(reconA_i, (1, 2, 0))

        axes[i, 0].imshow(oriA_i)
        axes[i, 1].imshow(reconA_i)

    return fig


def plt_recon(original, recon, amount=3):
    """side-by-side plt, 4 cols, [amount] rows"""

    oriA, oriB = original["A"].detach().cpu(), original["B"].detach().cpu()
    reconA, reconB = recon["A"].detach().cpu(), recon["B"].detach().cpu()

    fig, axes = plt.subplots(ncols=4, nrows=amount, figsize=(4, amount))
    for i in range(amount):
        oriA_i, oriB_i = oriA[i], oriB[i]
        reconA_i, reconB_i = reconA[i], reconB[i]

        oriA_i = torch.permute(oriA_i, (1, 2, 0))
        oriB_i = torch.permute(oriB_i, (1, 2, 0))
        reconA_i = torch.permute(reconA_i, (1, 2, 0))
        reconB_i = torch.permute(reconB_i, (1, 2, 0))

        axes[i, 0].imshow(oriA_i)
        axes[i, 1].imshow(reconA_i)
        axes[i, 2].imshow(oriB_i)
        axes[i, 3].imshow(reconB_i)

    return fig


def plt_trans(original, recon, trans, amount=3):
    """ 6 cols [oriA, reconA, transA, oriB, reconB, transB], [amount] rows
    all input data in pairs pls"""
    oriA, oriB = original["A"].detach().cpu(), original["B"].detach().cpu()
    reconA, reconB = recon["A"].detach().cpu(), recon["B"].detach().cpu()
    transA, transB = trans["A"].detach().cpu(), trans["B"].detach().cpu()

    fig, axes = plt.subplots(ncols=6, nrows=amount, figsize=(6, amount))
    for i in range(amount):
        oriA_i, oriB_i = oriA[i], oriB[i]
        reconA_i, reconB_i = reconA[i], reconB[i]
        transA_i, transB_i = transA[i], transB[i]

        oriA_i = torch.permute(oriA_i, (1, 2, 0))
        oriB_i = torch.permute(oriB_i, (1, 2, 0))
        reconA_i = torch.permute(reconA_i, (1, 2, 0))
        reconB_i = torch.permute(reconB_i, (1, 2, 0))
        transA_i = torch.permute(transA_i, (1, 2, 0))
        transB_i = torch.permute(transB_i, (1, 2, 0))

        axes[i, 0].imshow(oriA_i)
        axes[i, 1].imshow(reconA_i)
        axes[i, 2].imshow(transA_i)
        axes[i, 3].imshow(oriB_i)
        axes[i, 4].imshow(reconB_i)
        axes[i, 5].imshow(transB_i)

    return fig