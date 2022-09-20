import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt_theme = 'PCA'

def pca_latent(train_dataloader):
    '''
    pca
    '''

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(1 * 6, 1 * 6)) # each col is a model, each row is a type of figure
    if plt_theme == "PCA":
        pca = PCA(n_components=2)
    elif plt_theme == "TSNE":
        pca = TSNE(n_components=2)
    else:
        raise NotImplementedError

    train_data = []
    train_label = []
    for data, label in train_dataloader:
        train_data.append(data.detach().cpu())
        train_label.append(label.detach().cpu())

    train_data = torch.cat(train_data)
    train_label = torch.cat(train_label)

    # pca_i fit output_i
    output = pca.fit_transform(train_data)


    scatter1 = axes.scatter(output[:, 0], output[:, 1], c=train_label, cmap="tab10")
    legend1 = axes.legend(*scatter1.legend_elements(), loc="lower left", title="Classes")
    axes.add_artist(legend1)

    return fig

def neuron_visual():
    cmap = ['tab:blue', 'tab:orange',
            'tab:green', 'tab:red',
            'tab:purple', 'tab:brown',
            'tab:pink', 'tab:gray']


