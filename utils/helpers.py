import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from anonymeter.neighbors.mixed_types_kneighbors import MixedTypeKNeighbors
from sklearn.manifold import TSNE

from typing import cast, Dict, Set

def print_results(results):

    print("Number of attacks:", results.n_attacks)
    print('\n')

    print("Number of successes for main attacks:", results.n_success)
    print("Successs rate of main attack:", results.attack_rate)
    print("Risk linked to the main attack: ", results.risk(baseline=False))
    print('\n')

    print("Number of successes for control attacks:", results.n_control)
    print("Successs rate of control attack:", results.control_rate)
    print('\n')

    print("Number of successes for baseline attacks:", results.n_baseline)
    print("Successs rate of baseline attack:", results.baseline_rate)
    print("Risk linked to the baseline attack: ", results.risk(baseline=True))

def _find_nn(syn: pd.DataFrame, ori: pd.DataFrame, n_jobs: int, n_neighbors: int) -> np.ndarray:
    nn = MixedTypeKNeighbors(n_jobs=n_jobs, n_neighbors=n_neighbors)

    if syn.ndim == 1:
        syn = syn.to_frame()

    if ori.ndim == 1:
        ori = ori.to_frame()

    nn.fit(syn)

    return cast(np.ndarray, nn.kneighbors(ori, return_distance=False))

def find_links(idx_0, idx_1, n_neighbors: int) -> Dict[int, Set[int]]:
    """Return synthetic records that link originals in the split datasets.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors considered for the link search.

    Returns
    -------
    Dict[int, Set[int]]
        Dictionary mapping the index of the linking synthetic record
        to the index of the linked original record.

    """

    links = {}
    for ii, (row0, row1) in enumerate(zip(idx_0, idx_1)):
        joined = set(row0[:n_neighbors]) & set(row1[:n_neighbors])
        if len(joined) > 0:
            links[ii] = joined

    return links

# use TNSE to do embedding then cluster on that
def tsne_embedding(df, n_components=2, perplexity=3):

  # preprocessing numerical
  numerical = df.select_dtypes(exclude='object')
  if not numerical.empty:   
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
    fit1 = TSNE(n_components=n_components, perplexity=perplexity, init='random').fit_transform(numerical) if numerical.shape[1]>1 else numerical.values.reshape((-1,1))
        
  # preprocessing categorical
  categorical = df.select_dtypes(include='object')
  if not categorical.empty:
    categorical = pd.get_dummies(categorical)
    fit2 = TSNE(n_components=n_components, perplexity=perplexity, init='random').fit_transform(categorical) if categorical.shape[1]>1 else categorical.values.reshape((-1,1))

  # Embedding numerical & categorical
  if not (numerical.empty or categorical.empty):
    tsne_embeddingding = fit1 * fit2
  elif numerical.empty:
     tsne_embeddingding = fit2
  elif categorical.empty:
    tsne_embeddingding = fit1
  else:
    raise ValueError("The dataframe provided has no categorical or numerical columns.")

  return tsne_embeddingding

def plot_linkability_attack(idx_a, idx_b, A, B, syn_A, syn_B, aux_cols, links):
    fig, ax = plt.subplots(2, 2, figsize=(14, 6), sharex='col', sharey='col')

    for i, (ia, ib) in enumerate(zip(idx_a, idx_b)):
        
        color = plt.cm.get_cmap('viridis')(i / len(A))
        
        ax[0][0].scatter(A[aux_cols[0][0]].iloc[i], A[aux_cols[0][1]].iloc[i], color=color, alpha=1, s=50)
        ax[0][0].scatter(syn_A[aux_cols[0][0]].iloc[ia], syn_A[aux_cols[0][1]].iloc[ia], color=color, alpha=0.1, s=50)

        ax[0][1].scatter(B[aux_cols[1][0]].iloc[i], B[aux_cols[1][1]].iloc[i], color=color, alpha=1, s=50)
        ax[0][1].scatter(syn_B[aux_cols[1][0]].iloc[ib], syn_B[aux_cols[1][1]].iloc[ib], color=color, alpha=0.1, s=50)

        if i in links.keys():
            ax[1][0].scatter(A[aux_cols[0][0]].iloc[i], A[aux_cols[0][1]].iloc[i], color=color, alpha=1, s=50)
            ax[1][0].scatter(syn_A[aux_cols[0][0]].iloc[ia], syn_A[aux_cols[0][1]].iloc[ia], color=color, alpha=0.1, s=50)

            ax[1][1].scatter(B[aux_cols[1][0]].iloc[i], B[aux_cols[1][1]].iloc[i], color=color, alpha=1, s=50)
            ax[1][1].scatter(syn_B[aux_cols[1][0]].iloc[ib], syn_B[aux_cols[1][1]].iloc[ib], color=color, alpha=0.1, s=50)

        else: continue
 
    ax[0][0].set_title('Dataset A')
    ax[0][0].set_xlabel(f'{aux_cols[0][0]}')
    ax[0][0].set_ylabel(f'{aux_cols[0][1]}')

    ax[0][1].set_title('Dataset B')
    ax[0][1].set_xlabel(f'{aux_cols[1][0]}')
    ax[0][1].set_ylabel(f'{aux_cols[1][1]}')

    ax[1][0].set_title('Dataset A - Links')
    ax[1][0].set_xlabel(f'{aux_cols[0][0]}')
    ax[1][0].set_ylabel(f'{aux_cols[0][1]}')

    ax[1][1].set_title('Dataset B - Links')
    ax[1][1].set_xlabel(f'{aux_cols[1][0]}')
    ax[1][1].set_ylabel(f'{aux_cols[1][1]}')    
        
    plt.tight_layout()
    plt.show()

def plot_linkability_embedded(embedding_A, embedding_B, embedding_syn_A, embedding_syn_B, idx_a, idx_b, original_to_plot, links):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # Plot all the synthetic data
    ax[0].scatter(embedding_syn_A[:, 0], embedding_syn_A[:, 1], color='grey', alpha=0.1, s=50)
    ax[1].scatter(embedding_syn_B[:, 0], embedding_syn_B[:, 1], color='grey', alpha=0.1, s=50)

    # Plot the original sample
    ax[0].scatter(embedding_A[original_to_plot, 0], embedding_A[original_to_plot, 1], color='orange', label='original observation', alpha=1, s=50)
    ax[1].scatter(embedding_B[original_to_plot, 0], embedding_B[original_to_plot, 1], color='orange', label='original observation', alpha=1, s=50)

    # Plot the nearest neighbors of the original sample
    ax[0].scatter(embedding_syn_A[idx_a[original_to_plot, :], 0], embedding_syn_A[idx_a[original_to_plot, :], 1], color='blue', label='synthetic nearest neighbors', alpha=0.5, s=50)
    ax[1].scatter(embedding_syn_B[idx_b[original_to_plot, :], 0], embedding_syn_B[idx_b[original_to_plot, :], 1], color='blue', label='synthetic nearest neighbors', alpha=0.5, s=50)

    # Plot the linking synthetic observation
    link_idx_a = np.where(idx_a[original_to_plot] == list(links[original_to_plot])[0])
    link_idx_b = np.where(idx_b[original_to_plot] == list(links[original_to_plot])[0])

    ax[0].scatter(embedding_syn_A[idx_a[original_to_plot, link_idx_a], 0], embedding_syn_A[idx_a[original_to_plot, link_idx_a], 1], color='green', label='linking observation', alpha=0.8, s=50)
    ax[1].scatter(embedding_syn_B[idx_b[original_to_plot, link_idx_b], 0], embedding_syn_B[idx_b[original_to_plot, link_idx_b], 1], color='green', label='linking observation', alpha=0.8, s=50)

    ax[0].legend()
    ax[0].set_title('Dataset A')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].legend()
    ax[1].set_title('Dataset B')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    # ax[0].set_xlim(-500, 500)
    # ax[0].set_ylim(-1500, 1500)

    # ax[1].set_xlim(-100, 100)
    # ax[1].set_ylim(-100, 50)

    plt.show()

def print_comaprison(results, model_names):
    
    values = []
    errors = []

    for i, attack in enumerate(results.keys()):
        for j, model in enumerate(model_names):
            values.append(results[attack][model][0].value)
            errors.append(results[attack][model][0].error)
        
        data = results[attack]

        values_main = [data[model][0].value for model in model_names]
        errors_main = [data[model][0].error for model in model_names]

    df = pd.DataFrame({
        'Attack': ['Singling Out'] * len(model_names) + ['Linkability'] * len(model_names) + ['Inference'] * len(model_names),
        'Model': len(results) * model_names,
        'Value': values,
        'Lower CI': [values[k]-errors[k] for k in range(len(values))],
        'Upper CI': [values[k]+errors[k] for k in range(len(values))]
    })

    plt.figure(figsize=(12, 6))

    barplot = sns.barplot(x='Attack', y='Value', hue='Model', data=df,)
    bar_centers = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
    plt.errorbar(x=sorted(bar_centers[:df.shape[0]]), y=values, yerr=errors, fmt='none', color='k', capsize=5)

    plt.xlabel('Attack Type')
    plt.ylabel('Succes Rate')
    plt.title('Success Rates of Attacks and 95% CIs')
    plt.legend(title='Model', bbox_to_anchor=(1, 1))

    # Mostra il grafico
    plt.show()
