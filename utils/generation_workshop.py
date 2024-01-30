import argparse
import sys
sys.path.append('/home/azureuser/cloudfiles/code/Users/francesca/Synthetic-Data-Evaluation/uk_anonym/single/code')
import warnings
from code.GaussianCopula import sample_GaussianCopula
from code.CTGAN import sample_CTGAN
from code.DPCTGAN import sample_DPCTGAN
from code.synthcity.src.privbayes import sample_PrivBayes

from sdv.metadata import SingleTableMetadata

import os
import time
import pandas as pd

import subprocess

warnings.filterwarnings("ignore")

# %%
# Defining input parameters
parser = argparse.ArgumentParser()

parser.add_argument("--person_ref", type=bool, default=False, help="include the patient id in the dataset")
parser.add_argument("--num_synthetic_samples", type=int, default=15000, help="number of synthetic samples to generate")

parser.add_argument("--model", type=str, default='all', 
                    help="What model to reproduce: 'all', 'GaussianCopula', 'CTGAN', 'CopulaGAN', 'REaLTabFormer', 'MCMedGAN', 'ADSGAN', 'PrivBayes ")

parser.add_argument("--diffpriv", type=bool, default=False, help="enable differential privacy guarantees")

args = parser.parse_args()

num_synthetic_samples = args.num_synthetic_samples
person_ref = args.person_ref
model = args.model
diffpriv = args.diffpriv
opacus_version = '1.4.0' if model=='ADSGAN' else '0.14.0' 
opacus_installation = f'pip install -q opacus=={opacus_version}'

# %%
# Load the handmade datasets in pandas dataframes
datapath = '/home/azureuser/cloudfiles/code/Users/francesca/Synthetic-Data-Evaluation/uk_anonym/workshop/fifa.csv'
df = pd.read_csv(datapath)
original_df = df.iloc[:15000, :]
control_df = df.iloc[15000:, :]

original_metadata = SingleTableMetadata()
original_metadata.detect_from_dataframe(data=original_df)

epsilon = 3.0

if model=='all' or model=='GaussianCopula':
    # GaussianCopula
    print('#################### Gaussian Copula ####################')
    start_time = time.time()
    original_synthetic_GaussianCopula = sample_GaussianCopula(dataframe=original_df, metadata=original_metadata, n_samples=num_synthetic_samples)
    end_time = time.time()
    GaussianCopula_time = end_time - start_time
    print(f'GaussianCopula: {GaussianCopula_time} seconds')

if model=='all' or model=='CTGAN':
    # CTGAN
    print('#################### CTGAN ####################')
    start_time = time.time()
    original_synthetic_CTGAN = sample_CTGAN(dataframe=original_df, metadata=original_metadata, n_samples=num_synthetic_samples)
    end_time = time.time()
    CTGAN_time = end_time - start_time
    print(f'CTGAN: {CTGAN_time} seconds')

if model=='all' or model=='DPCTGAN':
    # DPCTGAN
    print('#################### DPCTGAN ####################')
    subprocess.run(opacus_installation, shell=True)
    start_time = time.time()
    original_synthetic_DPCTGAN = sample_DPCTGAN(dataframe=original_df, n_samples=num_synthetic_samples, epsilon=epsilon)
    end_time = time.time()
    DPCTGAN_time = end_time - start_time
    print(f'DPCTGAN: {DPCTGAN_time} seconds')

if model=='all' or model=='PrivBayes':
    # PrivBayes
    print('#################### PrivBayes ####################')
    plugin = 'privbayes'
    start_time = time.time()
    original_synthetic_PrivBayes = sample_PrivBayes(dataframe=original_df, num_synthetic_samples=num_synthetic_samples, plugin=plugin)
    end_time = time.time()
    PrivBayes_time = end_time - start_time
    print(f'PrivBayes: {PrivBayes_time} seconds')

# %%
if model == 'all':
    print(f'GaussianCopula: {GaussianCopula_time} seconds')
    print(f'CTGAN: {CTGAN_time} seconds')
    print(f'DPCTGAN: {DPCTGAN_time} seconds')
    print(f'PrivBayes: {PrivBayes_time} seconds')
    