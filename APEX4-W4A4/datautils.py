"""
Data loading utilities for various text datasets.
Provides functions to load and preprocess datasets like WikiText-2, PTB, C4, and RedPajama.
Each function returns preprocessed data loaders suitable for model training and evaluation.
"""

from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import random
from tqdm import trange
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)


def get_wikitext2(nsamples, seed, seqlen, model):
    """
    Prepares data loaders for the Wikitext-2 dataset for training and testing.

    Args:
        nsamples (int): Number of random samples to generate from the training data.
        seed (int): Seed for random number generator to ensure reproducibility.
        seqlen (int): Sequence length for each input sample.
        model (str): Pretrained model identifier used for tokenization.

    Returns:
        tuple: A tuple containing the training loader and tokenized test data.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_wikitext2")
    
    # Load datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Initialize tokenizer and encode data
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate random samples for training
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    """
    Load and prepare the Penn Treebank (PTB) dataset for training and validation.

    Args:
        nsamples (int): The number of samples to generate for the training loader.
        seed (int): The seed value for random number generation, ensuring reproducibility.
        seqlen (int): The sequence length of each sample.
        model (str): The model identifier used to load a pre-trained tokenizer.

    Returns:
        tuple: A tuple containing the training loader and tokenized validation data.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_ptb")
    
    # Load datasets
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    # Initialize tokenizer and encode data
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    # Generate random samples for training
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    """
    Loads and preprocesses the C4 dataset for training and validation.

    Args:
        nsamples (int): Number of samples to generate for training.
        seed (int): Random seed for reproducibility.
        seqlen (int): The sequence length for each training sample.
        model (str): Model identifier for tokenizer initialization.

    Returns:
        tuple: A tuple containing training data loader and validation data tensor.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_c4")
    
    # Load datasets
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    # Generate training samples
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen + 2:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Generate validation samples
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    """
    Generates training and testing data loaders for the Penn Treebank dataset using a specified model tokenizer.

    Args:
        nsamples (int): Number of samples to generate in the training loader.
        seed (int): Random seed for reproducibility of sample selection.
        seqlen (int): Sequence length of each sample in the training data.
        model (str): Model identifier for the tokenizer.

    Returns:
        tuple: A tuple containing the training loader and tokenized test data.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_ptb_new")
    
    # Load datasets
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    # Initialize tokenizer and encode data
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    # Generate random samples for training
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    """
    Load and preprocess training and validation datasets from C4 dataset.

    Args:
        nsamples (int): Number of samples to process for training data.
        seed (int): Random seed for reproducibility.
        seqlen (int): Length of each input/output sequence.
        model (str): Model identifier for the tokenizer.

    Returns:
        tuple: Contains training loader and validation encoded tensor.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_c4_new")
    
    # Load datasets
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation'
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    # Generate training samples
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation data
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    return trainloader, valenc


def get_red_pajama(nsamples, seed, seqlen, model):
    """
    Load and preprocess RedPajama dataset for training.

    Args:
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        seqlen (int): Length of each sequence.
        model (str): Model identifier for tokenizer.

    Returns:
        tuple: Training loader and None (no validation set).
    """
    print("Loading red_pajama from togethercomputer/RedPajama-Data-1T-Sample")
    
    # Load dataset and tokenizer
    traindata = load_dataset(
        "togethercomputer/RedPajama-Data-1T-Sample",
        split="train",
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    
    # Generate training samples
    trainloader = []
    for _ in trange(nsamples, desc="Making red_pajama calibration set", leave=False):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    return trainloader, None


def get_chinese_c4(nsamples, seed, seqlen, model):
    """
    Load and preprocess Chinese C4 dataset for training.

    Args:
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        seqlen (int): Length of each sequence.
        model (str): Model identifier for tokenizer.

    Returns:
        tuple: Training loader and None (no validation set).
    """
    print("Loading Chinese_C4_Calibration from shjwudp/chinese-c4")
    
    # Load dataset and tokenizer
    traindata = load_dataset(
        "shjwudp/chinese-c4",
        split="train",
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    
    # Generate training samples
    trainloader = []
    for _ in trange(nsamples, desc="Making chinese c4 calibration set", leave=False):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    return trainloader, None


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    """
    Retrieve appropriate data loaders based on dataset name.

    Args:
        name (str): Dataset name (wikitext2, ptb, c4, etc.).
        nsamples (int): Number of samples to retrieve.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for samples.
        model (str): Model specification for preprocessing.

    Returns:
        tuple: Appropriate data loader for the specified dataset.

    Raises:
        ValueError: If dataset name is not supported.
    """
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    elif 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    elif 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    elif 'redpajama' in name:
        return get_red_pajama(nsamples, seed, seqlen, model)
    elif 'mix_chinese_redpa' in name:
        english, _ = get_red_pajama(nsamples//2, seed, seqlen, model)
        chinese, _ = get_chinese_c4(nsamples//2, seed, seqlen, model)
        return english + chinese, None
    
    raise ValueError(
        f"Only support wikitext2, c4, c4_new, ptb, ptb_new currently, but get {name}"
    )
