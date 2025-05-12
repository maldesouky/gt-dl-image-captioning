from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import csv
import torch
import os
from project_control import BeagleParameters as bp


# Compute corpus BLEU score
def compute_corpus_bleu(corpus_references_hypotheses, batch_size=250):
    """
    Compute corpus-level BLEU scores with support for batch processing.

    Args:
        corpus_references_hypotheses (tuple): (references, hypotheses) for BLEU scoring.
        batch_size (int): Batch size for multiprocessing.

    Returns:
        dict: BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    """
    references, hypotheses = corpus_references_hypotheses
    smoothing = SmoothingFunction()

    # For small datasets, use single-process computation
    if len(references) < batch_size:
        scores = _compute_bleu_weights(references, hypotheses, smoothing)
        return dict(zip(['bleu1', 'bleu2', 'bleu3', 'bleu4'], scores))

    # For larger datasets, use multiprocessing
    batches = []
    for i in range(0, len(references), batch_size):
        batch_refs = references[i:i + batch_size]
        batch_hyps = hypotheses[i:i + batch_size]
        batches.append((batch_refs, batch_hyps))

    # Use multiprocessing pool
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(_compute_batch_bleu,
                             [(batch_refs, batch_hyps, smoothing) for batch_refs, batch_hyps in batches])

    # Average results across batches
    final_scores = np.mean(results, axis=0)
    return dict(zip(['bleu1', 'bleu2', 'bleu3', 'bleu4'], final_scores))

def _compute_batch_bleu(references, hypotheses, smoothing):
    return _compute_bleu_weights(references, hypotheses, smoothing)

def _compute_bleu_weights(references, hypotheses, smoothing):
    weights = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25)
    ]

    return [100 * corpus_bleu(references, hypotheses,
                              weights=w, smoothing_function=smoothing.method1)
                              for w in weights]


# def load_checkpoint(checkpoint_path, encoder, decoder, complete_model):
#     """
#     Load a model checkpoint, resizing mismatched layers if necessary.

#     Args:
#         checkpoint_path (str): Path to the checkpoint.
#         encoder, decoder, complete_model: Model components.

#     Returns:
#         tuple: (epoch, step, best_lr_step_criterion_value, bleu_scores)
#     """
#     if not os.path.exists(checkpoint_path):
#         print(f"[Checkpoint] No checkpoint found at {checkpoint_path}. Starting fresh.")
#         return 1, 1, float('inf'), {'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0}

#     print(f"[Checkpoint] Loading checkpoint from {checkpoint_path}...")
#     checkpoint = torch.load(checkpoint_path)

#     # Load encoder state
#     try:
#         encoder_state = filter_state_dict(checkpoint["encoder_state_dict"], encoder)
#         encoder.load_state_dict(encoder_state, strict=True)
#     except RuntimeError as e:
#         print(f"[Error] Failed to load encoder state: {e}")
#         raise

#     # Handle decoder state resizing
#     decoder_state = checkpoint["decoder_state_dict"]
#     _resize_state(decoder_state, "attention.encoder_att.weight", decoder.attention.encoder_att.weight)
#     _resize_state(decoder_state, "attention.encoder_att.bias", decoder.attention.encoder_att.bias)
#     _resize_state(decoder_state, "attention.decoder_att.weight", decoder.attention.decoder_att.weight)
#     _resize_state(decoder_state, "attention.decoder_att.bias", decoder.attention.decoder_att.bias)
#     _resize_state(decoder_state, "attention.full_att.weight", decoder.attention.full_att.weight)
#     _resize_state(decoder_state, "embedding.weight", decoder.embedding.weight)
#     _resize_state(decoder_state, "decode_step.weight_ih", decoder.decode_step.weight_ih)
#     _resize_state(decoder_state, "fc.weight", decoder.fc.weight, axis=0)
#     _resize_state(decoder_state, "fc.bias", decoder.fc.bias)

#     decoder.load_state_dict(decoder_state, strict=False)

#     # Handle complete model state resizing
#     complete_model_state = checkpoint["complete_model_state_dict"]
#     _resize_state(complete_model_state, "decoder.attention.encoder_att.weight", complete_model.decoder.attention.encoder_att.weight)
#     _resize_state(complete_model_state, "decoder.attention.encoder_att.bias", complete_model.decoder.attention.encoder_att.bias)
#     _resize_state(complete_model_state, "decoder.attention.decoder_att.weight", complete_model.decoder.attention.decoder_att.weight)
#     _resize_state(complete_model_state, "decoder.attention.decoder_att.bias", complete_model.decoder.attention.decoder_att.bias)
#     _resize_state(complete_model_state, "decoder.attention.full_att.weight", complete_model.decoder.attention.full_att.weight)
#     _resize_state(complete_model_state, "decoder.embedding.weight", complete_model.decoder.embedding.weight)
#     _resize_state(complete_model_state, "decoder.decode_step.weight_ih", complete_model.decoder.decode_step.weight_ih)
#     _resize_state(complete_model_state, "decoder.fc.weight", complete_model.decoder.fc.weight, axis=0)
#     _resize_state(complete_model_state, "decoder.fc.bias", complete_model.decoder.fc.bias)

#     complete_model.load_state_dict(complete_model_state, strict=False)

#     # Handle missing keys with defaults
#     best_lr_step_criterion_value = checkpoint.get("best_lr_step_criterion_value", float('inf'))
#     bleu_scores = checkpoint.get("bleu_scores", {'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0})

#     return (
#         checkpoint["epoch"] + 1,
#         checkpoint["step"],
#         best_lr_step_criterion_value,
#         bleu_scores
#     )


# def _resize_state(state_dict, key, target_tensor, axis=None):
#     """
#     Resize a tensor in the state dictionary to match the target tensor's shape.

#     Args:
#         state_dict (dict): Model state dictionary.
#         key (str): Key of the tensor to resize.
#         target_tensor (torch.Tensor): Target tensor for resizing.
#         axis (int): Axis along which to resize (None for full resizing).
#     """
#     if key in state_dict:
#         src_tensor = state_dict[key]
#         target_shape = target_tensor.shape

#         if src_tensor.shape != target_shape:
#             print(f"[Resize] Resizing {key} from {src_tensor.shape} to {target_shape}.")
#             new_tensor = torch.zeros_like(target_tensor)
#             slices = tuple(slice(0, min(s, t)) for s, t in zip(src_tensor.shape, target_shape))
#             new_tensor[slices] = src_tensor[slices]
#             state_dict[key] = new_tensor


# def filter_state_dict(state_dict, model):
#     """
#     Filters the state_dict to match the model's keys.

#     Args:
#         state_dict (dict): State dictionary from checkpoint.
#         model (torch.nn.Module): Current model.

#     Returns:
#         dict: Filtered state dictionary.
#     """
#     model_keys = set(model.state_dict().keys())
#     return {k: v for k, v in state_dict.items() if k in model_keys}


def get_class_variables(class_object):
    return {k: v for k, v in class_object.__dict__.items() if not k.startswith('__') and not callable(v)}


def update_class_variables(class_object, new_values_dict):
    for k, v in new_values_dict.items():
        setattr(class_object, k, v)


def load_checkpoint(checkpoint_path):
    try:
        print(f"[Checkpoint] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    except:
        print(f"[Checkpoint] Failed to load checkpoint from {checkpoint_path}!")
        return None


# def save_checkpoint(checkpoint_path, encoder, decoder, complete_model, epoch, step, best_lr_step_criterion_value, bleu_scores, train_dataset):
def save_checkpoint(checkpoint_path, checkpoint):
    """
    Save the model checkpoint.

    """

    torch.save(checkpoint, checkpoint_path)
    print(f"[Checkpoint] Checkpoint saved at {checkpoint_path}.")


def load_embeddings(embeddings_file, word_map):
    """
    Load pre-trained embeddings and create a tensor aligned with the word map.

    Args:
        embeddings_file (str): Path to embeddings file (e.g., GloVe).
        word_map (dict): Word-to-index mapping.

    Returns:
        torch.Tensor: Embeddings tensor.
    """
    print("[Embeddings] Loading embeddings...")
    vocab = set(word_map.keys())

    embed_df = pd.read_table(
        embeddings_file, sep=' ', header=None, quoting=csv.QUOTE_NONE, names=['word'] + list(range(1, 301))
    )
    embed_df = embed_df[embed_df['word'].isin(vocab)]

    # Create embeddings matrix
    embed_matrix = np.zeros((len(word_map), embed_df.shape[1] - 1), dtype=np.float32)

    for _, row in embed_df.iterrows():
        if row['word'] in word_map:
            embed_matrix[word_map[row['word']]] = row[1:].values

    print(f"[Embeddings] For {len(word_map)} vocabularies, {len(word_map) - embed_df.shape[0]} are missing from pre-trained embeddings!")

    return torch.tensor(embed_matrix, dtype=torch.float32)

def get_changed_parameters(heading, params_old, params_new):
    # This function assumes that the dictionaries have the same keys
    keys = set(params_old.keys()).union(set(params_new.keys()))
    different_params = {}

    tabulated_string = '-' * 55 + '\n'
    tabulated_string += "{:<30} {:<10} {:<10}".format(' Changed ' + heading + ' Parameter', 'Current', 'Checkpoint') + '\n'
    tabulated_string += '-' * 55 + '\n'
    for key in keys:
        # Check for inequality, considering missing keys
        if params_old.get(key) != params_new.get(key):
            old_value = params_old.get(key) or 'N/A'
            new_value = params_new.get(key) or 'N/A'

            different_params[key] = {' Current': old_value, 'Loaded': new_value}
            tabulated_string += " {:<29} {:<10} {:<10}".format(key, old_value, new_value) + '\n'

    tabulated_string += '-' * 55 + '\n'
    return different_params, tabulated_string
