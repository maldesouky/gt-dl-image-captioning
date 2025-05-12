## Imports
import warnings
import os
from pprint import pprint
import pandas as pd

# Project control
from project_control import BeagleParameters as bp

# PyTorch
import torch

# Model
from Models.CompleteModel import CompleteModel
from Models.Decoder import DecoderWithAttention
from Models.Encoder import Encoder
from Trainers.Solver import Solver
from datasets import BTDataset

# Utils
from utils import get_class_variables, update_class_variables


def convert_checkpoint(dataset_name, checkpoint_path, output_path, device='cpu'):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Saved checkpoint model parameters

    # Initialize models
    my_dataset = BTDataset(
        dataset_name=dataset_name,
        split='train',
        transform=bp.get_resnet_transform(augment_dataset=True),
        min_word_freq=bp.model_control.min_word_freq,
        sample_size=None
    )
    dataset_dict = my_dataset.dataset_dict

    encoder = Encoder(encoded_image_size=bp.model_control.encoded_image_size).to(device)
    decoder = DecoderWithAttention(
        attention_dim=bp.model_control.attention_dim,
        embed_dim=bp.model_control.embed_dim,
        decoder_dim=bp.model_control.decoder_dim,
        vocab_size=checkpoint['vocab_size'],
        encoder_dim=bp.model_control.encoder_dim,
        dropout=bp.model_control.dropout
    ).to(device)
    complete_model = CompleteModel(encoder, decoder).to(device)
    training_suite = Solver(
        batch_size=bp.train_control.batch_size,
        device=device,
        lr=bp.train_control.learning_rate,
        epochs=bp.train_control.epochs,
        model=complete_model
    )

    # torch.save({
    #     'word_map': train_dataset.idx2word,
    #     'vocab_size': train_dataset.vocab_size,
    #     'encoder_state_dict': encoder.state_dict(),
    #     'decoder_state_dict': decoder.state_dict(),
    #     'complete_model_state_dict': complete_model.state_dict(),
    #     'epoch': epoch,
    #     'step': step,
    #     'best_lr_step_criterion_value': best_lr_step_criterion_value,
    #     'bleu_scores': bleu_scores
    # }, path)

    # Load model states
    word_map = checkpoint['word_map']
    vocab_size = checkpoint['vocab_size']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    complete_model.load_state_dict(checkpoint['complete_model_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    best_lr_step_criterion_value = checkpoint['best_lr_step_criterion_value']
    bleu_scores = checkpoint['bleu_scores']

    # Add new variables
    model_control_params = get_class_variables(bp.model_control)
    train_control_params = get_class_variables(bp.train_control)

    model_optimizer = training_suite.optimizer
    model_optimizer_scheduler = training_suite.scheduler
    model_training_suite = training_suite

    # Add new values to checkpoint
    # checkpoint['dataset_name'] = dataset_name
    # checkpoint['dataset_dict'] = dataset_dict
    checkpoint["model_training_suite"] = model_training_suite
    # checkpoint['encoder'] = encoder
    # checkpoint['decoder'] = decoder
    # checkpoint['complete_model'] = complete_model
    # checkpoint['model_optimizer'] = model_optimizer
    # checkpoint['model_optimizer_scheduler'] = model_optimizer_scheduler
    # checkpoint['model_control_params'] = model_control_params
    # checkpoint['train_control_params'] = train_control_params
    # checkpoint['model_type'] = 'LSTM'

    # Save modified checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Converted checkpoint saved to: {output_path}")
    pass

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorboard")

    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using %s\n" % device)
    device='cpu'

    convert_checkpoint(dataset_name    ='flickr30k',
                       checkpoint_path ='checkpoints/flickr30k_best_model-20241207-1021.pth',
                       output_path     ='checkpoints/flickr30k_best_model-20241207-1021-new.pth',
                       device          =device)
