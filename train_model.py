import os
from pathlib import Path
import warnings
import gc
from tqdm import tqdm

# Data and BLEU Scoring
from datasets import BTDataset
import nltk
import shutil

# Utils
from utils import (load_checkpoint, save_checkpoint, load_embeddings,
                   get_class_variables, update_class_variables, get_changed_parameters)

# PyTorch
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Project Modules
from utils import compute_corpus_bleu
from project_control import BeagleParameters as bp
from Models.CompleteModel import CompleteModel
from Models.Decoder import DecoderWithAttention
from Models.Encoder import Encoder
from Trainers.Solver import Solver, AverageMeter
from tensorboard import program
from Models.Transformer import PositionalEncoding, TransformerDecoder, ImageCaptioningModel

# Ensure necessary downloads
nltk.download('punkt', quiet=True)

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="tensorboard")
warnings.simplefilter(action='ignore', category=FutureWarning)


def initialize_device():
    """Initialize and return the computing device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    return device


def prepare_datasets(dataset_name, train_sample_size=None, valid_sample_size=None):
    """Create training and validation datasets."""
    train_dataset = BTDataset(
        dataset_name=dataset_name,
        split='train',
        transform=bp.get_resnet_transform(augment_dataset=True),
        min_word_freq=bp.model_control.min_word_freq,
        sample_size=train_sample_size,
        vocab_extend_list=None,
        vocab_extend_limit=None
    )
    val_dataset = BTDataset(
        dataset_name=dataset_name,
        split='val',
        transform=bp.get_resnet_transform(augment_dataset=False),
        min_word_freq=bp.model_control.min_word_freq,
        sample_size=valid_sample_size,
        vocab_extend_list=None,
        vocab_extend_limit=None
    )
    return train_dataset, val_dataset


def initialize_model(train_dataset, device, model_type='LSTM', pretrained_embeddings_file=None):
    """Initialize and return encoder, decoder, and combined model."""

    pretrained_embeddings = None
    pretrained_embeddings_dim = bp.model_control.embed_dim
    if pretrained_embeddings_file is not None:
        pretrained_embeddings = load_embeddings(pretrained_embeddings_file, train_dataset.word2idx).to(device)
        pretrained_embeddings_dim = pretrained_embeddings.size(1)
        # torch.cuda.empty_cache()

    if model_type == 'LSTM':
        encoder = Encoder(encoded_image_size=bp.model_control.encoded_image_size).to(device)

        print()

        decoder = DecoderWithAttention(
            attention_dim=bp.model_control.attention_dim,
            embed_dim=pretrained_embeddings_dim,
            decoder_dim=bp.model_control.decoder_dim,
            vocab_size=train_dataset.vocab_size,
            pretrained_embeddings=pretrained_embeddings,
            encoder_dim=bp.model_control.encoder_dim,
            dropout=bp.model_control.dropout
        ).to(device)
        lstm_model = CompleteModel(encoder, decoder).to(device)
        # lstm_model = torch.compile(lstm_model)
        return lstm_model

    elif model_type == 'Transformer':
        # Initialize Model
        assert (bp.model_control.transformer_embed_dim % bp.model_control.num_heads == 0
                ), f"transformer_embed_dim ({bp.model_control.transformer_embed_dim}) must be " + \
                   f"divisible by num_heads ({bp.model_control.num_heads})"

        transformer_model = ImageCaptioningModel(
            vocab_size=train_dataset.vocab_size,
            embed_dim=bp.model_control.transformer_embed_dim,
            num_heads=bp.model_control.num_heads,
            num_layers=bp.model_control.num_layers,
            ff_dim=bp.model_control.ff_dim,
            dropout=bp.model_control.transformer_dropout,
            max_len=bp.model_control.transformer_max_gen_length,
            device=device,
            # pretrained_embeddings=pretrained_embeddings
            pretrained_embeddings=pretrained_embeddings
        ).to(device)
        # transformer_model = torch.compile(transformer_model)

        return transformer_model


def train_and_evaluate(training_suite, train_loader, val_loader, writer, best_checkpoint_path, recurrent_checkpoint_path,
                       start_epoch, start_step, best_lr_step_criterion_value, bleu_scores):
    """
    Main training and evaluation loop with TensorBoard logs saved under the dataset name.
    """
    train_step = start_step
    early_stopping_patience = bp.train_control.early_stopping_patience  # Early stopping patience
    early_stopping_counter = 0

    # Already done by training pipeline
    # gc.collect()
    # torch.cuda.empty_cache()

    bleu_scores = bleu_scores

    if start_epoch > 1:
        print(f"[Training] Resuming training from epoch {start_epoch}...")

    for epoch in range(start_epoch, training_suite.epochs + 1):

        # Training phase
        print(f"\nEpoch: {epoch}, LR: {training_suite.scheduler.get_last_lr()[0]:.6f}, ES Counter: {early_stopping_counter}")
        train_loss, train_perplexity = AverageMeter(), AverageMeter()

        training_suite.model.train()
        train_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for idx, data in train_tqdm:
            loss, perplexity = training_suite.training_step(data, idx)
            train_loss.update(loss)
            train_perplexity.update(perplexity)
            train_step += 1

            # Adjust learning rate - for use with consine annealing with warmup
            if isinstance(training_suite.scheduler, torch.optim.lr_scheduler.SequentialLR):
                # Log learning rate
                writer.add_scalar('Batch Metrics/Learning Rate',
                                  training_suite.scheduler.get_last_lr()[0], train_step)

                training_suite.scheduler.step()

            # Log batch-level metrics
            writer.add_scalar(f'Batch Metrics/Training Loss', loss, train_step)
            # writer.add_scalar(f'Batch Metrics/Training Perplexity', perplexity, train_step)

        print(f"[Training] Epoch {epoch}: Avg Loss = {train_loss.avg:.4f}, Avg Perplexity = {train_perplexity.avg:.4f}")

        # Validation phase
        valid_loss, valid_perplexity = AverageMeter(), AverageMeter()
        references, hypotheses = [], []

        training_suite.model.eval()

        valid_tqdm = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        for idx, data in valid_tqdm:
            loss, perplexity, preds, refs = training_suite.validation_step(data)
            valid_loss.update(loss)
            valid_perplexity.update(perplexity)
            references.extend(refs)

            # Convert predictions from token IDs to words
            for pred in preds:
                caption_words = [train_loader.dataset.idx2word[token] for token in pred if token > 2]  # Remove special tokens
                hypotheses.append(caption_words)

        # Adjust learning rate - for use with on plateau scheduler
        if isinstance(training_suite.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            training_suite.scheduler.step(valid_loss.avg)

            # Log learning rate
            writer.add_scalar('Batch Metrics/Learning Rate',
                            training_suite.scheduler.get_last_lr()[0], epoch)


        # # Adjust learning rate - for use with on plateau scheduler
        # if isinstance(training_suite.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #     training_suite.scheduler.step(valid_loss.avg)

        # Calculate BLEU scores
        bleu_scores = compute_corpus_bleu((references, hypotheses))
        print(f"[Validation] Epoch {epoch}: Avg Loss = {valid_loss.avg:.4f}, Avg Perplexity = {valid_perplexity.avg:.4f}, " +
              f"BLEU-1: {bleu_scores['bleu1']:.2f}, BLEU-2: {bleu_scores['bleu2']:.2f}, " +
              f"BLEU-3: {bleu_scores['bleu3']:.2f}, BLEU-4: {bleu_scores['bleu4']:.2f}")

        # writer.add_scalar('Average Metrics/Learning Rate',
        #                   training_suite.scheduler.get_last_lr()[0], epoch)

        # Adjust learning rate
        best_model_judge = bleu_scores['bleu4']
        # best_lr_judge = valid_perplexity.avg
        # training_suite.scheduler.step()
        # training_suite.scheduler.step(best_lr_judge)

        # Log epoch-level metrics
        writer.add_scalars(f'Average Metrics/Loss', {
            'Train': train_loss.avg,
            'Validation': valid_loss.avg
        }, epoch)

        writer.add_scalars(f'Average Metrics/Perplexity', {
            'Train': train_perplexity.avg,
            'Validation': valid_perplexity.avg
        }, epoch)

        writer.add_scalars(f'Average Metrics/BLEU', {
            'BLEU-1': bleu_scores['bleu1'],
            'BLEU-2': bleu_scores['bleu2'],
            'BLEU-3': bleu_scores['bleu3'],
            'BLEU-4': bleu_scores['bleu4']
        }, epoch)

        if (bp.train_control.save_every is not None) or (bp.train_control.save_best):
            checkpoint = {
                'dataset_name': train_loader.dataset.dataset_name,
                'dataset_dict': train_loader.dataset.dataset_dict,
                'word_map': train_loader.dataset.idx2word,
                'vocab_size': train_loader.dataset.vocab_size,
                'model_training_suite': training_suite,
                'model_control_params': get_class_variables(bp.model_control),
                'train_control_params': get_class_variables(bp.train_control),

                # BEGIN: These should be removed, but are left for backward compatibility
                'encoder_state_dict': training_suite.model.encoder.state_dict(),
                'decoder_state_dict': training_suite.model.decoder.state_dict(),
                'complete_model_state_dict': training_suite.model.state_dict(),
                # END of backwards compatibility comment

                'epoch': epoch,
                'step': train_step,
                'best_lr_step_criterion_value': best_lr_step_criterion_value,
                'bleu_scores': bleu_scores
            }

            # Save recurrent model checkpoint
            if (epoch % bp.train_control.save_every == 0):
                print(f"[Checkpoint] Saving model checkpoint at epoch {epoch}...")
                save_checkpoint(recurrent_checkpoint_path, checkpoint)

            # Save best model checkpoint
            if best_model_judge > best_lr_step_criterion_value and bp.train_control.save_best:
                print(f"[Checkpoint] New best model found! Saving...")
                save_checkpoint(best_checkpoint_path, checkpoint)

                best_lr_step_criterion_value = best_model_judge
                early_stopping_counter = 0  # Reset patience counter
            else:
                early_stopping_counter += 1  # Increment patience counter

        # Early stopping check
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs!")
            break

    print("\nTraining and evaluation complete!")


def reset_tensorboard_logs(logdir):
    if os.path.exists(logdir):
        shutil.rmtree(logdir)  # Delete the entire directory
        print(f"[TensorBoard] TensorBoard log directory '{logdir}' has been reset.")
    else:
        print(f"[TensorBoard] Log directory '{logdir}' does not exist.")


def setup_tensorboard(dataset_name, model_type, suffix, reset=False):
    """Initialize and clean TensorBoard directory."""
    suffix = ('_' + suffix) if (suffix != "") else ""

    tensorboard_path = f"tensorboard/{dataset_name}_{model_type}{suffix}"
    if reset:
        reset_tensorboard_logs(tensorboard_path)  # Reset logs if specified
    os.makedirs(tensorboard_path, exist_ok=True)
    return SummaryWriter(tensorboard_path)


def run_training_pipeline(dataset_name, batch_size=None, continue_checkpoint=None, train_sample_size=None, valid_sample_size=None,
                          best_checkpoint_path=None, recurrent_checkpoint_path=None, model_type='LSTM'):
    """
    Function to execute the training and evaluation pipeline.
    Also sets up and launches TensorBoard for monitoring.

    Args:
        batch_size (int, optional): Batch size for training and validation. If None, uses the default from `bp.train_control.batch_size`.
        reset (bool, optional): Whether to reset TensorBoard logs. Default is False.
    """

    assert (best_checkpoint_path is not None), "Best checkpoint path must be provided!"

    if bp.train_control.save_every is not None:
        assert (recurrent_checkpoint_path is not None), "Recurrent checkpoint path must be provided, or disable save_every (set to None)!"

    pretrained_embeddings_file = bp.model_control.pretrained_embeddings_file

    gc.collect()
    torch.cuda.empty_cache()

    # Initialization
    device = initialize_device()

    start_fresh = (continue_checkpoint is None)
    if continue_checkpoint is not None:
        checkpoint = load_checkpoint(continue_checkpoint)

        if checkpoint is None:
            print(f"[Checkpoint] Failed to load checkpoint from {continue_checkpoint}! Starting new training...")
            continue_checkpoint = None
            start_fresh = True
        else:
            # Verify model type
            if checkpoint['model_training_suite'].model.model_type != model_type:
                print(f"[Checkpoint] Model type mismatch! Expected {model_type}, but found " +
                      f"{checkpoint['model_training_suite'].model.model_type}. Starting new training...")
                continue_checkpoint = None
                start_fresh = True

        # Load checkpoint for either LSTM or Transformer models
        if not start_fresh:
            # Load checkpoint
            train_dataset, val_dataset = prepare_datasets(dataset_name,
                                                          train_sample_size=train_sample_size,
                                                          valid_sample_size=valid_sample_size)

            training_suite = checkpoint['model_training_suite']

            # Update Model Control parameters
            changed_params, tabulated_string = get_changed_parameters('Model', get_class_variables(bp.model_control), checkpoint['model_control_params'])
            if changed_params != {}:
                print(tabulated_string)
                update_class_variables(bp.model_control, changed_params)
            else:
                print("[Checkpoint] No changes to model parameters!")

            # Update Training Control parameters
            changed_params, tabulated_string = get_changed_parameters('Training', get_class_variables(bp.train_control), checkpoint['train_control_params'])
            if changed_params != {}:
                print(tabulated_string)
                update_class_variables(bp.train_control, changed_params)
            else:
                print("[Checkpoint] No changes to training parameters!")

            # encoder = training_suite.model.encoder
            # decoder = training_suite.model.decoder
            complete_model = training_suite.model

            start_epoch = checkpoint['epoch'] + 1
            start_step = checkpoint['step']
            best_lr_step_criterion_value = checkpoint['best_lr_step_criterion_value']
            bleu_scores = checkpoint['bleu_scores']

    # Do not continue training, start new training
    else:
        # Start new training
        start_fresh = True

    if start_fresh:
        start_epoch, start_step, best_lr_step_criterion_value, bleu_scores = 1, 0, 0, {'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0}
        train_dataset, val_dataset = prepare_datasets(dataset_name,
                                                        train_sample_size=train_sample_size,
                                                        valid_sample_size=valid_sample_size)
        complete_model = initialize_model(train_dataset, device, model_type, pretrained_embeddings_file)

        # Solver
        training_suite = Solver(
            batch_size=batch_size,
            device=device,
            lr=bp.train_control.learning_rate,
            epochs=bp.train_control.epochs,
            model=complete_model,
            train_dataset=train_dataset
        )

    # Set dynamic batch size or default
    batch_size = batch_size or bp.train_control.batch_size

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=bp.train_control.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=bp.train_control.num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # TensorBoard setup with optional reset
    reset_tb = (continue_checkpoint is None)
    pretrained_embeddings_file_suffix = Path(pretrained_embeddings_file).stem if (pretrained_embeddings_file is not None) else ""
    writer = setup_tensorboard(dataset_name, model_type, pretrained_embeddings_file_suffix, reset=reset_tb)

    # TensorBoard launch
    tb_working_dir = os.getcwd() + os.sep + "tensorboard"
    tb = program.TensorBoard()
    tb_args = ['', '--logdir', tb_working_dir, '--reload_interval', '60', '--window_title', 'BeagleTensors Performance Monitor']
    tb.configure(argv=tb_args)
    url = tb.launch()
    print(f"\n[TensorBoard] TensorBoard listening on {url}")

    # Training loop
    try:
        train_and_evaluate(
            training_suite,
            train_loader,
            val_loader,
            writer,
            best_checkpoint_path,
            recurrent_checkpoint_path,
            start_epoch,
            start_step,
            best_lr_step_criterion_value,
            bleu_scores
        )
    finally:
        # Clean up
        writer.close()
        print("\nTraining complete! TensorBoard session is still active.")

    # Wrap up TensorBoard
    tb_args[-1] = '"' + tb_args[-1] + '"'
    print(f"\n[TensorBoard] TensorBoard session terminated! To restart, use:\n"
          f'tensorboard --logdir="{tb_working_dir}" --reload_interval=60')


if __name__ == '__main__':
    # run_training_pipeline(dataset_name='flickr8k', batch_size=64, continue_training=False, model_type='Transformer')

    dataset_name = 'flickr8k'
    model_type = 'Transformer'
    pretrained_embeddings_file_suffix = '_' + Path(bp.model_control.pretrained_embeddings_file).stem if (bp.model_control.pretrained_embeddings_file is not None) else ""

    best_checkpoint_path = f"checkpoints/{dataset_name}_{model_type}{pretrained_embeddings_file_suffix}_best_model.pth"
    recurrent_checkpoint_path = f"checkpoints/{dataset_name}_{model_type}{pretrained_embeddings_file_suffix}_recurrent_model.pth"

    continue_checkpoint = best_checkpoint_path

    run_training_pipeline(dataset_name=dataset_name,
                          batch_size=bp.train_control.batch_size,
                          continue_checkpoint=None,
                          train_sample_size=None,
                          valid_sample_size=None,
                          best_checkpoint_path=best_checkpoint_path,
                          recurrent_checkpoint_path=recurrent_checkpoint_path,
                          model_type=model_type)
