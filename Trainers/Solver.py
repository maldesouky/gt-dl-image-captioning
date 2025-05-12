import torch
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from project_control import BeagleParameters as bp
from Models.Transformer import generate_square_subsequent_mask
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR
import math


class AverageMeter:
    """Keeps track of the average, current value, and count of metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics to initial state."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update metrics with a new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Solver:
    """Handles training and validation of the model."""

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", bp.train_control.batch_size)
        self.device = kwargs.get("device", "cpu")
        self.lr = kwargs.get("lr", bp.train_control.learning_rate)
        self.momentum = kwargs.get("momentum", bp.train_control.momentum)
        self.steps = kwargs.get("steps", bp.train_control.steps)
        self.epochs = kwargs.get("epochs", bp.train_control.epochs)
        self.warmup = kwargs.get("warmup_epochs", bp.train_control.warmup_epochs)
        self.gradient_clip_norm = kwargs.get("gradient_clip_norm", bp.train_control.gradient_clip_norm)
        self.accumulation_steps = kwargs.get("accumulation_steps", bp.train_control.accumulation_steps)
        self.model = kwargs.get("model", None)
        self.train_dataset = kwargs.get("train_dataset", None)

        assert self.model is not None, "A model must be provided for training."

        self.criterion = nn.CrossEntropyLoss(ignore_index=bp.model_control.PAD_token,
                                             label_smoothing=0.1)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=bp.train_control.weight_decay
        )


        # Total number of warmup steps
        steps_per_epoch = math.ceil(self.train_dataset.dataset_length / bp.train_control.batch_size)
        warmup_steps = bp.train_control.warmup_epochs * steps_per_epoch
        total_steps = bp.train_control.epochs * steps_per_epoch


        if self.train_dataset.dataset_name == "coco":
            warmup_scheduler = LinearLR(
                optimizer=self.optimizer,
                start_factor=0.1,  # Start at 10% of the base learning rate
                total_iters=warmup_steps
            )

            cosine_scheduler = CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=total_steps - warmup_steps,  # Remaining steps after warm-up
                eta_min=bp.train_control.min_learning_rate
            )

            self.scheduler = SequentialLR(
                optimizer=self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )

        elif self.train_dataset.dataset_name == "flickr30k":
            warmup_scheduler = LinearLR(
                optimizer=self.optimizer,
                start_factor=0.1,  # Start at 10% of the base learning rate
                total_iters=warmup_steps
            )

            cosine_scheduler = CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=(total_steps - warmup_steps) // 3,  # Remaining steps after warm-up
                eta_min=bp.train_control.min_learning_rate
            )

            self.scheduler = SequentialLR(
                optimizer=self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )

        elif self.train_dataset.dataset_name == "flickr8k":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=0.635,
                patience=2,
                threshold=1e-4,
                min_lr=bp.train_control.min_learning_rate,
                mode="min"
            )

            # warmup_scheduler = LinearLR(
            #     optimizer=self.optimizer,
            #     start_factor=0.1,  # Start at 10% of the base learning rate
            #     total_iters=warmup_steps
            # )

            # cosine_scheduler = CosineAnnealingLR(
            #     optimizer=self.optimizer,
            #     T_max=(total_steps - warmup_steps) // 1,  # Remaining steps after warm-up
            #     eta_min=bp.train_control.min_learning_rate
            # )

            # self.scheduler = SequentialLR(
            #     optimizer=self.optimizer,
            #     schedulers=[warmup_scheduler, cosine_scheduler],
            #     milestones=[warmup_steps]
            # )


        # Learning rate scheduler - OLD
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=self.optimizer,
        #     factor=0.5,
        #     patience=1,
        #     threshold=0.5,
        #     min_lr=bp.train_control.min_learning_rate,
        #     mode="max"
        # )

        # self.scheduler = optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=0.8
        # )

        # self.scheduler = optim.lr_scheduler.StepLR(
        #     optimizer=self.optimizer,
        #     step_size=1,
        #     gamma=0.8
        # )

        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler(self.device)

        # Best model tracking
        self.best_loss = float("inf")
        self.best_model = None

    # Generic training step
    def training_step(self, training_data, step: int):
        if self.model.model_type == "LSTM":
            return self.training_step_LSTM(training_data, step)
        elif self.model.model_type == "Transformer":
            return self.training_step_transformer(training_data, step)

    # Generic validation step
    def validation_step(self, validation_data):
        if self.model.model_type == "LSTM":
            return self.validation_step_LSTM(validation_data)
        elif self.model.model_type == "Transformer":
            return self.validation_step_transformer(validation_data)


    # Transformer training step
    def training_step_transformer(self, training_data, step: int):
        images, captions, lengths = training_data
        images, captions, lengths = images.to(self.device), captions.to(self.device), lengths.to(self.device)

        tgt_seq_len = captions.size(1) - 1
        target_mask = generate_square_subsequent_mask(tgt_seq_len, self.device).to(self.device)

        # Forward pass with mixed precision
        with torch.amp.autocast(self.device.type):
            outputs = self.model(images, captions[:, :-1], target_mask=target_mask)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))

            # Normalize loss for accumulation
            loss = loss / self.accumulation_steps

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Step optimizer and scaler every accumulation step
        if (step + 1) % self.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        # Metrics (scale back loss for reporting)
        train_loss = loss.item() * self.accumulation_steps
        train_perplexity = np.exp(train_loss)

        return train_loss, train_perplexity


    # Transformer validation step
    def validation_step_transformer(self, validation_data):
        images, captions, lengths, references = validation_data
        images, captions, lengths = images.to(self.device), captions.to(self.device), lengths.to(self.device)

        with torch.no_grad():
            # Calculate loss using teacher forcing
            tgt_seq_len = captions.size(1) - 1
            target_mask = generate_square_subsequent_mask(tgt_seq_len, self.device).to(self.device)
            output = self.model(images, captions[:, :-1], target_mask=target_mask)
            loss = self.criterion(output.view(-1, output.size(-1)), captions[:, 1:].reshape(-1))

            # Generate captions for BLEU evaluation
            # Use beam search for generation - beam size of 0 means greedy decode
            generated_captions = []
            for image in images:
                # Get encoder output and ensure correct dimensions
                encoder_out = self.model.encoder(image.unsqueeze(0))
                encoder_out = self.model.encoder_projection(encoder_out['patch_embeddings'])

                sequence = self.model.decoder.beam_search(
                                                   memory = encoder_out,
                                                   start_token = bp.model_control.SOS_token,
                                                   end_token = bp.model_control.EOS_token,
                                                   beam_size = bp.model_control.beam_size,
                                                   max_len = bp.model_control.max_gen_length
                                                )
                generated_captions.extend(sequence)

        # Metrics
        valid_loss = loss.item()
        valid_perplexity = np.exp(valid_loss)

        return valid_loss, valid_perplexity, generated_captions, references


    # LSTM training step
    def training_step_LSTM(self, training_data, step: int):
        """
        Perform a single training step.
        Args:
            training_data (tuple): A batch of training data (images, captions, lengths).
            step (int): Current step in the training loop for gradient accumulation.

        Returns:
            tuple: (training loss, perplexity)
        """
        images, captions, lengths = training_data

        # Send data to device
        images, captions, lengths = images.to(self.device), captions.to(self.device), lengths.to(self.device)

        # Forward pass with mixed precision
        with torch.amp.autocast(self.device.type):
            predictions, alphas = self.model(images, captions, lengths)
            targets = captions[:, 1:]  # Remove <SOS> token

            packed_predictions = pack_padded_sequence(predictions, [l - 1 for l in lengths], batch_first=True)[0]
            packed_targets = pack_padded_sequence(targets, [l - 1 for l in lengths], batch_first=True)[0]
            loss = self.criterion(packed_predictions, packed_targets)

        # Normalize loss for accumulation
        loss = loss / self.accumulation_steps

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Step optimizer and scaler every accumulation step
        if (step + 1) % self.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        # Metrics
        train_loss = loss.item() * self.accumulation_steps  # Scale back loss for reporting

        # Add doubly stochastic attention regularization
        loss += bp.train_control.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        train_perplexity = np.exp(train_loss)

        return train_loss, train_perplexity

    # LSTM validation step
    def validation_step_LSTM(self, validation_data):
        """
        Perform a single validation step.
        Args:
            validation_data (tuple): A batch of validation data (images, captions, lengths, caption_tokens_all).

        Returns:
            tuple: (validation loss, perplexity, predictions, references)
        """
        images, captions, lengths, references = validation_data

        # Send data to device
        images, captions, lengths = images.to(self.device), captions.to(self.device), lengths.to(self.device)

        with torch.no_grad():
            # Forward pass with mixed precision for loss computation
            with torch.amp.autocast(self.device.type):
                predictions, _ = self.model(images, captions, lengths)
                targets = captions[:, 1:]  # Remove <SOS> token

                packed_predictions = pack_padded_sequence(predictions, [l - 1 for l in lengths], batch_first=True)[0]
                packed_targets = pack_padded_sequence(targets, [l - 1 for l in lengths], batch_first=True)[0]
                loss = self.criterion(packed_predictions, packed_targets)

            # Metrics
            valid_loss = loss.item()
            valid_perplexity = np.exp(valid_loss)

            if (bp.model_control.beam_size == 0) or (bp.model_control.beam_size is None):
                return valid_loss, valid_perplexity, predictions.argmax(dim=-1).tolist(), references
            else:
                # Generate predictions using beam search
                encoder_out = self.model.encoder(images)
                batch_predictions = []
                for i in range(encoder_out.size(0)):  # For each image in batch
                    single_image_encoder_out = encoder_out[i:i+1]
                    predicted_sequence = self.model.decoder.beam_search(
                        single_image_encoder_out,
                        beam_size=bp.model_control.beam_size,
                        max_length=bp.model_control.max_gen_length
                    )
                    batch_predictions.append(predicted_sequence)

            return valid_loss, valid_perplexity, batch_predictions, references
