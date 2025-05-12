import torch
from torch import nn
from project_control import BeagleParameters as bp
import numpy as np

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (Batch, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (Batch, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (Batch, num_pixels)
        alpha = self.softmax(att)  # (Batch, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (Batch, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, pretrained_embeddings=None,
                 encoder_dim=bp.model_control.encoder_dim, dropout=bp.model_control.dropout):

        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Moved here to match the tutorial's decoder implementation
        # self.dropout_layer = nn.Dropout(p=self.dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.init_weights()

        # Fine-tune embeddings if specified
        # Must be True if NOT using pre-trained embeddings
        if pretrained_embeddings is not None:
            # Update embedding parameters
            self.embedding.weight = nn.Parameter(pretrained_embeddings)

            # if fine_tune_embeddings:
            #     for p in self.embedding.parameters():
            #         p.requires_grad = fine_tune_embeddings


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        nn.init.xavier_uniform_(self.embedding.weight)
        # bias = np.sqrt(3.0 / self.embed_dim)
        # self.embedding.weight = nn.Parameter(pretrained_embeddings)

        self.fc.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)


    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # Flatten image
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(captions)  # (Batch, max_caption_length, embed_dim)
        # print(f"Embeddings shape: {embeddings.shape}, First timestep embeddings: {embeddings[:, 0, :5]}")

        h, c = self.init_hidden_state(encoder_out)  # (Batch, decoder_dim)
        # print(f"Initial Hidden State h: {h[:2, :]}, c: {c[:2, :]}")

        decode_lengths = (caption_lengths - 1).tolist()  # Exclude <EOS> token

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])  # Get batch size for this timestep

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            # print(f"Attention Weighted Encoding (t={t}): {attention_weighted_encoding[:2, :]}")
            # print(f"Attention Weights (t={t}): {alpha[:2, :]}")

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout_layer(h))  # Get prediction (logits) for each token
            # print(f"Logits at timestep {t}: {preds[:2, :5]}")

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha  # Store attention weights

        return predictions, alphas


    def beam_search(self, encoder_out, beam_size=bp.model_control.beam_size, max_length=bp.model_control.max_gen_length):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Pre-compute device and initialize tensors
        device = encoder_out.device
        sos_token = torch.LongTensor([bp.model_control.SOS_token]).to(device)

        # Initialize beam state
        k_prev_words = sos_token.repeat(beam_size)
        seqs = sos_token.repeat(beam_size, 1)
        top_k_scores = torch.zeros(beam_size, 1).to(device)

        # Initialize model states
        encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)
        h, c = self.init_hidden_state(encoder_out)

        # Track completed sequences
        complete_seqs = []
        complete_seqs_scores = []

        for step in range(max_length):
            embeddings = self.embedding(k_prev_words)

            # Batch process attention and decoder
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )

            scores = self.fc(self.dropout_layer(h))
            scores = torch.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step > 0:
                scores = scores.view(-1)

            # Get top k scores and words
            top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)
            prev_word_inds = top_k_words // self.vocab_size
            next_word_inds = top_k_words % self.vocab_size

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            # Check which sequences are complete
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) 
                            if next_word != bp.model_control.EOS_token]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            # Break if all beams are complete
            beam_size = len(incomplete_inds)
            if beam_size == 0:
                break

            # Update states for next step
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds]

        # Handle case where no complete sequences found
        if not complete_seqs:
            return seqs[0].tolist()

        # Return sequence with highest score
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        return complete_seqs[i]
