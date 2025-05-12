import torch
from torch import nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from project_control import BeagleParameters as bp

def generate_square_subsequent_mask(size, device):
    """Optimized mask generation"""
    if not hasattr(generate_square_subsequent_mask, "masks"):
        generate_square_subsequent_mask.masks = {}

    key = (size, device)
    if key not in generate_square_subsequent_mask.masks:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        generate_square_subsequent_mask.masks[key] = mask

    return generate_square_subsequent_mask.masks[key]

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        """
        Positional encoding to inject sequence order information into embeddings.

        Args:
            embed_dim (int): Embedding dimension.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.pe = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.01)  # Reduced magnitude

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: Positional encoded embeddings.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim = bp.model_control.transformer_embed_dim,
                 num_heads = bp.model_control.num_heads,
                 num_layers = bp.model_control.num_layers,
                 ff_dim = bp.model_control.ff_dim,
                 dropout = bp.model_control.transformer_dropout,
                 max_len = bp.model_control.transformer_max_gen_length,
                 pretrained_embeddings=None
                ):
        """
        Transformer-based decoder.

        Args:
            embed_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            ff_dim (int): Dimensionality of feed-forward layer.
            vocab_size (int): Size of the output vocabulary.
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length.
            pretrained_embeddings (torch.Tensor, optional): Pretrained embeddings for the token embedding layer.
        """
        super(TransformerDecoder, self).__init__()

        # Embedding layers
        if pretrained_embeddings is not None:
            self.token_embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        # Pre-layer normalization
        self.pre_norm = nn.LayerNorm(embed_dim)

        # Learned temperature parameter with constraints
        self.temperature = nn.Parameter(torch.clamp(torch.ones(1) * 0.07, min=0.01, max=1.0))

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            norm_first=True  # Pre-normalization
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final output projection
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights using Xavier initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if not hasattr(self.token_embedding, 'weight') or not self.token_embedding.weight.requires_grad:
            nn.init.xavier_uniform_(self.token_embedding.weight)

        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)  # Adjusted gain for output layer

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, target_seq, memory, target_mask=None, memory_mask=None):
        """
        Forward pass for the Transformer decoder.

        Args:
            target_seq (torch.Tensor): Input token sequence of shape (batch_size, tgt_seq_len).
            memory (torch.Tensor): Encoder outputs of shape (batch_size, src_seq_len, embed_dim).
            target_mask (torch.Tensor, optional): Mask for target sequence (for causal masking).
            memory_mask (torch.Tensor, optional): Mask for encoder-decoder attention.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, tgt_seq_len, vocab_size).
        """
        target_emb = self.token_embedding(target_seq)
        target_emb = self.positional_encoding(target_emb)
        target_emb = self.dropout(target_emb)
        target_emb = self.pre_norm(target_emb)
        target_emb = target_emb * torch.exp(self.temperature)

        output = self.transformer_decoder(
            tgt=target_emb.permute(1, 0, 2),
            memory=memory.permute(1, 0, 2),
            tgt_mask=target_mask,
            memory_mask=memory_mask,
        )

        output = self.fc_out(output.permute(1, 0, 2))
        return output

    def beam_search(self,
                    memory,
                    start_token = bp.model_control.SOS_token,
                    end_token = bp.model_control.EOS_token,
                    beam_size = bp.model_control.beam_size,
                    max_len = bp.model_control.max_gen_length # max seq length
                    ):
        """
        Beam search for sequence generation. Defaults to greedy decoding if beam_size is 0.

        Args:
            memory (torch.Tensor): Encoder outputs of shape (batch_size, src_seq_len, embed_dim).
            start_token (int): Start token ID.
            end_token (int): End token ID.
            beam_size (int): Number of beams. If 0, performs greedy decoding.
            max_len (int): Maximum sequence length.

        Returns:
            List[List[int]]: Generated sequences for each batch.
        """
        if beam_size == 0:
            return self.greedy_decode(memory, start_token, end_token, max_len)

        device = memory.device
        batch_size = memory.size(0)
        vocab_size = self.fc_out.out_features

        # Initialize beams
        sequences = torch.full((batch_size * beam_size, 1), start_token, dtype=torch.long, device=device)
        scores = torch.zeros(batch_size * beam_size, device=device)
        memory = memory.repeat_interleave(beam_size, dim=0)

        for step in range(max_len):
            seq_len = sequences.size(1)
            target_mask = generate_square_subsequent_mask(seq_len, device)

            logits = self.forward(sequences, memory, target_mask=target_mask)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            scores = scores.unsqueeze(1) + log_probs
            scores = scores.view(batch_size, -1)

            top_scores, top_indices = scores.topk(beam_size, dim=-1)

            # Update sequences
            prev_seq_indices = top_indices // vocab_size
            next_tokens = top_indices % vocab_size

            sequences = torch.cat([
                sequences[prev_seq_indices.view(-1)],
                next_tokens.view(-1, 1)
            ], dim=-1)

            scores = top_scores.view(-1)

            # Check for end tokens
            if (next_tokens == end_token).all():
                break

        # Extract best sequences
        final_seqs = []
        for i in range(batch_size):
            best_seq = sequences[i * beam_size:(i + 1) * beam_size][0].tolist()
            final_seqs.append(best_seq)

        return final_seqs

    def greedy_decode(self, memory, start_token, end_token, max_len):
        """
        Greedy decoding for sequence generation.

        Args:
            memory (torch.Tensor): Encoder outputs of shape (batch_size, src_seq_len, embed_dim).
            start_token (int): Start token ID.
            end_token (int): End token ID.
            max_len (int): Maximum sequence length.

        Returns:
            List[List[int]]: Generated sequences for each batch.
        """
        device = memory.device
        batch_size = memory.size(0)
        sequences = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        for step in range(max_len):
            target_mask = generate_square_subsequent_mask(sequences.size(1), device)
            logits = self.forward(sequences, memory, target_mask=target_mask)
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sequences = torch.cat([sequences, next_tokens], dim=1)

            if (next_tokens == end_token).all():
                break

        return sequences.tolist()

class ImageCaptioningModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim = bp.model_control.transformer_embed_dim,
                 num_heads = bp.model_control.num_heads,
                 num_layers = bp.model_control.num_layers,
                 ff_dim = bp.model_control.ff_dim,
                 dropout = bp.model_control.transformer_dropout,
                 max_len = bp.model_control.transformer_max_gen_length,
                 device='cpu',
                 pretrained_embeddings=None):

        super(ImageCaptioningModel, self).__init__()

        self.model_type = 'Transformer'
        self.max_len = max_len
        self.device = device

        # Load ViT model
        vit = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
        return_nodes = {'encoder.ln': 'patch_embeddings'}
        self.encoder = create_feature_extractor(vit, return_nodes=return_nodes)
        self.encoder_projection = nn.Linear(768, embed_dim)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            vocab_size=vocab_size,
            dropout=dropout,
            max_len=max_len,
            pretrained_embeddings=pretrained_embeddings
        )

    def forward(self, images, captions, target_mask=None):
        features = self.encoder(images)['patch_embeddings']
        cls_token = features[:, 0:1, :]  # Include CLS token
        patch_embeddings = features[:, 1:, :]
        memory = self.encoder_projection(torch.cat([cls_token, patch_embeddings], dim=1))
        memory = self.encoder_norm(memory)

        if target_mask is None:
            target_mask = generate_square_subsequent_mask(captions.size(1), self.device)

        logits = self.decoder(captions, memory, target_mask=target_mask)
        return logits
