import torch
from torch import nn
from project_control import BeagleParameters as bp
from PIL import Image
import numpy as np
import skimage
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os

class CompleteModel(nn.Module):
    """
    A complete model for image captioning, combining an encoder and a decoder.
    """

    def __init__(self, encoder, decoder):
        super(CompleteModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model_type = 'LSTM'

    def forward(self, images, captions, caption_lengths):
        """
        Perform a single forward pass through the model.

        Args:
            images (Tensor): Batch of input images.
            captions (Tensor): Batch of target captions.
            caption_lengths (Tensor): Lengths of each caption in the batch.

        Returns:
            predictions (Tensor): Predicted word distributions.
            alphas (Tensor): Attention weights.
        """
        encoder_out = self.encoder(images)  # Extract image features
        predictions, attention_weights = self.decoder(encoder_out, captions, caption_lengths)
        return predictions, attention_weights

    def generate_captions(self, image_paths, word_map, transform=None, max_length=None, clean_captions=False):
        """
        Generate captions for a batch of images.

        Args:
            image_paths (list[str]): List of paths to images.
            word_map (dict): Mapping from word IDs to words.
            transform (callable): Transformation function for preprocessing images.
            max_length (int): Maximum length of generated captions.
            clean_captions (bool): Whether to clean captions by removing special tokens.

        Returns:
            list[list[str]]: Generated captions for each image.
        """
        assert image_paths is not None, "Image paths must be provided."
        assert word_map is not None, "Word map must be provided."

        transform = transform or bp.get_resnet_transform(augment_dataset=False)
        max_length = max_length or bp.model_control.max_gen_length

        self.eval()  # Set the model to evaluation mode
        device = next(self.parameters()).device

        image_captions = []

        for image_path in image_paths:
            # Preprocess the image
            image = self._preprocess_image(image_path, transform, device)

            # Generate caption tokens
            caption_tokens, attention_weights = self._generate_tokens(image, max_length, clean_captions)

            # Convert tokens to words
            caption_words = [word_map[token] for token in caption_tokens]
            image_captions.append(caption_words)

        return image_captions, attention_weights

    def _preprocess_image(self, image_path, transform, device):
        """
        Preprocess a single image for input to the model.

        Args:
            image_path (str): Path to the image.
            transform (callable): Transformation function for preprocessing.
            device (torch.device): Device to move the image to.

        Returns:
            Tensor: Preprocessed image tensor.
        """
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image.to(device)

    def _generate_tokens(self, image, max_length, clean_captions):
        """
        Generate caption tokens for a single image.

        Args:
            image (Tensor): Preprocessed image tensor.
            max_length (int): Maximum length of generated captions.
            clean_captions (bool): Whether to clean captions by removing special tokens.

        Returns:
            list[int]: Generated caption tokens.
        """
        with torch.no_grad():
            # Encode the image
            features = self.encoder(image)
            batch_size, _, _, encoder_dim = features.size()

            # Flatten features if necessary
            features = features.view(batch_size, -1, encoder_dim)

            # Initialize LSTM states
            mean_encoder_out = features.mean(dim=1)
            h = self.decoder.init_h(mean_encoder_out)
            c = self.decoder.init_c(mean_encoder_out)

            # Generate tokens
            caption_tokens = []
            token = torch.tensor([[bp.model_control.SOS_token]], dtype=torch.long, device=image.device)  # Add batch dimension [1,1]

            attention_weights = []
            for _ in range(max_length):
                word_embedding = self.decoder.embedding(token)  # [1, 1, embed_dim]
                word_embedding = word_embedding.squeeze(1)  # [1, embed_dim]
                attention_weighted_encoding, alphas = self.decoder.attention(features, h)
                attention_weights.append(alphas)

                # Apply gating mechanism
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                # Update LSTM state
                h, c = self.decoder.decode_step(
                    torch.cat([word_embedding, attention_weighted_encoding], dim=1),
                    (h, c)
                )

                # Predict next word
                preds = self.decoder.fc(h)
                token = preds.argmax(dim=1)
                caption_tokens.append(token.item())

                # Stop if <EOS> token is generated
                if token.item() == bp.model_control.EOS_token:
                    break

            # Optionally clean the captions
            if clean_captions:
                caption_tokens = [token for token in caption_tokens if token > 2]

        return caption_tokens, torch.cat(attention_weights)

    def generate_saliency_maps(self, image_paths, word_map, transform=None,
                               max_length=bp.model_control.max_gen_length, attention_smoothing=False):
        """
        Generate saliency maps for a batch of images showing where the model attends at each word generation step.

        Args:
            image_paths (list[str]): List of paths to images
            word_map (dict): Mapping from word IDs to words
            transform (callable, optional): Transformation function for preprocessing images
            save_path (str, optional): Directory to save the visualization results
            max_length (int): Maximum caption length

        Returns:
            list[tuple]: List of (caption, attention_weights) for each image
                caption: list of generated words
                attention_weights: numpy array of shape (caption_length, attention_height, attention_width)
        """
        assert image_paths is not None, "Image paths must be provided."
        assert word_map is not None, "Word map must be provided."

        transform = transform or bp.get_resnet_transform(augment_dataset=False)

        self.eval()
        device = next(self.parameters()).device
        results = []

        for image_path in image_paths:
            # Preprocess image
            image = self._preprocess_image(image_path, transform, device)

            # Generate tokens and get attention weights
            tokens, attention_weights = self._generate_tokens(image, max_length, clean_captions=False)

            # Convert tokens to words
            caption = [word_map[token] for token in tokens]

            # Reshape attention weights for visualization
            # we have a square image of size image_size x image_size
            image_size = int(np.sqrt(attention_weights.size(-1)))
            num_images = attention_weights.size(0)
            attention_weights = attention_weights.view(num_images, image_size, image_size).cpu().detach().numpy()

            results.append((caption, attention_weights))

            # Save visualization
            self._save_attention_visualization(image_path, caption, attention_weights, attention_smoothing)

        return results

    def _save_attention_visualization(self, image_path, caption, attention_weights, attention_smoothing=False):
        """
        Save visualization of attention weights overlaid on the original image.

        Args:
            image_path (str): Path to the original image
            caption (list): Generated caption words
            attention_weights (numpy.ndarray): Attention weights for each word
            save_path (str): Directory to save visualizations
        """
        # Load original image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))

        # Create a figure with a single row
        # Number of subplots = original image + one for each word
        n_subplots = len(caption) + 1
        plt.figure(figsize=(4 * n_subplots, 4))

        # Plot original image first
        font_size = 30  # Increased font size
        plt.subplot(1, n_subplots, 1)
        plt.imshow(img)
        plt.text(0, 15, f'Original', color='black', backgroundcolor='white',
                 fontsize=font_size, fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 bbox=dict(facecolor='white', edgecolor='white', pad=2))
        plt.axis('off')

        # Plot attention maps
        for idx, (word, alpha) in enumerate(zip(caption, attention_weights)):
            plt.subplot(1, n_subplots, idx + 2)  # +2 because we started with original image

            # Plot focus word
            plt.text(0, 15, f'{word}', color='black', backgroundcolor='white',
                 fontsize=font_size, fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 bbox=dict(facecolor='white', edgecolor='white', pad=2))
            plt.imshow(img)

            if attention_smoothing:
                alpha = skimage.transform.pyramid_expand(alpha, upscale=16, sigma=8)
            else:
                alpha = skimage.transform.resize(alpha, [224, 224])

            plt.imshow(alpha, alpha=0.6)
            plt.set_cmap(cm.gray)
            plt.axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{os.path.splitext(image_path)[0]}_attention.png")
        plt.close()
