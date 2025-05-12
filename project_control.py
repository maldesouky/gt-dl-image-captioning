## parameters.py
##
## This file contains all fixed parameters used by any class
## or function inside this project.

import torchvision.transforms as transforms
from multiprocessing import cpu_count

# Long Live Beagle Tensors!
class BeagleParameters:

    # This static class has model control parameters
    # Feel free to modify here or set these programmatically in code

    class model_control:
        encoded_image_size = 14                             # Encoder image output
        attention_dim = 512                                 # Attention layer size
        embed_dim = 300                                     # Embedding size (300 for GloVe, 256 for Stanford)
        decoder_dim = 512                                   # Hidden layer size
        encoder_dim = 2048                                  # DO NOT CHANGE THIS - Encoded image with attention size - leave this fixed at 2048. This is ResNet's output
        dropout = 0.5                                       # Drop out ratio
        min_word_freq = 5                                   # Min. word frequency to include in word map
        max_gen_length = 30                                 # Maximum length for generating captions for test images
        PAD_token = 0                                       # Padding token
        SOS_token = 1                                       # Start of sentence token
        EOS_token = 2                                       # End of sentence token
        UNK_token = 3                                       # Unknown token
        beam_size = 5                                       # Beam search, set to 0 to disable

        # Pre-trained embeddings
        pretrained_embeddings_file = None                   # Use pre-trained embeddings?
        # pretrained_embeddings_file = \
        #     './data/glove/glove.6B.300d.txt'                # Use pre-trained embeddings?
        fine_tune_embeddings = True                         # Fine-tune embeddings? If use_pretrained_embeddings <> None, this must be True

        # Transformer-specific parameters
        num_heads = 8                                       # Number of attention heads
        num_layers = 4                                      # Number of transformer layers
        ff_dim = 2048                                       # Feed-forward network dimension
        transformer_max_gen_length = 1000                   # Maximum sequence length for transformer (must be more than the longest caption)
        transformer_embed_dim = 512                         # Embed_dim % num_heads == 0
        transformer_dropout = 0.1                           # Transformer drop-out ratio

    class train_control:
        random_seed = 42                                    # Random seed
        batch_size = 8                                     # Batch size
        # num_workers = cpu_count()                           # Number of worker threads for DataLoader, set to #CPU cores
        num_workers = 4                                     # Number of worker threads for DataLoader
        learning_rate = 1e-4                                # Initial learning rate
        min_learning_rate = 1e-5                            # Minimum learning rate
        weight_decay = 1e-2                                 # Weight decay
        momentum = 0.9                                      # Momentum (for use with SGD with Momentum)
        warmup_epochs = 6                                   # Number of warmup epochs
        epochs = 50                                         # Number of epochs
        steps = [3, 5]                                      # LR stepping control
        save_best = True                                    # Save best model?
        save_every = 1                                      # Save every n epochs - Set to None to disable
        accumulation_steps = 8                              # Gradient accumulation steps (set to 1 to disable)
        gradient_clip_norm = 5.0                            # clip gradients at an absolute value of
        early_stopping_patience = 10                        # Early stopping patience
        alpha_c = 1.0                                       # Doubly stochastic attention regularization

    __datasets = {
        'flickr8k': {
            'annotations_path': 'data/flickr8k/dataset_flickr8k.json',
            'image_dirs': {'train': 'data/flickr8k/images',
                           'val': 'data/flickr8k/images',
                           'test': 'data/flickr8k/images'},
            'train_split_tag': "train",
            'valid_split_tag': "val",
            'test_split_tag': "test"
        },
        'flickr30k': {
            'annotations_path': 'data/flickr30k/dataset_flickr30k.json',
            'image_dirs': {'train': 'data/flickr30k/flickr30k_images/flickr30k_images/flickr30k_images',
                           'val': 'data/flickr30k/flickr30k_images/flickr30k_images/flickr30k_images',
                           'test': 'data/flickr30k/flickr30k_images/flickr30k_images/flickr30k_images'},
            'train_split_tag': "train",
            'valid_split_tag': "val",
            'test_split_tag': "test"
        },
        'coco': {
            'annotations_path': 'data/coco/dataset_coco.json',
            'image_dirs': {'train': 'data/coco/train2014',
                           'val': 'data/coco/val2014',
                           'test': 'data/coco/test2014'},
            'train_split_tag': "train",
            'valid_split_tag': "val",
            'test_split_tag': "test"
        }
    }

    __resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    __resnet_transform_with_augmentation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    @classmethod
    def get_dataset_list(self):
        return self.__datasets

    @classmethod
    def get_resnet_transform(self, augment_dataset=False):
        if augment_dataset:
            return self.__resnet_transform
        else:
            return self.__resnet_transform_with_augmentation
