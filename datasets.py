## Dataset Definition
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from project_control import BeagleParameters as bp
import pandas as pd
from collections import Counter
from itertools import takewhile
import gc
import numpy as np
from itertools import islice


# BeagleTensors Dataset! We're special!
class BTDataset(Dataset):
    def __init__(self, dataset_name, split, transform=bp.get_resnet_transform(),
                 min_word_freq=2, sample_size=None, vocab_extend_list=None, vocab_extend_limit=None):
        dataset_list = bp.get_dataset_list()

        # Check dataset validity
        if dataset_name not in dataset_list.keys():
            raise ValueError(f'Dataset "{dataset_name}" must be one of {list(dataset_list.keys())}')

        # Check split validity
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Split must be one of ["train", "val", "test"]!')

        # Check extend list
        if (vocab_extend_list is not None) and (vocab_extend_list != []):
            if vocab_extend_list=='others':
                vocab_extend_list = list(dataset_list.keys())
                vocab_extend_list.remove(dataset_name)

            if dataset_name in vocab_extend_list:
                raise ValueError(f'Dataset cannot be a member of vocab_extend_list! Add only those datasets you want to extend with their vocabulary!')

        vocab_extend_limit = (vocab_extend_limit or 0)

        self.transform = transform
        self.dataset_name = dataset_name
        self.split = split
        self.word2idx = {"<PAD>": bp.model_control.PAD_token,
                         "<SOS>": bp.model_control.SOS_token,
                         "<EOS>": bp.model_control.EOS_token,
                         "<UNK>": bp.model_control.UNK_token}
        self.idx2word = None
        self.min_word_freq = min_word_freq
        self.word_counts = Counter()
        self.dataset_length = 0
        self.sample_size = sample_size

        # Set images directory
        try:
            self.images_dir = dataset_list[self.dataset_name]['image_dirs'][self.split]
        except KeyError:
            raise ValueError(f"Image directory not found for dataset '{dataset_name}' and split '{split}'")

        # Initialize split file list
        # vocab extend list must NOT include the dataset name
        try:
            karpathy_data = []
            with open(dataset_list[dataset_name]['annotations_path'], 'r', encoding='utf-8') as f:
                karpathy_data.extend(json.load(f)['images'])

            vocab_extend_data = []
            if vocab_extend_list is not None:
                for extend_dataset_name in vocab_extend_list:
                    with open(dataset_list[extend_dataset_name]['annotations_path'], 'r', encoding='utf-8') as f:
                        vocab_extend_data.extend(json.load(f)['images'])

        except FileNotFoundError:
            raise FileNotFoundError(f"Annotations file not found for dataset '{dataset_name}'")

        # Filter karpathy_data['images'] for the specified split
        filtered_images = [img for img in karpathy_data if img['split'] == self.split]
        vocab_extend_data_filtered = [img for img in vocab_extend_data if img['split'] == self.split]

        # Vectorized operations for extracting captions, tokens, and lengths
        captions_list = []
        tokens_list = []
        lengths_list = []

        if sample_size is not None:
            # filtered_images = np.random.choice(filtered_images, sample_size, replace=False)
            filtered_images = filtered_images[:sample_size]

        # Extract vocabulary before sampling
        for img in filtered_images:
            captions = [sentence['raw'] for sentence in img['sentences']]
            tokens = [sentence['tokens'] for sentence in img['sentences']]
            lengths = [len(sentence['tokens']) + 2 for sentence in img['sentences']]  # +2 for <SOS> and <EOS>

            captions_list.append(captions)
            tokens_list.append(tokens)
            lengths_list.append(lengths)

        # Extend vocabulary, if required
        extend_tokens_list = []
        if (vocab_extend_list is not None) and (vocab_extend_list != []):
            for img in vocab_extend_data_filtered:
                extend_tokens = [sentence['tokens'] for sentence in img['sentences']]
                extend_tokens_list.append(extend_tokens)

        # Create DataFrame for the filtered images
        df = pd.DataFrame(list(filtered_images))

        # Assign the lists directly to DataFrame columns
        df['captions'] = captions_list
        df['caption_tokens'] = tokens_list
        df['token_lenghts'] = lengths_list

        # Initialize `caption_picked` with zeros for each caption
        df['caption_picked'] = [np.zeros(len(tokens), dtype=int).tolist() for tokens in tokens_list]

        # Select the relevant columns
        df = df[['filename', 'captions', 'caption_tokens', 'token_lenghts', 'caption_picked']]

        # Reset index for `dataset_dict` conversion
        self.dataset_dict = df.reset_index(drop=True).to_dict(orient='index')

        ## Build word map
        # Build word counts while you are at it!
        for tokens in tokens_list:
            for token_list in tokens:
                self.word_counts.update(token_list)

        # Check extend limit, then update if necessary
        vocab_extend_limit = (vocab_extend_limit or 0)
        if (len(self.word_counts) < vocab_extend_limit) or (vocab_extend_limit == 0):
            # extend
            for extend_tokens in extend_tokens_list:
                for extend_token_list in extend_tokens:
                    self.word_counts.update(extend_token_list)

        if vocab_extend_limit == 0:
            vocab_extend_limit = len(self.word_counts)

        words_with_min_freq = dict(takewhile(lambda i: i[1] >= self.min_word_freq,
                                              self.word_counts.most_common()))

        words_with_min_freq = dict(islice(words_with_min_freq.items(), vocab_extend_limit))

        self.word2idx.update({k: i + len(self.word2idx) for i, (k, _) in enumerate(words_with_min_freq.items())})
        self.idx2word = {i: k for k, i in self.word2idx.items()}

        self.dataset_length = len(self.dataset_dict)
        self.vocab_size = len(self.word2idx)

        # Free memory
        del df, karpathy_data, dataset_list
        gc.collect()

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Load and transform image
        image_path = os.path.join(self.images_dir, self.dataset_dict[idx]['filename'])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if self.split == "val":
            # Choosed a fixed caption for validation
            caption_idx = 0
        else:
            # Pick the caption with the least number of picks
            caption_idx = np.argmin(self.dataset_dict[idx]['caption_picked'])

        self.dataset_dict[idx]['caption_picked'][caption_idx] += 1

        # Process caption
        caption_tokens = ["<SOS>"] + self.dataset_dict[idx]['caption_tokens'][caption_idx] + ["<EOS>"]

        # Convert words to indices
        caption_tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in caption_tokens]

        if self.split == 'train':
            return image, torch.tensor(caption_tokens), self.dataset_dict[idx]['token_lenghts'][caption_idx]
        else:
            # For validation of testing, also return ALL 'captions_per_image' captions to find BLEU-4 score
            return (image, torch.tensor(caption_tokens),
                    self.dataset_dict[idx]['token_lenghts'][caption_idx],
                    self.dataset_dict[idx]['caption_tokens'])

    def collate_fn(self, batch):
        ## Custom collate function for DataLoader

        # Sort batch by caption length (descending)
        batch.sort(key=lambda x: len(x[1]), reverse=True)

        if self.split == 'train':
            images, captions, lengths = zip(*batch)
        else:
            images, captions, lengths, captions_all = zip(*batch)

        # Stack images
        images = torch.stack(images, 0)

        # Pad captions to same length
        target_captions = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            target_captions[i, :end] = cap[:end]

        if self.split == 'train':
            return images, target_captions, torch.tensor(lengths)
        else:
            return images, target_captions, torch.tensor(lengths), captions_all

    def get_total_captions_picked(self):
        return np.sum([self.dataset_dict[i]['caption_picked'] for i in range(len(self.dataset_dict))])
