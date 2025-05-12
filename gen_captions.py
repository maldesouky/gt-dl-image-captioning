## Imports
import os
from pprint import pprint
import pandas as pd
from pathlib import Path
import warnings

# PyTorch
import torch

def caption_images_from_checkpoint(checkpoint_path, images_list, export_csv_file_name=None):
    print(f"[Checkpoint] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)

    word_map = checkpoint["word_map"]
    complete_model = checkpoint["model_training_suite"].model
    image_captions, _ = complete_model.generate_captions(images_list, word_map, clean_captions=True)

    for i in range(len(image_captions)):
        image_captions[i] = ' '.join(image_captions[i])

    image_captions_dict = dict(zip(images_list, image_captions))

    # Export to CSV if desired
    if export_csv_file_name is not None:
        pd.DataFrame.from_dict(image_captions_dict, orient='index').to_csv(export_csv_file_name)

    return image_captions_dict

def pretty_print_captions(captions_dict):
    print()
    for (filename, caption) in captions_dict.items():
        print(f"{filename}:")
        print(f"   {caption}\n")
    print()

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorboard")

    checkpoint_path = "./checkpoints/coco_LSTM_glove.6B.300d_best_model.pth"
    # checkpoint_path = "./checkpoints/flickr30k_LSTM_glove.6B.300d_best_model.pth"
    images_list = sorted([os.path.join('./test_examples/caption', f) for f in os.listdir('./test_examples/caption') if os.path.isfile(os.path.join('./test_examples/caption', f))])
    # export_csv_file_name = './test_examples/output/' + Path(checkpoint_path).stem + '.csv'
    images_list = ['test_examples/attention/Show, Attend, and Tell 03.jpg']

    captions = caption_images_from_checkpoint(checkpoint_path, images_list, export_csv_file_name=None)

    pretty_print_captions(captions)
