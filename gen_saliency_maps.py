import torch
import warnings

def generate_saliency_maps_from_checkpoint(checkpoint_path, image_paths, attention_smoothing, device):
    # Load the trained model
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model with the saved weights
    model = checkpoint['model_training_suite'].model.to(device)

    # Load word map
    word_map = checkpoint['word_map']

    # Generate saliency maps
    results = model.generate_saliency_maps(
        image_paths=image_paths,
        word_map=word_map,
        attention_smoothing=attention_smoothing
    )

    # Print generated captions
    for image_path, (caption, _) in zip(image_paths, results):
        print(f"\nImage: {image_path}")
        print(f"Generated caption: {' '.join(caption)}")

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorboard")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'checkpoints/coco_LSTM_glove.6B.300d_best_model.pth'
    image_paths = ["test_examples/attention/Show, Attend, and Tell 01.jpg",
                   "test_examples/attention/Show, Attend, and Tell 02.jpg",
                   "test_examples/attention/Show, Attend, and Tell 03.jpg",
                   "test_examples/attention/Show, Attend, and Tell 04.jpg",
                   "test_examples/attention/Show, Attend, and Tell 05.jpg",
                   "test_examples/attention/Show, Attend, and Tell 06.jpg"]

    generate_saliency_maps_from_checkpoint(checkpoint_path = checkpoint_path,
                                           image_paths= image_paths,
                                           attention_smoothing=False,
                                           device = device)
