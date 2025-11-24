from datasets import load_from_disk
import argparse
import os


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Update a prepared dataset to Hugging Face Hub."
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,  
        help="Path to the local dataset directory.",
    )
    parser.add_argument(
        "-n",
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to push to Hugging Face Hub.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Load the dataset from the specified local path
    ds = load_from_disk(args.dataset_path)
    
    # Push the dataset to Hugging Face Hub
    if args.dataset_name is None:
        dataset_name = os.path.basename(args.dataset_path)
    else:
        dataset_name = args.dataset_name
    ds.push_to_hub(
        repo_id=f"{dataset_name}",
        private=True, # Rehost in public is not allowed for Common Voice
    )
    
    print(f"Dataset pushed to Hugging Face Hub from {args.dataset_path}.")