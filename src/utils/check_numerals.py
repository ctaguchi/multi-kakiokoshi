# see if there's unwanted numeral characters in the dataset
import argparse
import pandas as pd
import re
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload segmented dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        required=True,
        help="Language code of the dataset",
    )
    return parser.parse_args() 


if __name__ == "__main__":
    args = get_args()
    tsv_path = os.path.join("data",
                            "mcv-sps-st-09-2025",
                            args.language,
                            f"ss-corpus-{args.language}.tsv")
    print(f"Checking TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["transcription"]) # sometimes there are missing transcriptions
    for idx, row in df.iterrows():
        text = row["transcription"]
        # print(text)
        if re.search(r'\d', text):
            print(f"Row {idx} (audio {row['audio_file']}) contains numeral characters: {text}")
    print("Done.")