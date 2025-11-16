import pandas as pd
from torch.utils.data import Dataset, random_split
import torch
import torchaudio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor
)
from functools import partial
from dataclasses import dataclass

from typing import Dict, Any, Union, Optional, List, Tuple
import os
import argparse
import glob
import json


LANG2ID_FILE = "../utils/lang2id.csv"
LANG2ID_DF = pd.read_csv(LANG2ID_FILE, header=0)
IDS = LANG2ID_DF.id.tolist()
LANGS = LANG2ID_DF.lang.tolist()
ISOS = LANG2ID_DF.iso.tolist()

LANG2ID = dict(zip(LANGS, IDS))
LANG2ISO = dict(zip(LANGS, ISOS))
ISO2ID = dict(zip(ISOS, IDS))
ID2LANG = dict(zip(IDS, LANGS))
ID2ISO = dict(zip(IDS, ISOS))
ISO2LANG = dict(zip(ISOS, LANGS))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="If set, run monolingual training for each language separately.",
    )
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        choices=ISOS,
        help="ISO code of the language to train on (only for monolingual training)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training.",
    )
    return parser.parse_args()


def load_data(isos: List[str],
              data_dir: str = "../mcv-sps-st-09-2025") -> Tuple[List[pd.DataFrame], Dict[str, int]]:
    """Load audio data and transcriptions from tsv files.
    Args:
        isos: list of ISO language codes
    Returns:
    """
    dfs: List[pd.DataFrame] = []
    iso2vocab_size = dict()
    for iso in isos:
        iso_data_dir = os.path.join(data_dir, f"sps-corpus-1.0-2025-09-05-{iso}")
        audio_dir = os.path.join(iso_data_dir, "audio")
        audio_files = glob.glob(
            os.path.join(audio_dir, f"spontaneous-speech-{iso}*")
        )
        tsv_file = os.path.join(iso_data_dir, f"spontaneous-speech-{iso}_*.tsv")
        
        assert audio_files, f"No {iso} audio files found"
        assert os.path.exists(tsv_file), f"No {iso} tsv file found"
        
        audio_files = [os.path.basename(f) for f in audio_files]
        
        df = pd.read_csv(tsv_file, sep="\t")
        df = df[df["audio_file"].isin(audio_files)]
        df = df.dropna(subset=["transcription"])
        dfs.append(df)
        
        vocab = get_vocab(df)
        print(f"{iso} vocab size: {len(vocab)}")
        with open(f"{iso}_vocab.json", "w") as f:
            f.write(json.dumps(vocab))
        
        iso2vocab_size[iso] = len(vocab)

    return dfs, iso2vocab_size


class MultilingualASRDataset(Dataset):
    def __init__(self,
                 tsv: Union[str, pd.DataFrame, List[str], List[pd.DataFrame]],
                 lang2id: Optional[Dict[str, int]] = None,
                 audio_root: Optional[str] = None,
                 target_sample_rate: int = 16000):
        """
        Args:
            tsv_path: path to the tsv file(s) or dataframe(s)
            processor: Wav2Vec2Processor (i.e. feature_extractor + tokenizer)
            lang2id: dict mapping strings -> integer IDs
            audio_root: optional root directory to prepend to 'path' column
            target_sample_rate: resample all audio to this rate. 16000 for wav2vec2.
        """
        if isinstance(tsv, list):
            if isinstance(tsv[0], str):
                self.df = pd.concat([pd.read_csv(f, sep="\t") for f in tsv])
            elif isinstance(tsv[0], pd.DataFrame):
                self.df = pd.concat(tsv)
        else:
            if isinstance(tsv, str):
                self.df = pd.read_csv(tsv, sep="\t")
            elif isinstance(tsv, pd.DataFrame):
                self.df = tsv
            else:
                raise ValueError("tsv must be a string(s) or a dataframe(s)")
            
        self.df = self.df.dropna(subset=["transcription"])
        # self.processor = processor
        self.lang2id = lang2id
        self.audio_root = audio_root
        self.target_sample_rate = target_sample_rate

        if target_sample_rate != 16000:
            raise ValueError("Only 16000 supported for now")
        
    def __len__(self):
        """Get the number of samples"""
        return len(self.df)
    
    def __getitem__(self,
                    idx: int) -> Dict[str, Any]:
        """Get the idx-th sample"""
        row = self.df.iloc[idx]

        # Load audio
        rel_path = row["audio_file"]
        audio_path = (
            os.path.join(self.audio_root, rel_path)
            if self.audio_root is not None else rel_path
        )
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Force mono; average channels if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # we want shape (samples,), float32
        speech = waveform.squeeze(0).numpy()

        # text + language
        text = row["transcription"]
        lang = row["language"]
        lang_id = self.lang2id[lang]

        return {
            "speech": speech,
            "text": text,
            "lang_id": lang_id
        }
        

def get_vocab(tsv: Union[str, pd.DataFrame, List[str], List[pd.DataFrame]]):
    """Get the custom vocabulary."""
    if isinstance(tsv, list):
        if isinstance(tsv[0], str):
            df = pd.concat([pd.read_csv(f, sep="\t") for f in tsv])
        elif isinstance(tsv[0], pd.DataFrame):
            df = pd.concat(tsv)
    else:
        if isinstance(tsv, str):
            df = pd.read_csv(tsv, sep="\t")
        elif isinstance(tsv, pd.DataFrame):
            df = tsv
        else:
            raise ValueError("tsv must be a string(s) or a dataframe(s)")

    df = df.dropna(subset=["transcription"])
    
    sents = df["transcription"].tolist()
    vocab = dict()
    for sent in sents:
        for char in sent:
            if char not in vocab:
                vocab[char] = len(vocab)
    vocab["|"] = len(vocab) # word delimiter token
    return vocab


def ctc_collate_fn_multi(features: List[dict],
                         feature_extractor: Wav2Vec2FeatureExtractor,
                         tokenizer_map: Dict[str, Wav2Vec2CTCTokenizer]):
    """Collate function for multilingual CTC training."""
    speeches = [f["speech"] for f in features]
    texts = [f["text"] for f in features]
    lang_ids = torch.tensor([f["lang_id"] for f in features], dtype=torch.long)
    langs = [ID2LANG[l.item()] for l in lang_ids]
    isos = [ID2ISO[l.item()] for l in lang_ids]

    # Audio
    audio_inputs = feature_extractor(
        speeches,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True
    )
    input_values = audio_inputs["input_values"]

    if "attention_mask" in audio_inputs:
        attention_mask = audio_inputs["attention_mask"]
    else:
        attention_mask = (input_values != 0).long()

    input_lengths = attention_mask.sum(dim=1) # (B,)

    # Text
    labels_per_example: List[torch.Tensor] = []
    label_lengths: List[int] = []

    for text, iso in zip(texts, isos):
        tokenizer = tokenizer_map[iso]
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        labels_per_example.append(input_ids)
        label_lengths.append(len(input_ids))
    
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "input_lengths": input_lengths,
        "labels_list": labels_per_example,
        "label_lengths": label_lengths,
        "lang_ids": lang_ids,
        "langs": langs,
        "isos": isos,
        "texts": texts
    }
    

def ctc_collate_fn_mono(features: List[dict],
                        feature_extractor: Wav2Vec2FeatureExtractor,
                        tokenizer: Wav2Vec2CTCTokenizer):
    """Collate function for monolingual CTC training."""
    speeches = [f["speech"] for f in features]
    texts = [f["text"] for f in features]

    # Audio
    audio_inputs = feature_extractor(
        speeches,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True
    )
    input_values = audio_inputs["input_values"]

    if "attention_mask" in audio_inputs:
        attention_mask = audio_inputs["attention_mask"]
    else:
        attention_mask = (input_values != 0).long()

    input_lengths = attention_mask.sum(dim=1) # (B,)

    # Text
    labels_per_example: List[torch.Tensor] = []
    label_lengths: List[int] = []

    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        labels_per_example.append(input_ids)
        label_lengths.append(len(input_ids))

    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "input_lengths": input_lengths,
        "labels_list": labels_per_example,
        "label_lengths": label_lengths
    }
    
    
@dataclass
class DataCollatorCTCWithPadding:
    """
    Off-the-shelf data collator used in Hugging Face examples.
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set, it will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self,
                 features: List[Feature]) -> Dict[str, torch.Tensor]:
        """Split inputs and labels since they have to be of different lengths
        and need different padding methods
        """
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        label_texts = [feature["labels"] for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor(
            text=label_texts,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels

        return batch


def multi_main(args):
    """Main function for multilingual training."""
    dfs, iso2vocab_size = load_data(ISOS)
    
    # tokenizer
    tokenizer_map = {}
    for iso in ISOS:
        tokenizer_map[iso] = Wav2Vec2CTCTokenizer(
            vocab_file=f"{iso}_vocab.json",
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )
    
    # dataset
    dataset = MultilingualASRDataset(
        tsv=dfs,
        lang2id=LANG2ID,
        audio_root="../mcv-sps-st-09-2025"
    )
    train_dataset, dev_dataset = random_split(dataset, [0.9, 0.1])
    
    base_processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path)
    feature_extractor = base_processor.feature_extractor
    
    collate_fn = partial(
        ctc_collate_fn_multi,
        feature_extractor=feature_extractor,
        tokenizer_map=tokenizer_map
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )


def mono_main(args):
    """Main function for monolingual training."""
    dfs, iso2vocab_size = load_data([args.lang])
    
    # tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=f"{args.lang}_vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    # dataset
    dataset = MultilingualASRDataset(
        tsv=dfs[0],
        lang2id=ISO2ID,
        audio_root="../mcv-sps-st-09-2025"
    )
    train_dataset, dev_dataset = random_split(dataset, [0.9, 0.1])
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    collate_fn = partial(
        ctc_collate_fn_mono,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    


if __name__ == "__main__":
    args = get_args()
    
    if args.mono:
        assert args.lang is not None, "Please specify --lang for monolingual training."
        mono_main(args)
    else:
        raise NotImplementedError("Multilingual training not implemented yet.")