# TODO:
# - Language-specific digraphs to the vocab
# - DONE: Remove empty text samples
# - (Optional) Concatenate samples when segments are too short
# - DONE: Implement a training cycle with combined segments
# - (Optional) Prepare a segmented dev version to compare the performance against the longer version
# - Implement a character-level language model with pyctcdecode

from datasets import load_dataset, Dataset, DatasetDict, Audio, concatenate_datasets
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
from safetensors.torch import save_file as safe_save_file
from audiomentations import Compose, PitchShift, AddBackgroundNoise, TimeStretch
import wandb
import torch
import jiwer
import numpy as np
import huggingface_hub
import regex
import re
import unicodedata
import torch.nn as nn

import argparse
from typing import List, Dict, Any, Union, Optional, Callable, Literal
import os
import json
from dataclasses import dataclass
import dotenv
from pathlib import Path
import warnings
import gc
import copy


dotenv.load_dotenv() # Load the .env variables
huggingface_hub.login(token=os.environ["HF_TOKEN"])


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR / ".." / ".." / "models").resolve()
SSCLangs = [ # Languages to be trained with Spontaneous Speech Corpus data
    "aln", "bew", "bxk", "el-CY", "cgg", "hch", "kcn", "koo", "led", "lke",
    "lth", "meh", "mmc", "pne", "ruc", "rwm", "sco", "tob", "top", "ttj", "ukv",
]
CVLangs = [ # Languages to be trained with Common Voice data
    "ady", "bas", "kbd", "qxp", "ush"
]
UNCASED_LANGS = ["ukv"]
USERNAME = "ctaguchi"
DATA_COLUMNS = ["client_id", "audio_id", "audio_file", "duration_ms", "prompt_id",
                "prompt", "transcription", "votes", "age", "gender", "language",
                "split", "char_per_sec", "quality_tags", "audio", "segments"] # "audio" and "segments" are the newly added column through preprocessing
DISCARDED_COLUMNS = ["client_id", "audio_id", "audio_file", "duration_ms", "prompt_id",
                     "prompt", "votes", "age", "gender", "language", "split",
                     "char_per_sec", "quality_tags"]
Feature = Dict[str, Union[List[int], torch.Tensor]]


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Pretrained model."
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        required=True,
        choices=SSCLangs + CVLangs,
        help="Language code."
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repo (directory) name to save the model."
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="If set, the datasets will be forcibly redownloaded (don't use cache)."
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes (CPU) for `Dataset.map`."
    )
    parser.add_argument(
        "--freeze_feature_encoder",
        action="store_true",
        help="If set, the feature encoder's parameters will be frozen."
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=30,
        help="The number of epochs to run."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for GPU training."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warm up steps."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Evaluation steps."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Training logging steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Model saving steps. This has to be a round multiple of the evaluation steps."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, the model will be pushed to Hugging Face."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ["WANDB_PROJECT"] if os.environ["WANDB_PROJECT"] else None,
        help="WandB project name. Default to the project name set as an env variable."
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Run name to be shown on wandb."
    )
    parser.add_argument(
        "--replace_ctc",
        action="store_true",
        help="Replace the CTC layer (when the pretrained model already has a trained CTC)"
    )
    parser.add_argument(
        "--maximize_training_data",
        action="store_true",
        help="If set, the dataset loader will use as much data as possible, leaving only 1000 samples for evaluation."
    )
    
    # Data augmentation
    parser.add_argument(
        "--pitch_shift",
        action="store_true",
        help="For data augmentation with `audiomentations`."
    )
    parser.add_argument(
        "--min_semitones",
        type=float,
        default=0.0,
        help="Minimum semitones for pitch shift (lowering).",
    )
    parser.add_argument(
        "--max_semitones",
        type=float,
        default=10.0,
        help="Maximum semitones for pitch shift (raising).",
    )
    parser.add_argument(
        "--add_background_noise",
        action="store_true",
        help="For data augmentation with `audiomentations`."
    )
    parser.add_argument(
        "--background_noise_dir",
        type=str,
        default=None,
        help="The audio directory containing background noise files."
    )
    parser.add_argument(
        "--time_stretch",
        action="store_true",
        help="For data augmentation with `audiomentations`."
    )
    parser.add_argument(
        "--min_rate",
        type=float,
        default=0.8,
        help="Minimum rate for time stretch.",
    )
    parser.add_argument(
        "--max_rate",
        type=float,
        default=1.25,
        help="Maximum rate for time stretch.",
    )
    parser.add_argument(
        "--init_adapter_layer",
        action="store_true",
        help="If set, model.init_adapter_layer() will be called instead of replacing the adapter from scratch."
    )
    parser.add_argument(
        "--adapter_lang",
        type=str,
        default=None,
        help="If a language code is specified, the pretrained adapter of the language will be used for the training."
    )
    
    # Sample combination for adapting to longer data
    parser.add_argument(
        "--train_with_longer_samples",
        action="store_true",
        help="Additional training with combined samples to form longer segments."
    )
    parser.add_argument(
        "--train_with_superlong_samples",
        action="store_true",
        help="Train with superlong samples (~10seconds)"
    )
    parser.add_argument(
        "--train_with_maxlong_samples",
        action="store_true",
        help="Train with maximum-length (original) samples."
    )
    parser.add_argument(
        "--maxlong_batch_size",
        type=int,
        default=1,
        help="Batch size for the original audio training."
    )
    parser.add_argument(
        "--maxlong_epoch",
        type=int,
        default=1,
        help="Number of epochs for the original audio samples."
    )
    parser.add_argument(
        "--superlong_batch_size",
        type=int,
        default=1,
        help="Batch size for the superlong samples."
    )
    parser.add_argument(
        "--superlong_epoch",
        type=int,
        default=1,
        help="Number of epochs for the superlong samples."
    )
    parser.add_argument(
        "--long_batch_size",
        type=int,
        default=4,
        help="Batch size for the second training with longer samples."
    )
    parser.add_argument(
        "--long_epoch",
        type=int,
        default=5,
        help="Number of epochs for the second traiing with longer samples."
    )
    parser.add_argument(
        "--mix_long_short",
        action="store_true",
        help="If set, mix the short segments and combined segments."
    )
    parser.add_argument(
        "--use_jw_data",
        action="store_true",
        help="If set, additional training data will be loaded."
    )
    parser.add_argument(
        "--run_original_at_end",
        action="store_true",
        help="If set, training with the original dataset will be run at the end"
    )
    parser.add_argument(
        "--train_with_original_only",
        action="store_true",
        help="If set, train the model with the original audio only."
    )
    
    return parser.parse_args()


def load_data(language: str,
              force_redownload: bool) -> DatasetDict:
    """Load dataset.
    
    Args:
        language (str): Language code.
    Return:
        DatasetDict: containing the train and dev sets.
    """
    download_mode = "force_redownload" if force_redownload else "reuse_dataset_if_exists"
    if args.language in SSCLangs:
        # load from huggingface
        dataset_name = f"mcv-sps-{language}-segmented"
        dataset = load_dataset(f"{USERNAME}/{dataset_name}",
                               split="train",
                               download_mode=download_mode)
        
        # train-dev split
        train_data = dataset.filter(lambda sample: sample["split"] == "train")
        dev_data = dataset.filter(lambda sample: sample["split"] == "dev")
        datasetdict = DatasetDict(
            {"train": train_data,
             "dev": dev_data}
        )
        
        # Remove unnecessary columns
        datasetdict = datasetdict.map(
            lambda x: x,
            remove_columns=DISCARDED_COLUMNS
        )
    elif args.language in CVLangs:
        # Load from common voice
        dataset_name = f"common_voice_{language}"
        datasetdict = load_dataset(f"{USERNAME}/{dataset_name}")
    else:
        raise ValueError(f"{args.language} is not a supported language for this experiment.")

    return datasetdict


def is_short_enough(sample: Dict[str, Any]) -> bool:
    """Remove samples that are too long.
    Important because dev set audios are not segmented.
    """
    sr = sample["audio"]["sampling_rate"]
    n_samples = len(sample["audio"]["array"])
    duration_sec = n_samples / sr
    return duration_sec <= 90.0


def is_long_enough(sample: Dict[str, Any]) -> bool:
    """Remove samples that are too short.
    It seems Wav2Vec2 models are not good at learning short segments.
    """
    sr = sample["audio"]["sampling_rate"]
    n_samples = len(sample["audio"]["array"])
    duration_sec = n_samples / sr
    return duration_sec >= 1.5


def has_transcription(sample: Dict[str, Any]) -> bool:
    """Remove samples that have no transcription."""
    return len(sample["transcription"].strip()) > 0


def batch_collapse_segments(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse the segments as samples,
    for batch processing with `.map()`."""
    new_audio = []
    new_texts = []
    
    for audio, segments in zip(batch["audio"], batch["segments"]):
        # audio is a dict from HF Audio feature: {"array": np.ndarray, "sampling_rate": int}
        waveform = audio["array"]
        sr = audio["sampling_rate"]

        for seg in segments:
            start_t = seg["start"]          # in seconds
            end_t   = seg["end"]            # in seconds
            text    = seg["norm_transcription"]

            if text is None or text == "":
                continue  # skip empty transcriptions if any

            start_sample = int(start_t * sr)
            end_sample   = int(end_t * sr)

            # Slice the waveform
            seg_waveform = waveform[start_sample:end_sample]

            # Append as a new example
            new_audio.append({
                "array": seg_waveform,
                "sampling_rate": sr,
            })
            new_texts.append(text)
    
    return {"audio": new_audio,
            "transcription": new_texts}
    

def combine_segments(dataset: Dataset,
                     sample_idx: int,
                     start_idx: int,
                     end_idx: int):
    """Combine neighboring samples in the specified range."""
    sample = dataset[sample_idx]
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    
    start_segment = sample["segments"][start_idx]
    start_t = start_segment["start"]
    
    end_segment = sample["segments"][end_idx]
    end_t = end_segment["end"]
    text = " ".join([sample["segments"][i]["norm_transcription"] for i in range(start_idx, end_idx + 1)])
    start_sample = int(start_t * sr)
    end_sample = int(end_t * sr)
    segment_audio = audio[start_sample:end_sample]
    return segment_audio, sr, text


def combine_segments_in_dataset(dataset: Dataset,
                                combine_last: bool = True,
                                width: Optional[int] = None,
                                min_length: Optional[int] = None) -> Dataset:
    """Combine samples in a dataset to form longer samples.
    Args:
        dataset (Dataset): the dataset.
        combine_end (bool): whether to combine the remaining segments with the previous.
        width (Optional[int]): Number of segments to combine
        min_length:
    """
    assert width or min_length, "Specify the sample merging strategy."
    assert not (width and min_length), "`width` and `min_length` cannot be combined."
    
    combined_dataset = []
    if width:
        for sample_idx, sample in enumerate(dataset):
            for i in range(0, len(sample["segments"]), width):
                if i + width - 1 > len(sample["segments"]) - 1 and combine_last: # combine the last one
                    end_idx = len(sample["segments"]) - 1
                    start_idx = max(0, i - width)
                    if len(combined_dataset) >= 1:
                        combined_dataset.pop()
                else:
                    end_idx = i + width - 1
                    start_idx = i
                    
                segment_audio, sr, text = combine_segments(
                    dataset=dataset,
                    sample_idx=sample_idx,
                    start_idx=start_idx,
                    end_idx=end_idx
                )
                    
                combined_dataset.append(
                    {
                        "audio": {"array": segment_audio, "sampling_rate": sr},
                        "transcription": text,
                        "length": len(segment_audio) / sr
                    }
                )
    
    elif min_length:
        for sample_idx, sample in enumerate(dataset):
            segments = sample["segments"]
            prev_start_idx = 0
            start_idx = 0
            start_t = segments[start_idx]["start"]
            
            for i in range(len(segments)):
                end_t = sample["segments"][i]["end"]
                if end_t - start_t > min_length or i == len(segments) - 1:
                    if i == len(segments) - 1 and combine_last:
                        start_idx = prev_start_idx
                        if len(combined_dataset) >= 1:
                            combined_dataset.pop() # remove the previous segment
                        
                    segment_audio, sr, text = combine_segments(
                        dataset=dataset,
                        sample_idx=sample_idx,
                        start_idx=start_idx,
                        end_idx=i
                    )
                    combined_dataset.append(
                        {
                            "audio": {"array": segment_audio, "sampling_rate": sr},
                            "transcription": text,
                            "length": len(segment_audio) / sr # debug
                        }
                    )
                    
                    if i < len(segments) - 1:
                        prev_start_idx = start_idx
                        start_idx = i + 1
                        start_t = segments[start_idx]["start"]
    
    combined_dataset = Dataset.from_list(combined_dataset)
    return combined_dataset

    
APOSTROPHES = "'’ʻʼ`"
DIACRITICS = "\u0301\u0300\u0302\u030C\u0304" # For Mazahua
ALLOWED_CHARS = fr"[^\p{{Latin}}\p{{Greek}}\p{{Cyrillic}}\p{{M}}{APOSTROPHES}{DIACRITICS}\u0306\u0384 ]+"
ALLOWED_CHARS_PATTERN = regex.compile(ALLOWED_CHARS)

def normalize_text(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Regex-based text normalization:
    - Lowercase
    - Keep Latin, Greek, combining marks, apostrophes, breve, and tonos
    - Remove everything else
    """
    def clean(text: str) -> str:
        if text is None:
            return ""

        # Lowercase
        text = text.lower()

        # Normalize to NFD so regex can see combining marks
        text = unicodedata.normalize("NFD", text)

        # Remove all disallowed characters
        # text = re.sub(ALLOWED_CHARS, "", text)
        text = ALLOWED_CHARS_PATTERN.sub("", text)

        # Re-compose
        text = unicodedata.normalize("NFC", text)

        return text

    # Apply to all transcriptions in batch
    batch["transcription"] = [clean(t) for t in batch["transcription"]]
    return batch


BRACKETED = re.compile(r"\[[^\]]+\]")
UNINTELL_PAREN = re.compile(r"\(\?+\)")
REPL_PUNC = re.compile('[,?¿¡!";:»«“”]+') # »«“” were added by the user
MULTISPACE = re.compile("  +")

def normalize_text_official(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Official normalization used by the shared task."""
    t = batch["transcription"]
    t = re.sub(BRACKETED, " ", t)
    t = re.sub(UNINTELL_PAREN, " ", t)
    t = t.replace(" ... ", " ")
    t = t.replace("#x27;", "'")
    t = re.sub(REPL_PUNC, " ", t)
    t = t.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    t = re.sub(MULTISPACE, " ", t)
    batch["transcription"] = t
    return batch


def get_vocab_from_dataset(datasetdict: DatasetDict,
                           orthographic: bool = True,
                           language: Optional[str] = None) -> Dict[str, int]:
    """Create a vocab dict from the dataset.
    For the SSC datasets, use the normalized transcription `norm_transcription`.
    `std_transcription` is rigorously standardized transcription by uroman,
    discarding some phonetically/orthographically important cues such as diacritics.
    """
    # def extract_chars(batch: dict) -> Dict[str, Any]:
    #     segments: List[Dict[str, str | int]] = batch["segments"]
    #     transcriptions = [seg["norm_transcription"] for seg in segments]
    #     text = "".join(transcriptions)
    #     return {"chars": transcriptions}
        
    train = datasetdict["train"]
    dev = datasetdict["dev"]
    dataset = concatenate_datasets([train, dev])
    # dataset_chars = dataset.map(extract_chars,
    #                             num_proc=num_proc)
    
    all_chars = set()
    # for chars in dataset_chars["chars"]:
    #     all_chars.update(chars)
    for transcription in dataset["transcription"]:
        all_chars.update(set(transcription))
        
    # orthographic digraphs
    if orthographic:
        assert language is not None, "`language` arg needs to be specified when orthographic=True.`"
        with open("src/utils/digraphs.json", "r") as f:
            digraphs_all = json.load(f)
        if language not in digraphs_all.keys():
            warnings.warn("Digraphs are not defined for the language. Fallback to default vocab.")
            digraphs = {}
        else:
            digraphs = set(digraphs_all[language])
        
        all_chars.update(digraphs)
    
    vocab = {c: i for i, c in enumerate(all_chars)}
    vocab["|"] = vocab[" "]
    del vocab[" "]
    
    # CTC special tokens
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    
    return vocab


def prepare_dataset(batch: dict,
                    processor: Wav2Vec2Processor,
                    augmentor=None) -> dict:
    """Prepare the dataset for the training.
    Add `input_values` and `labels` to the dataset.
    """
    # word-delimiter token
    batch["transcription"] = batch["transcription"].replace(" ", "|")
    
    audio = batch["audio"]
    if augmentor is not None: # do data augmentation
        audio["array"] = augmentor(samples=audio["array"],
                                   sample_rate=audio["sampling_rate"])

    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0] # batched output is un-batched

    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = batch["transcription"]

    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received.
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


def make_compute_metrics(processor: Wav2Vec2Processor):
    def compute_metrics(pred) -> Dict[str, float]:
        """Compute the evaluation score (CER)."""
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids,
                                        skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(label_ids,
                                        group_tokens=False,
                                        skip_special_tokens=True)

        cer = jiwer.cer(
            reference=label_str,
            hypothesis=pred_str
        )
        wer = jiwer.wer(
            reference=label_str,
            hypothesis=pred_str
        )
        
        # log full results
        max_examples_to_log = 10
        table = wandb.Table(columns=["prediction", "reference"])
        for p, r in list(zip(pred_str, label_str))[:max_examples_to_log]:
            table.add_data(p, r)
        wandb.log({"eval/examples": table})

        return {"cer": cer, "wer": wer}

    return compute_metrics


def run_train(mode: Literal["main", "long", "superlong", "maxlong"],
              train_dataset: Dataset,
              eval_dataset: Dataset,
              data_collator,
              processor,
              compute_metrics: Callable,
              output_dir: str,
              args: argparse.Namespace) -> None:
    """The training phase."""
    if mode == "main" or args.train_with_original_only:
        batch_size = args.batch_size
        num_train_epochs = args.epoch
        
        if args.adapter_lang:
            model = Wav2Vec2ForCTC.from_pretrained(args.model,
                                                   target_lang=args.adapter_lang,
                                                   ignore_mismatched_sizes=True) # important
            model.load_adapter(args.adapter_lang)
        else:
            ignore_mismatched_sizes = True if args.model == "facebook/mms-1b-all" else False
            new_vocab_size = len(processor.tokenizer)
                
            model = Wav2Vec2ForCTC.from_pretrained(
                args.model,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                feat_proj_dropout=0.0,
                mask_time_prob=0.05,
                layerdrop=0.0,
                ctc_loss_reduction="mean",
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=new_vocab_size,
                ignore_mismatched_sizes=ignore_mismatched_sizes
            )
            
            # Replace/resize CTC head if necessary
            if ignore_mismatched_sizes:
                if args.replace_ctc: # Totally replace the CTC layer with a new linear CTC layer
                    in_features = model.lm_head.in_features
                    model.lm_head = nn.Linear(in_features, new_vocab_size)
                    model.config.vocab_size = new_vocab_size
                elif args.init_adapter_layer: # Use the off-the-shelf method of the model
                    model.init_adapter_layers()
                else:
                    raise ValueError("Either args.replace_ctc or args.init_adapter_layer must be activated.")
            
    elif mode == "long":
        batch_size = args.long_batch_size
        num_train_epochs = args.long_epoch
        model = Wav2Vec2ForCTC.from_pretrained(output_dir)
    elif mode == "superlong":
        batch_size = args.superlong_batch_size
        num_train_epochs = args.superlong_epoch
        model = Wav2Vec2ForCTC.from_pretrained(output_dir)
    elif mode == "maxlong" and not args.train_with_original_only:
        batch_size = args.maxlong_batch_size
        num_train_epochs = args.maxlong_epoch
        model = Wav2Vec2ForCTC.from_pretrained(output_dir)
    else:
        raise ValueError
    
    if args.freeze_feature_encoder:
        model.freeze_base_model() # prevents overfitting, faster training
        
    if args.init_adapter_layer:
        adapter_weights = model._get_adapters()
        for param in adapter_weights.values():
            param.requires_grad = True
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.repo_name,
        hub_token=os.environ["HF_TOKEN"],
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    print("Training started.")
    trainer.train()
    trainer.save_model() # will be saved to `output_dir`
    
    # the trainer won't be used in the next stage
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return model


def main(args: argparse.Namespace) -> None:
    """Main function."""
    print("Loading data...")
    datasetdict = load_data(args.language,
                            force_redownload=args.force_redownload)
    
    if args.language in SSCLangs:
        train = datasetdict["train"]
        dev = datasetdict["dev"]
        print("Data loaded.")
        
        # Make a longer version
        if args.train_with_longer_samples:
            long_train = combine_segments_in_dataset(
                dataset=train,
                combine_last=True,
                min_length=5
            )
            # Sample cleaning
            long_train = long_train.filter(has_transcription)
            # long_train = long_train.map(normalize_text,
            #                             batched=True)
            long_train = long_train.map(normalize_text_official)
        # Superlong version
        if args.train_with_superlong_samples:
            superlong_train = combine_segments_in_dataset(
                dataset=train,
                combine_last=True,
                min_length=10
            )
            # Sample cleaning
            superlong_train = superlong_train.filter(has_transcription)
            # superlong_train = superlong_train.map(normalize_text,
            #                                       batched=True)
            superlong_train = superlong_train.map(normalize_text_official)
        # Max long version
        if args.train_with_maxlong_samples: # original length; this'll be a MUST because the dev set is case-sensitive
            max_train = train.filter(has_transcription)
            max_train = train.map(normalize_text_official,
                                  remove_columns=["segments"])
        
        if args.train_with_original_only:
            train = max_train
            dev = dev.remove_columns(["segments"])
            dev = dev.filter(is_short_enough)
            print("Original data prepared.")
            
        else:
            print("Collapsing segments...")
            train = train.map(batch_collapse_segments,
                            batched=True,
                            batch_size=16,
                            num_proc=4,
                            remove_columns=train.column_names) # should take about 5 mins, and the memory should be ok within 12.7GB
            train = train.filter(is_long_enough)
            print("Segments that are too short have been removed.")
            # The total number of samples can reach around 10k
            dev = dev.map(lambda x: x,
                          remove_columns=["segments"]) # we are not using segments for dev data
            dev = dev.filter(is_short_enough)
            print("Segments collapsed.")
            
        
        # Additional training data
        if args.use_jw_data:
            assert not (args.use_jw_data and args.train_with_original_only), "args.use_jw_data and args.train_with_original_only cannot be True at the same time."
            print("Loading the additional data...")
            additional_dataset_name = f"jw_{args.language}"
            additional_train = load_dataset(f"{USERNAME}/{additional_dataset_name}")["train"].remove_columns(["path"])
            train = concatenate_datasets([train, additional_train])
            print("Additional data loaded and concatenated to the main train set.")
    
    elif args.language in CVLangs:
        train = datasetdict["train"] # kbd CV data contains JW data by default
        dev = datasetdict["dev"]
        if args.maximize_training_data:
            # just leave 1000 samples for dev and test
            threshold = min(1000, len(dev) * 0.5)
            test = datasetdict["test"]
            other = datasetdict["other"]
            dev_train = Dataset.from_dict(dev[threshold:]).cast_column("audio", Audio(sampling_rate=16000))
            dev = Dataset.from_dict(dev[:threshold]).cast_column("audio", Audio(sampling_rate=16000))
            test_train = Dataset.from_dict(test[threshold:]).cast_column("audio", Audio(sampling_rate=16000))
            test = Dataset.from_dict(test[:threshold]).cast_column("audio", Audio(sampling_rate=16000))
            train = concatenate_datasets([train, dev_train, test_train, other])
    
    # Sample cleaning
    train = train.filter(has_transcription)
    dev = dev.filter(has_transcription)
    print("Samples with an empty transcription removed.")
    
    # Text normalization
    print("Normalizing the text...")
    # train = train.map(normalize_text,
    #                   batched=True)
    train = train.map(normalize_text_official)
    # dev = dev.map(normalize_text,
    #               batched=True)
    dev = dev.map(normalize_text_official)
    print("Text normalized.")
    
    datasetdict = DatasetDict({"train": train, "dev": dev})
    
    # Tokenizer/Processor
    if args.adapter_lang:
        print(f"Using the pretrained adapter for {args.adapter_lang}...")
        processor = Wav2Vec2Processor.from_pretrained(args.model)
        processor.tokenizer.set_target_lang(args.adapter_lang)
        # not to be confused with the training target language --
        # we can use the English adapter for training Scots
        
    else:
        print("Creating the vocab...")
        vocab = get_vocab_from_dataset(datasetdict,
                                    language=args.language,
                                    orthographic=True)
        vocab_dict = {args.language: vocab}
        # save vocab
        model_dir = os.path.join(MODEL_DIR, args.repo_name)
        vocab_file = os.path.join(model_dir, "vocab.json")
        os.makedirs(model_dir, exist_ok=True)
        with open(vocab_file, "w") as f:
            json.dump(vocab_dict, f)
        print("Vocab created.")
        
        print("Preparing the tokenizer...")
        # tokenizer = Wav2Vec2CTCTokenizer(
        #     vocab_file=vocab_file,
        #     unk_token="[UNK]",
        #     pad_token="[PAD]",
        #     word_delimiter_token="|",
        #     target_lang=args.language
        # )
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            model_dir,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
            target_lang=args.language
        )
        if args.push_to_hub:
            # Save the tokenizer
            tokenizer.push_to_hub(args.repo_name)
        print("Tokenizer prepared.")
    
        print("Defining the feature extractor...")
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        print("Feature extractor defined.")
        
        print("Defining the processor...")
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
        print("Processor defined.")
    
    if (args.pitch_shift or args.add_background_noise or args.time_stretch):
        augment_methods = [
            AddBackgroundNoise(sounds_path=args.background_noise_dir) if args.add_background_noise else None,
            PitchShift(min_semitones=args.min_semitones,
                       max_semitones=args.max_semitones,
                       p=args.pitch_shift) if args.pitch_shift else None,
            TimeStretch(min_rate=args.min_rate,
                        max_rate=args.max_rate,
                        p=args.time_stretch) if args.time_stretch else None
        ]
        augmentor = Compose([method for method in augment_methods if method is not None])

    else:
        augmentor = None
    
    print("Formatting the dataset for training...")
    datasetdict = datasetdict.map(prepare_dataset,
                                  fn_kwargs={"augmentor": augmentor,
                                             "processor": processor},
                                  remove_columns=datasetdict["train"].column_names)
    print("Short-segment dataset prepared.")
    if args.train_with_longer_samples:
        long_train = long_train.map(prepare_dataset,
                                    fn_kwargs={"augmentor": augmentor,
                                               "processor": processor},
                                    remove_columns=long_train.column_names)
        if args.mix_long_short:
            datasetdict["train"] = concatenate_datasets([datasetdict["train"], long_train])
        print("Long-segment dataset prepared.")
    if args.train_with_superlong_samples:
        superlong_train = superlong_train.map(prepare_dataset,
                                              fn_kwargs={"augmentor": augmentor,
                                                         "processor": processor},
                                              remove_columns=superlong_train.column_names)
        if args.mix_long_short:
            datasetdict["train"] = concatenate_datasets([datasetdict["train"], superlong_train])
        print("Super-long-segment dataset prepared.")
    if args.train_with_maxlong_samples and not args.train_with_original_only:
        max_train = max_train.map(prepare_dataset,
                                  fn_kwargs={"augmentor": augmentor,
                                                         "processor": processor},
                                              remove_columns=max_train.column_names)
        if args.mix_long_short and not args.run_original_at_end:
            datasetdict["train"] = concatenate_datasets([datasetdict["train"], max_train])
        print("Max-long-segment dataset prepared.")
    print("Dataset formatted.")
    
    print("*** DEBUG ***")
    sample = datasetdict["train"][0]
    print(sample["labels"])
    print(processor(
        text=[sample["labels"]],
        return_tensors="pt"
    ).input_ids)
    print("Training data size:", len(datasetdict["train"]))
    
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True
    )
        
    # Evaluation metrics
    compute_metrics = make_compute_metrics(processor=processor)
        
    # wandb login
    try:
        wandb_api_key = os.environ["WANDB_API_KEY"]
    except KeyError as e:
        print("WandB API key not found in the environment.")
        print(e)
        
    output_dir = os.path.join(MODEL_DIR, args.repo_name)
    
    wandb.login(key=wandb_api_key)
    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    if not args.train_with_original_only:
        # Run training with smallest segmented audio
        model = run_train(mode="main",
            train_dataset=datasetdict["train"],
            eval_dataset=datasetdict["dev"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            output_dir=output_dir,
            processor=processor,
            args=args)
        print("Main training done.")
    
    if args.train_with_longer_samples and not args.mix_long_short:
        print("Starting the training with the long dataset.")
        model = run_train(mode="long",
              train_dataset=long_train,
              eval_dataset=datasetdict["dev"],
              data_collator=data_collator,
              compute_metrics=compute_metrics,
              output_dir=output_dir,
              processor=processor,
              args=args)
    
    if args.train_with_superlong_samples and not args.mix_long_short:
        print("Starting the training with the superlong dataset.")
        model = run_train(mode="superlong",
              train_dataset=long_train,
              eval_dataset=datasetdict["dev"],
              data_collator=data_collator,
              compute_metrics=compute_metrics,
              output_dir=output_dir,
              processor=processor,
              args=args)
        
    if args.train_with_maxlong_samples:
        print("Starting the training with the original dataset.")
        if args.run_original_at_end:
            model = run_train(mode="maxlong",
                    train_dataset=max_train,
                    eval_dataset=datasetdict["dev"],
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    output_dir=output_dir,
                    processor=processor,
                    args=args)
        elif args.train_with_original_only:
            model = run_train(mode="maxlong",
                    train_dataset=datasetdict["train"],
                    eval_dataset=datasetdict["dev"],
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    output_dir=output_dir,
                    processor=processor,
                    args=args)
    
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(args.language)
    adapter_file = os.path.join(output_dir, adapter_file)

    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
        
    wandb.finish()

if __name__ == "__main__":
    args = get_args()
    main(args)