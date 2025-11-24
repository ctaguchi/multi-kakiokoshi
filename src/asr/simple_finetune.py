from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
from audiomentations import Compose, PitchShift, AddBackgroundNoise, TimeStretch
import wandb
import torch
import jiwer
import numpy as np
import huggingface_hub

import argparse
from typing import List, Dict, Any, Union, Optional
import os
import json
from dataclasses import dataclass
import dotenv
from pathlib import Path


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
        "--push_to_hub",
        action="store_true",
        help="If set, the model will be pushed to Hugging Face."
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Run name to be shown on wandb."
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
    
    return parser.parse_args()


def load_data(language: str) -> DatasetDict:
    """Load dataset.
    
    Args:
        language (str): Language code.
    Return:
        DatasetDict: containing the train and dev sets.
    """
    if args.language in SSCLangs:
        # load from huggingface
        dataset_name = f"mcv-sps-{language}-segmented"
        dataset = load_dataset(f"{USERNAME}/{dataset_name}",
                               split="train")
        
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
        # Load from common voice locally
        raise NotImplementedError
    else:
        raise ValueError(f"{args.language} is not a supported language for this experiment.")

    return datasetdict


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


def get_vocab_from_dataset(datasetdict: DatasetDict,
                           num_proc: int = 1) -> Dict[str, int]:
    """Create a vocab dict from the dataset.
    For the SSC datasets, use the normalized transcription `norm_transcription`.
    `std_transcription` is rigorously standardized transcription by uroman,
    discarding some phonetically/orthographically important cues such as diacritics.
    """
    def extract_chars(batch: dict) -> Dict[str, Any]:
        segments: List[Dict[str, str | int]] = batch["segments"]
        transcriptions = [seg["norm_transcription"] for seg in segments]
        text = "".join(transcriptions)
        return {"chars": transcriptions}
        
    train = datasetdict["train"]
    dev = datasetdict["dev"]
    dataset = concatenate_datasets([train, dev])
    dataset_chars = dataset.map(extract_chars,
                                num_proc=num_proc)
    
    all_chars = set()
    for chars in dataset_chars["chars"]:
        all_chars.update(chars)
    vocab = {c: i for i, c in enumerate(all_chars)}
    vocab["|"] = len(vocab) # word delimiter token
    
    return vocab


def prepare_dataset(batch: dict,
                    processor: Wav2Vec2Processor,
                    augmentor=None) -> dict:
    """Prepare the dataset for the training.
    Add `input_values` and `labels` to the dataset.
    """
    audio = batch["audio"]
    if augmentor is not None: # data augmentation
        audio["array"] = augmentor(samples=audio["array"],
                                   sample_rate=audio["sampling_rate"])

    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0] # batched output is un-batched

    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = batch["text"]

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
        wandb.log({"val/examples": table})

        return {"cer": cer, "wer": wer}

    return compute_metrics


def main(args: argparse.Namespace) -> None:
    """Main function."""
    datasetdict = load_data(args.language)
    train = datasetdict["train"]
    dev = datasetdict["dev"]
    
    train = train.map(batch_collapse_segments,
                      batched=True,
                      batch_size=16,
                      num_proc=4,
                      remove_columns=train.column_names) # should take about 5 mins, and the memory should be ok within 12.7GB
    # The total number of samples can reach around 10k
    dev = dev.map(lambda x: x,
                  remove_columns=["segments"]) # we are not using segments for dev data
    
    datasetdict = DatasetDict({"train": train, "dev": dev})
    
    vocab = get_vocab_from_dataset(datasetdict)
    # save vocab
    vocab_file = os.path.join(args.repo_name, "vocab.json")
    with open(vocab_file, "w") as f:
        json.dump(vocab, f)
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_file,
        # unk_token="[UNK]",
        # pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
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
    
    datasetdict = datasetdict.map(prepare_dataset,
                                  fn_kwargs={"augmentor": augmentor,
                                             "processor": processor},
                                  remove_columns=datasetdict.column_names)
    
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True
    )
    
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder() # prevents overfitting, faster training
        
    # wandb login
    try:
        wandb_api_key = os.environ["WANDB_API_KEY"]
    except KeyError as e:
        print("WandB API key not found in the environment.")
        print(e)
    
    wandb.login(key=wandb_api_key)
        
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, args.repo_name),
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=args.epoch,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        push_to_hub=args.push_to_hub,
        push_to_hub_model_id=args.repo_name,
        hub_token=os.environ["HF_TOKEN"],
        report_to="wandb",
        run_name=args.wandb_run_name,
        load_best_model_at_end=True,
    )
    
    compute_metrics = make_compute_metrics(processor=processor)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=datasetdict["train"],
        eval_dataset=datasetdict["dev"],
        tokenizer=processor.feature_extractor,
    )
    
    trainer.train()
    trainer.save_model()
    

if __name__ == "__main__":
    args = get_args()
    main(args)