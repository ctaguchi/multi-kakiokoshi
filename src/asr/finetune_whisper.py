from datasets import load_dataset, DatasetDict
import jiwer
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import argparse
import os

from simple_finetune import get_args, MODEL_DIR


MAX_LENGTH = 448 # label sequence length allowed by the model


def prepare_dataset(batch,
                    processor):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"],
                                                sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = jiwer.wer(label_str, pred_str)

    return {"wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



if __name__ == "__main__":
    args = get_args()
    
    ds = load_dataset("ctaguchi/mcv-sps-sco-segmented", split="train")
    train = ds.filter(lambda x: x["split"] == "train")
    dev = ds.filter(lambda x: x["split"] == "dev")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model # 290MB, 74M params
    
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    # tokenizer = WhisperTokenizer.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name,
                                                 task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.generation_config.forced_decoder_ids = None
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # max 448 tokens
    train_clipped = train.filter(lambda x: len(processor.tokenizer(x["transcription"]).input_ids) <= MAX_LENGTH)
    dev_clipped = dev.filter(lambda x: len(processor.tokenizer(x["transcription"]).input_ids) <= MAX_LENGTH)
    
    ds = DatasetDict({"train": train_clipped, "dev": dev_clipped})
    ds = ds.map(prepare_dataset,
                remove_columns=ds.column_names["train"],
                num_proc=args.num_proc,
                fn_kwargs={"processor": processor})
    
    output_dir = os.path.join(MODEL_DIR, args.repo_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.learning_rate,
        warmup_steps=100,
        num_train_epochs=args.epoch,
        # max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=1000,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    trainer.train()
    trainer.save_model()