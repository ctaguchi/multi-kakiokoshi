from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import torch
import pandas as pd
from datasets import Dataset, Audio

import argparse
import os
from typing import List, Dict, Any
import glob
import shutil
import json


TEST_DATA_DIR = "src/data/mdc_asr_shared_task_test_data"
TEST_RESULTS_DIR = "src/data/test_results/"
UNSEEN = ["ady", "bas", "kbd", "qxp", "ush"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        help="Language to test."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model."
    )
    parser.add_argument(
        "-t",
        "--task_type",
        type=str,
        choices=["multilingual-general", "small-model", "unseen-langs"],
        help="Task type."
    )
    parser.add_argument(
        "-i",
        "--id",
        type=int,
        help="Results ID."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Evaluation batch size."
    )
    parser.add_argument(
        "--use_tmp_vocab",
        action="store_true",
        help="Use temporary vocab file."
    )
    return parser.parse_args()


def load_testdata(test_data_dir: str,
                  language: str) -> Dataset:
    """Load the test data."""
    audio_dir = os.path.join(test_data_dir, "audios")
    pattern = os.path.join(audio_dir, f"spontaneous-speech-{language}-*.mp3")
    test_audios = glob.glob(pattern)
    test_dataset = Dataset.from_dict({"audio": test_audios})
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    return test_dataset


def batched_prediction(batch: Dict[str, Any],
                       model: Wav2Vec2ForCTC,
                       processor: Wav2Vec2Processor,
                       device: str) -> Dict[str, Any]:
    """Batched prediction for faster evaluation.
    This should be run as `Dataset.map(batched_prediction, batched=True)`."""
    # batch["audio"] is a list of dicts: {"array": np.ndarray, "sampling_rate": 16000}
    arrays = [example["array"] for example in batch["audio"]]
    inputs = processor(
        arrays,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    batch["pred_str"] = pred_str
    
    return batch


def load_processor_with_temp_vocab(model_dir, temp_vocab_path):
    """
    model_dir: directory with the checkpoint (may NOT contain the vocab you want)
    temp_vocab_path: path to the 'real' vocab.json you want to use
    """

    vocab_in_model = os.path.join(model_dir, "vocab.json")
    backup_path     = vocab_in_model + ".bak"

    # Step 1 — rename/delete existing vocab if it exists
    if os.path.exists(vocab_in_model):
        os.rename(vocab_in_model, backup_path)

    # Step 2 — copy the "correct" vocab.json into the model directory
    shutil.copy(temp_vocab_path, vocab_in_model)

    try:
        # Step 3 — load processor from model_dir
        processor = Wav2Vec2Processor.from_pretrained(model_dir)
    
    finally:
        # Step 4 — restore: remove temporary vocab.json
        if os.path.exists(vocab_in_model):
            os.remove(vocab_in_model)

        # Step 5 — restore original vocab if it existed
        if os.path.exists(backup_path):
            os.rename(backup_path, vocab_in_model)

    return processor


def build_flat_vocab(nested_vocab, lang):
    flat = nested_vocab[lang]
    # make sure it's sorted by ID
    flat = {k: int(v) for k, v in flat.items()}
    return flat


def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.task_type == "small-model":
        model = Wav2Vec2ForCTC.from_pretrained(
            args.model,
        ).to(device)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(
            args.model,
            target_lang=args.language
        ).to(device)
    
    if not os.path.exists(os.path.join(args.model, "vocab.json")):
        # It is a checkpoint; look for the upper folder
        model_dir_with_vocab = os.path.dirname(args.model)
        vocab_file = os.path.join(os.path.dirname(args.model), "vocab.json")
    else:
        model_dir_with_vocab = args.model
        vocab_file = os.path.join(args.model, "vocab.json")
    
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    
    if args.task_type == "small-model":
        feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True
            )
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_dir_with_vocab,
                unk_token="[UNK]",
                pad_token="[PAD]",
                word_delimiter_token="|",
            )
        
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    else:
        # MMS-style
        flat_vocab = build_flat_vocab(vocab, lang=args.language)
        tmp_vocab_path = "vocab.json.tmp"
        with open(tmp_vocab_path, "w") as f:
            json.dump(flat_vocab, f, ensure_ascii=False)
        
        if args.use_tmp_vocab:
            processor = load_processor_with_temp_vocab(model_dir=model_dir_with_vocab,
                                                    temp_vocab_path=tmp_vocab_path)
        else:
            # processor = Wav2Vec2Processor.from_pretrained(model_dir_with_vocab)
            # processor.tokenizer.set_target_lang(args.language)
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True
            )
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_dir_with_vocab,
                unk_token="[UNK]",
                pad_token="[PAD]",
                word_delimiter_token="|",
                target_lang=args.language
            )
            processor = Wav2Vec2Processor(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer
            )
    
    # make temporary flat vocab file
    # 3. Create a temp directory
    # tmpdir = tempfile.TemporaryDirectory()  # persists until object deleted
    # tmp_path = tmpdir.name

    # # 4. Write new vocab.json
    # out_vocab_path = os.path.join(tmp_path, "vocab.json")
    # with open(out_vocab_path, "w") as f:
    #     json.dump(vocab, f, ensure_ascii=False)
        
    
    # processor = Wav2Vec2Processor.from_pretrained(model_dir_with_vocab)
    
    # 
    # processor = Wav2Vec2Processor.from_pretrained(tmp_path)
    # 
    
    # Load the test data
    test_dataset = load_testdata(test_data_dir=TEST_DATA_DIR, language=args.language)
    print("Test data size:", len(test_dataset))
    
    # Run inference
    print("Running the inference...")
    test_dataset = test_dataset.map(batched_prediction,
                                    batched=True,
                                    batch_size=args.batch_size,
                                    fn_kwargs={
                                        "model": model,
                                        "processor": processor,
                                        "device": device
                                    })
    print(test_dataset.column_names)
    preds: List[str] = test_dataset["pred_str"]
    
    # Load the test tsv sheet
    tsv_dir = os.path.join(TEST_DATA_DIR, args.task_type) # e.g. data/mdc_asr_shared_task_test_data/multilingual-general
    tsv_file = os.path.join(tsv_dir, f"{args.language}.tsv")
    tsv = pd.read_csv(tsv_file, sep="\t")
    tsv["sentence"] = preds
    
    # Save the results
    output_dir = os.path.join(TEST_RESULTS_DIR, str(args.id), args.task_type) # e.g. data/test_results/1/multilingual-general
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{args.language}.tsv")
    tsv.to_csv(results_file, sep="\t")


if __name__ == "__main__":
    args = get_args()
    main(args)
