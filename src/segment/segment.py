from datasets import load_dataset, Dataset, Audio
import torch
import torchaudio
import torchaudio.functional as F
from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np
import uroman as ur
import re
import os
import sys
import regex

import argparse
from typing import List, Tuple, Dict, Union, Any

# local
# sys.path.append("/home/chihiro/utils")
from utils.utils import prepare_dataset, BETAWI_NUMBERS, BUKUSU_NUMBERS


uroman = ur.Uroman()


def get_args():
    parser = argparse.ArgumentParser(
        description="Audio segmentation and aligner for data augmentation."
    )
    parser.add_argument(
        "-d",
        "--dataset_path_or_name",
        type=str,
        required=True,
        help="Path to the dataset or the name of the dataset from the Hugging Face Hub.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        required=True,
        help="Three-letter language code for the dataset (e.g., 'eng' for English).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the segmented dataset.",
    )
    parser.add_argument(
        "-n",
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to use for dataset mapping.",
    )
    return parser.parse_args()


FA_MODEL = None
VAD_MODEL = None
LABELS = None
DICTIONARY = None


def init_models(device: str = "cpu") -> Tuple[torch.nn.Module, torch.nn.Module, List[str], Dict[str, int]]:
    """Initialize VAD and forced alignment models."""
    global FA_MODEL, VAD_MODEL, LABELS, DICTIONARY
    
    if FA_MODEL is None:
        bundle = prepare_fa_model(device=device)
        FA_MODEL = bundle["model"]
        LABELS = bundle["labels"]
        DICTIONARY = bundle["dictionary"]

    if VAD_MODEL is None:
        vad_model = load_silero_vad()
        VAD_MODEL = vad_model.to(device).eval()    
        

# def normalize_text(text: str) -> Dict[str, str]:
#     """Normalize text using uroman.
#     To recover the original text, the original text is also returned in the dict.
#     """
#     text = re.sub(r"<[^>]+>", "", text) # Remove tags like <hesitation> , used in OmniLingual dataset
#     text = re.sub(r"\[[^\]]+\]", "", text) # Remove tags like [laugh], used in Common Voice dataset; maybe we should keep them?
#     text = text.replace("  ", " ")
#     return vad_model, fa_model, labels, dictionary
    

def normalize_text(text: str,
                   lang: str) -> Dict[str, str]:
    """Normalize text using uroman.
    To recover the original text, the original text is also returned in the dict.
    """
    original_text = text
    text = re.sub(r"<[^>]+>", "", text) # Remove tags like <hesitation> , used in OmniLingual dataset
    text = re.sub(r"\[[^\]]+\]", "", text) # Remove tags like [laugh], used in Common Voice dataset; maybe we should keep them?
    text = re.sub(";#x27;", "'", text) # for some languages like Rutoro ttj
    text = re.sub("&#x27;", "'", text) # same as above
    text = text.replace("(...)", "") # unwanted word
    greek_breve = chr(774)
    greek_acute = chr(900)
    text = regex.sub(r'[^\p{L}\p{N}\'\- ]+' + greek_breve + greek_acute, "", text)
    text = re.sub(r'[“”\"]', "", text)
    text = re.sub(r'\s+[,\.!?;:]\s*', lambda x: x.group().lstrip(), text) # remove spaces before punctuation
    text = text.replace(".,", ".")
    text = re.sub(r'\.+ ', ". ", text)
    text = re.sub(r'\.+', ".", text)
    text = re.sub(r'\(\s*', lambda x: x.group().rstrip(), text) # remove spaces after starting paren
    # if there's no whitespace after punctuation, add it
    text = re.sub(r'([.,;:!?])(?=\S)', r'\1 ', text)
    # if there is a floating comma or period (" , " or " . ")
    text = text.replace(" , ", ", ").replace(" . ", ". ")
    text = re.sub(r'\s+\.', ".", text)
    text = re.sub(r'\"', "", text) # remove punctuation marks

    if lang == "aln":  # Gheg Albanian
        text = text.replace("Ε", "E") # Somehow there's a greek capital epsilon in Gheg Albanian (aln) data
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text) # Sometimes there are Greek letters in the aln data; remove them
    elif lang == "bew": # Betawi
        text = re.sub(r'\d+', lambda x: BETAWI_NUMBERS.get(x.group(), x.group()), text)
    elif lang == "bxk": # Bukusu
        text = re.sub(r'\d+', lambda x: BUKUSU_NUMBERS.get(x.group(), x.group()), text) # just convert to string of digits
    elif lang == "el-CY": # Greek (Cyprus)
        # text = text.replace(" ,", ",").replace(" .", ".") # fix space before comma
        # text = re.sub(r"\s...\s", " ", text) # remove ...
        text = text.replace("–", "") # remove dashes
        text = re.sub(r"-", "", text) # remove hyphens
    elif lang == "sco": # Scots
        text = re.sub(r'[^A-Za-z\'’,\. ]+', '', text) # Use only alphabet, apostrophe, space, comma, period
    elif lang in {"cgg", # Chiga
                  "hch", # Wixárika
                  "kcn", # Nubi
                  "koo", # Konzo
                  "led", # Lendu
                  "lke", # Kenyi
                  "lth", # Thur
                  "meh", # Southwestern Tlaxiaco Mixtec
                  "mmc", # Michoacán Mazahua
                  "pne", # Western Penan
                  "ruc", # Ruuli
                  "rwm", # Amba
                  "sco", # Scots
                  "tob", # Toba Qom
                  "top", # Papantla Totonac
                  "ttj", # Rutoro
                  "ukv", # Kuku
                  }:
        pass # not for now
    else:
        raise NotImplementedError(f"Normalization for language '{lang}' is not implemented.")
    text = text.replace("  ", " ").strip()
    
    # Standardized text using uroman
    std_text = uroman.romanize_string(text)
    # std_text = text.lower().replace(",", "").replace(".", "").strip()
    # std_text = re.sub(r'[^A-Za-z0-9 ]+', '', text).lower()
    std_text = re.sub(r"[^A-Za-z\' ]+", "", std_text) # the final standardized text only contains alphabetical lowercase letters, apostrophes, and spaces
    std_text = std_text.lower().strip()
    return {"original_text": original_text,
            "norm_text": text,
            "std_text": std_text}


def get_transcript(sample: Dict[str, Any],
                   lang: str,
                   text_column_name: str = "raw_text") -> Dict[str, List[str]]:
    """Get transcript for the given data sample.
    The `text_column_name` should be 'raw_text' for OmniLingual dataset,
    and 'transcription' for Common Voice dataset.
    
    Args:
        sample (Dict[str, Any]): The sample from the dataset.
        text_column_name (str): The name of the text column in the dataset.
    Returns:
        Dict[str, List[str]]: A dictionary with keys 'norm_text' and 'std_text',
                              each containing a list of words.
    """
    texts = normalize_text(sample[text_column_name], lang=lang)
    assert len(texts["norm_text"].split()) == len(texts["std_text"].split()), \
        "The number of words in norm_text and std_text do not match:\n" \
        f"Length of norm_text: {len(texts['norm_text'].split())}\n" \
        f"Length of std_text: {len(texts['std_text'].split())}\n" \
        f"orig_text: {texts['original_text']}\n" \
        f"norm_text: {texts['norm_text']}\n" \
        f"std_text: {texts['std_text']}"
    texts = {
        "original_text": texts["original_text"],
        "norm_text": texts["norm_text"].split(),
        "std_text": texts["std_text"].split(),
    }
    return texts


def prepare_fa_model(device: str = "cpu") -> Dict[str, Any]:
    """Prepare the forced alignment model."""
    bundle = torchaudio.pipelines.MMS_FA # Forced alignment with MMS
    fa_model = bundle.get_model(with_star=False).to(device).eval()
    return {"model": fa_model,
            "labels": bundle.get_labels(star=None),
            "dictionary": bundle.get_dict(star=None)}


def get_emission(audio: torch.Tensor,
                 model: torch.nn.Module,
                 device: str = "cpu") -> torch.Tensor:
    """
    Args:
        audio (torch.Tensor or np.ndarray):
        model (torch.nn.Module):
        device (str): The device to run the model on.
    Returns:
        torch.Tensor: Emission matrix
    """
    # if isinstance(audio, np.ndarray):
    #     audio = torch.from_numpy(audio).float().unsqueeze(0)
    with torch.inference_mode():
        emission, _ = model(audio.to(device))
    return emission


def align(emission: torch.Tensor,
          tokens: List[int],
          device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform character-level forced alignment.
    Args:
        emission (torch.Tensor): Emission matrix from the model.
        tokens (List[int]): List of token IDs representing the transcript.
        device (str): The device to run the alignment on.
    Returns:
        alignments (torch.Tensor): Alignment matrix.
        scores (torch.Tensor): Alignment scores.
    """
    targets = torch.tensor([tokens],
                           dtype=torch.int32,
                           device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


def unflatten(list_, lengths):
    """Function used to unflatten for word-level alignment."""
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def vad(audio: torch.Tensor,
        vad_model: torch.nn.Module) -> List[Dict[str, float]]:
    """Perform voice activity detection using silero-vad."""
    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        min_silence_duration_ms=50,
        threshold=0.5
    )
    return speech_timestamps


def segment_audio(sample: Dict[str, Any],
                  lang: str,
                  text_column_name: str = "raw_text",
                  device: str = "cpu") -> Dict[str, Union[float, str]]:
    """Segment audio files in the dataset based on silence detection."""
    assert sample["audio"]["sampling_rate"] == 16000, "Sampling rate must be 16kHz."
    
    # Make sure models are loaded
    init_models(device)

    fa_model = FA_MODEL
    vad_model = VAD_MODEL
    dictionary = DICTIONARY
    
    # Alignment 
    texts = get_transcript(sample,
                           text_column_name=text_column_name,
                           lang=lang)
    std_text = texts["std_text"]
    norm_text = texts["norm_text"]
    print("Original text:", texts["original_text"])
    print("norm_text:", " ".join(norm_text))
    print("std_text: ", " ".join(std_text))
    
    # tokenization
    # tokens = [dict´ionary[c] for word in std_text for c in word]  # Flatten list of characters
    tokens = []
    for word in std_text:
        for c in word:
            if c in dictionary:
                tokens.append(dictionary[c])
            else:
                # Optional: log unknown characters
                print(f"[WARN] char {repr(c)} not in dictionary; skipping")
    
    if len(tokens) == 0:
        # Nothing we can align; just do VAD segmentation and return empty transcripts
        print("[WARN] Empty token sequence after normalization; skipping alignment.")
        audio_np = sample["audio"]["array"]
        audio = torch.from_numpy(audio_np).float().unsqueeze(0)

        speech_timestamps = vad(audio, vad_model)

        return [{
            "start": None,
            "end": None,
            "std_transcription": "",
            "norm_transcription": ""
        }]
                
    audio = sample["audio"]["array"] # np.ndarray
    audio = torch.from_numpy(audio).float().unsqueeze(0)  # Convert to torch.Tensor

    emission = get_emission(audio, fa_model)
    aligned_tokens, alignment_scores = align(emission=emission,
                                             tokens=tokens,
                                             device=args.device)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    word_spans = unflatten(token_spans, [len(word) for word in std_text])
    
    num_frames = emission.size(1)
    
    # VAD
    speech_timestamps = vad(audio, vad_model)
    ratio = int(audio.size(1) / num_frames)
    sr = sample["audio"]["sampling_rate"]
    
    i = 0  # Pointer for alignment
    transcriptions: List[str] = []
    
    for ts in speech_timestamps:
        segment_transcription = ""
        
        while i < len(word_spans):
            word_start = word_spans[i][0].start * ratio / sr
            word_end = word_spans[i][-1].end * ratio / sr
            # If word_start is after the end of the current speech segment,
            # then the current word is not part of this segment; go to next speech segment
            # ts["start"] < ts["end"] < word_start < word_end
            if ts["start"] < ts["end"] <= word_start < word_end:
                break
            
            # elif word_start is within the current speech segment but word_end is
            # # after the end of the current speech segment,
            # then the current word is partially included in this segment;
            # we include this word too and go to next speech segment
            # ts["start"] <= word_start < ts["end"] < word_end
            elif ts["start"] <= word_start < ts["end"] < word_end:
                segment_transcription += std_text[i] + " "
                i += 1
                break
            # elif word_start < ts["end"] <= word_end:
            #     segment_transcription += std_text[i] + " "
            #     i += 1
            #     break
            
            # elif word_end is within the current speech segment,
            # then the current word is fully included in this segment;
            # ts["start"] <= word_start < word_end <= ts["end"],
            # we include this word and continue to the next word
            elif ts["start"] <= word_start < word_end <= ts["end"]:
                segment_transcription += std_text[i] + " "
                i += 1
                
            elif word_start < word_end <= ts["start"] < ts["end"]:
                i += 1  # Skip this word; it is before the current speech segment
                
            elif word_start < ts["start"] < word_end < ts["end"]:
                # This case is when the word_start is before the current speech segment
                # but word_end is within the current speech segment.
                # include this word and go to the next word
                segment_transcription += std_text[i] + " "
                i += 1
            
            elif word_start < ts["start"] < ts["end"] <= word_end:
                # This case is when the segment is fully within the current word.
                # include this word and go to the next speech segment
                segment_transcription += std_text[i] + " "
                i += 1
                break
        
            else:
                # This case si when both word_start and word_end are outside the current speech segment
                raise ValueError("Unexpected case in segmentation.\n"
                                 f"word_start: {word_start},\n"
                                 f"word_end: {word_end},\n"
                                 f"ts['start']: {ts['start']},\n"
                                 f"ts['end']: {ts['end']}\n,"
                                 f"transcript so far: {segment_transcription}")

        transcriptions.append(segment_transcription.strip())
        
    # Map back to the original orthography before applying uroman
    word_idx = 0
    norm_transcriptions = []
    for std_transcription in transcriptions:
        # print(std_transcription)
        std_words = std_transcription.split()
        if len(std_words) == 0:
            norm_transcriptions.append("")
            continue
        orig_words = []
        for std_word in std_words:
            orig_words.append(norm_text[word_idx])
            word_idx += 1
        norm_transcription = " ".join(orig_words)
        norm_transcriptions.append(norm_transcription)
        
    return [{
        "start": ts["start"],
        "end": ts["end"],
        "std_transcription": transcriptions[idx],
        "norm_transcription": norm_transcriptions[idx]
    } for idx, ts in enumerate(speech_timestamps)]
    

def segment_dataset(dataset: Dataset,
                    text_column_name: str,
                    lang: str,
                    device: str,
                    num_proc: int = 1) -> Dataset:
    """Segment audio files in the dataset based on silence detection."""
    def segment_wrapper(sample):
        # return segment_audio(sample, fa_model, vad_model, labels, dictionary)
        segments = segment_audio(sample,
                                 lang=lang,
                                 text_column_name=text_column_name,
                                 device=device)
        sample["segments"] = segments
        return sample
    
    def keep_segments(sample: dict):
        """Filter out samples without transcription after normalization"""
        if len(sample["segments"]) > 0:
            if sample["segments"][0]["start"] is not None:
                return True
            else:
                return False
        else:
            return False
    
    dataset = dataset.map(segment_wrapper,
                                    num_proc=num_proc)
    dataset = dataset.filter(keep_segments)
    return dataset
        

if __name__ == "__main__":
    args = get_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if args.dataset_path_or_name == "facebook/omnilingual-asr-corpus":
        dataset = load_dataset(args.dataset_path_or_name,
                               language=args.language)
        text_column_name = "raw_text"
    elif args.dataset_path_or_name == "mcv-sps-st-09-2025":
        dataset = prepare_dataset(args.dataset_path_or_name,
                                  args.language,
                                  num_proc=args.num_proc)
        text_column_name = "transcription"
    elif args.dataset_path_or_name == "common-voice-23":
        dataset = prepare_dataset(args.dataset_path_or_name,
                                  args.language,
                                  num_proc=args.num_proc)
        text_column_name = "sentence"
    else:
        raise ValueError("Unsupported dataset. Supported datasets are: "
                         "'facebook/omnilingual-asr-corpus', "
                         "'mcv-sps-st-09-2025', and 'common-voice-23'.")
    
    print(f"Dataset loaded: {dataset}")
    print(f"Dataset size: {len(dataset['train'])} samples")
        
    if dataset["train"][0]["audio"]["sampling_rate"] != 16000:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    segmented_dataset = segment_dataset(dataset["train"],
                                        text_column_name=text_column_name,
                                        lang=args.language,
                                        num_proc=args.num_proc,
                                        device=args.device)
    
    if args.output_path is None:
        DATA_DIR = "./data"
        args.output_path = os.path.join(
            DATA_DIR,
            args.dataset_path_or_name,
            args.language,
            "segmented_dataset"
        )
            
    segmented_dataset.save_to_disk(args.output_path)
    
 