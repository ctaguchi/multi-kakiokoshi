from datasets import load_dataset, Dataset, Audio
import torch
import torchaudio
import torchaudio.functional as F
from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np
import uroman as ur
import re

import argparse
from typing import List, Tuple, Dict, Union, Any
import tqdm


uroman = ur.Uroman()


def get_args():
    parser = argparse.ArgumentParser(
        description="Audio segmentation and aligner for data augmentation."
    )
    parser.add_argument(
        "--dataset_path_or_name",
        type=str,
        required=True,
        help="Path to the dataset or the name of the dataset from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    return parser.parse_args()


def normalize_text(text: str) -> Dict[str, str]:
    """Normalize text using uroman.
    To recover the original text, the original text is also returned in the dict.
    """
    text = re.sub(r"<[^>]+>", "", text) # Remove tags like <hesitation> , used in OmniLingual dataset
    text = re.sub(r"\[[^\]]+\]", "", text) # Remove tags like [laugh], used in Common Voice dataset; maybe we should keep them?
    text = text.replace("  ", " ")
    text = text.replace(",", "").replace(".", "").strip()
    std_text = uroman.romanize_string(text)
    std_text = text.lower()
    return {"norm_text": text, "std_text": std_text}


def get_transcript(sample: Dict[str, Any],
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
    texts = normalize_text(sample[text_column_name])
    texts = {
        "norm_text": texts["norm_text"].split(),
        "std_text": texts["std_text"].split(),
    }
    return texts


def prepare_fa_model(device: str = "cpu") -> Dict[str, Any]:
    """Prepare the forced alignment model."""
    bundle = torchaudio.pipelines.MMS_FA # Forced alignment with MMS
    fa_model = bundle.get_model(with_star=False).to(device)
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
                  fa_model: torch.nn.Module,
                  vad_model: torch.nn.Module,
                  labels: List[str],
                  dictionary: Dict[str, int]) -> Dict[str, Union[float, str]]:
    """Segment audio files in the dataset based on silence detection."""
    assert sample["audio"]["sampling_rate"] == 16000, "Sampling rate must be 16kHz."
    
    # Alignment 
    texts = get_transcript(sample)
    std_text = texts["std_text"]
    tokens = [dictionary[c] for word in std_text for c in word]  # Flatten list of characters
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
    transcriptions = []
    
    for ts in speech_timestamps:
        segment_transcription = ""
        
        while i < len(word_spans):
            word_start = word_spans[i][0].start * ratio / sr
            word_end = word_spans[i][-1].end * ratio / sr
            # If word_start is after the end of the current speech segment,
            # then the current word is not part of this segment; go to next speech segment
            if ts["end"] < word_start:
                break
            
            # elif word_start is within the current speech segment but word_end is
            # # after the end of the current speech segment,
            # then the current word is partially included in this segment;
            # we include this word too and go to next speech segment
            elif word_start < ts["end"] <= word_end:
                segment_transcription += std_text[i] + " "
                i += 1
                break
            
            # elif word_end is within the current speech segment,
            # then the current word is fully included in this segment;
            # we include this word and continue to the next word
            elif word_end <= ts["end"]:
                segment_transcription += std_text[i] + " "
                i += 1
        
            else:
                raise ValueError("Unexpected case in segmentation.")

        transcriptions.append(segment_transcription.strip())
        
    # Map back to the original orthography before applying uroman
    norm_text = texts["norm_text"]
    word_idx = 0
    norm_transcriptions = []
    for std_transcription in transcriptions:
        std_words = std_transcription.split()
        orig_words = []
        for _ in std_words:
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
                    fa_model: torch.nn.Module,
                    vad_model: torch.nn.Module,
                    labels: List[str],
                    dictionary: Dict[str, int]) -> Dataset:
    """Segment audio files in the dataset based on silence detection."""
    def segment_wrapper(sample):
        return segment_audio(sample, fa_model, vad_model, labels, dictionary)
    
    segmented_dataset = dataset.map(segment_wrapper,
                                    remove_columns=dataset.column_names,
                                    num_proc=4)
    return segmented_dataset
        

if __name__ == "__main__":
    args = get_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    dataset = load_dataset(args.dataset_path_or_name)
    if dataset[0]["audio"]["sampling_rate"] != 16000:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    vad_model = load_silero_vad()
    
    bundle = prepare_fa_model()
    fa_model = bundle["model"]
    labels = bundle["labels"]
    dictionary = bundle["dictionary"]

    for i in tqdm.trange(len(dataset)):
        texts = get_transcript(dataset, i)
        std_text = texts["std_text"]
        tokens = [dictionary[c] for word in std_text for c in word]  # Flatten list of characters
        audio = dataset[i]["audio"]["array"] # np.ndarray
        
        emission = get_emission(audio, fa_model)
        aligned_tokens, alignment_scores = align(emission=emission,
                                                 tokens=tokens,
                                                 device=args.device)
        
        
        num_frames = emission.size(1)
        
        
        ... # TODO