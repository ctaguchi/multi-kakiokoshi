from datasets import Dataset, DatasetDict, Audio, concatenate_datasets
import torchaudio
import torch
# import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import os
import glob
from typing import List, Dict, Union, Any, Literal


DATA_DIR = "./data"
assert os.path.exists(DATA_DIR), f"{DATA_DIR} does not exist."


def map_audio(sample: Dict[str, Any],
              audio_dir: str) -> Dict[str, Any]:
    """Map audio file path to audio data."""
    # audio_path = sample["audio_file"]
    audio_path = os.path.join(audio_dir, sample["audio_file"])
    
    # loading audio with torchaudio can be buggy, so we use librosa as a fallback
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torch.from_numpy(waveform).float()
        # Convert to mono by averaging channels if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        speech = waveform.squeeze(0).numpy().astype(np.float32)
        
    except Exception as e:
        waveform, sample_rate = librosa.load(audio_path, sr=None)
        waveform = waveform.astype(np.float32)
        if waveform.ndim == 1:
            waveform = waveform[:, None]
        speech = waveform
    
    sample["audio"] = {"path": audio_path,
                       "array": speech,
                       "sampling_rate": sample_rate}
    return sample


def get_vocab_from_df(
        tsv: Union[str, pd.DataFrame, List[str], List[pd.DataFrame]]
    ) -> Dict[str, int]:
    """Get the custom vocabulary.

    Args:
        tsv (Union[str, pd.DataFrame, List[str], List[pd.DataFrame]]): Path(s) to the TSV file(s) or DataFrame(s).
    Returns:
        dict: Vocabulary dictionary mapping characters to indices.
    """
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


def load_tsv(tsv_file: str,
             audio_files: List[str]) -> pd.DataFrame:
    """Load TSV file into a pandas DataFrame."""
    audio_files = [os.path.basename(f) for f in audio_files]
    df = pd.read_csv(tsv_file, sep="\t")
    df = df[df["audio_file"].isin(audio_files)]
    df = df.dropna(subset=["transcription"]) # sometimes there are missing transcriptions
    return df


def prepare_dataset(dataset_dir: Literal["mcv-sps-st-09-2025", "common-voice-23"],
                    language: str,
                    num_proc: int = 1) -> Dataset:
    """Load dataset from the specified directory.
    For Common Voice datasets, `audio_dir` is named `clips`.
    For Spontaneous speech datasets, `audio_dir` is named `audio`.
    
    - Identify the audio files and corresponding TSV files based on the dataset structure.
    - Load the TSV file into a pandas DataFrame.
    - Filter the DataFrame to include only rows with audio files present in the audio directory.
    - Drop rows with missing transcriptions.
    - Convert the DataFrame to a Hugging Face Dataset.
    - Map the audio file paths to actual audio data.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        language (str): Language code for the dataset. E.g., "en", "de".
    Returns:
        Dataset: Loaded dataset.
    """
    if "mcv-sps-st-09-2025" in dataset_dir:
        lang_dir = language
        audio_dir = os.path.join(DATA_DIR,
                                 dataset_dir,
                                 lang_dir,
                                 "audios")
        audio_files = glob.glob(
            os.path.join(audio_dir, f"spontaneous-speech-{language}*")
        )
        assert len(audio_files) > 0, f"No audio files found in {audio_dir}"
        
        tsv_file = os.path.join(DATA_DIR,
                                dataset_dir,
                                lang_dir,
                                f"ss-corpus-{language}.tsv")
        df = load_tsv(tsv_file, audio_files)
        df["audio"] = df["audio_file"].apply(
            lambda fn: os.path.join(audio_dir, fn)
        )
        # print(f"Loaded {len(df)} samples for language {language} from {tsv_file}")
        
        # vocab = get_vocab_from_df(df)
        
        # Convert df to the dataset
        dataset = Dataset.from_pandas(df)
        dataset = DatasetDict({"train": dataset})
        # sponetaneous speech corpus contains the following columns:
        # client_id, audio_id, audio_file, duration_ms, prompt_id, prompt, transcription, votes, age, gender, language, split, char_per_sec, quality_tags
        # From audio file, we need to create the "audio" column
        # print(audio_dir)
        # dataset = dataset.map(map_audio,
        #                       num_proc=num_proc,
        #                     #   with_indices=True, # for debugging
        #                       fn_kwargs={"audio_dir": audio_dir})
        dataset = dataset.cast_column("audio",
                                      Audio(sampling_rate=16000, mono=True))
        
    elif "common-voice-23" in dataset_dir:
        lang_dir = language
        audio_dir = os.path.join(DATA_DIR,
                                 dataset_dir,
                                 lang_dir,
                                 "clips")
        audio_files = glob.glob(
            os.path.join(audio_dir, f"commo_voice_{language}_*")
        )
        assert len(audio_files) > 0, f"No audio files found in {audio_dir}"
        
        train_tsv_file = os.path.join(DATA_DIR,
                                      dataset_dir,
                                      lang_dir,
                                      "train.tsv")
        dev_tsv_file = os.path.join(DATA_DIR,
                                    dataset_dir,
                                    lang_dir,
                                    "dev.tsv")
        test_tsv_file = os.path.join(DATA_DIR,
                                     dataset_dir,
                                     lang_dir,
                                     "test.tsv")
        df_train = load_tsv(train_tsv_file, audio_files)
        df_dev = load_tsv(dev_tsv_file, audio_files)
        df_test = load_tsv(test_tsv_file, audio_files)
        
        dataset_train = Dataset.from_pandas(df_train)
        dataset_dev = Dataset.from_pandas(df_dev)
        dataset_test = Dataset.from_pandas(df_test)
        dataset = Dataset.from_dict({
            "train": dataset_train,
            "dev": dataset_dev,
            "test": dataset_test
        })
        # common voice dataset contains the following columns:
        # client_id, path, sentence_id, sentence, sentence_domain, up_votes, down_votes, age, gender, accents, variant, locale, segment
        dataset = dataset.map(map_audio,
                              num_proc=num_proc,
                              fn_kwargs={"audio_dir": audio_dir})
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
    else:
        raise ValueError(f"Unsupported dataset directory: {dataset_dir}")

    return dataset


BETAWI_NUMBERS = {
    "1": "satu",
    "2": "dua",
    "3": "tiga",
    "4": "empat",
    "5": "lima",
    "6": "enam",
    "7": "tujuh",
    "8": "delapan",
    "9": "sembilan",
    "0": "nol",
    "10": "sepuluh",
    "11": "sebelas",
    "12": "dua belas",
    "13": "tiga belas",
    "14": "empat belas",
    "15": "lima belas",
    "16": "enam belas",
    "17": "tujuh belas",
    "18": "delapan belas",
    "19": "sembilan belas",
    "20": "dua puluh",
    "21": "dua puluh satu",
    "22": "dua puluh dua",
    "23": "dua puluh tiga",
    "24": "dua puluh empat",
    "25": "dua puluh lima",
    "26": "dua puluh enam",
    "27": "dua puluh tujuh",
    "28": "dua puluh delapan",
    "29": "dua puluh sembilan",
    "30": "tiga puluh",
    "31": "tiga puluh satu",
    "32": "tiga puluh dua",
    "33": "tiga puluh tiga",
    "34": "tiga puluh empat",
    "35": "tiga puluh lima",
    "36": "tiga puluh enam",
    "37": "tiga puluh tujuh",
    "38": "tiga puluh delapan",
    "39": "tiga puluh sembilan",
    "40": "empat puluh",
    "50": "lima puluh",
    "60": "enam puluh",
    "70": "tujuh puluh",
    "80": "delapan puluh",
    "90": "sembilan puluh",
    "100": "seratus",
    "500": "lima ratus",
    "2002": "dua ribu dua",
    "2023": "dua ribu tiga", # this is not correct, but it is how they correspond in the audio and transcription in spontaneous-speech-bew-21256.mp3
}

BUKUSU_NUMBERS = {
    "2006": "two thousand and six",
}


def concat_with_random_silence(
    audio_list: List[np.ndarray],
    sr: int = 16_000,
    max_silence_sec: float = 1.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Concatenate audio arrays with random silence (0..max_silence_sec) between each.
    """
    if rng is None:
        rng = np.random.default_rng()

    chunks = []
    for i, arr in enumerate(audio_list):
        arr = arr.astype(np.float32)
        chunks.append(arr)
        if i < len(audio_list) - 1:
            silence_sec = rng.uniform(0.0, max_silence_sec)
            silence = np.zeros(int(sr * silence_sec), dtype=np.float32)
            chunks.append(silence)

    return np.concatenate(chunks)


from datasets import Dataset
import numpy as np

def build_augmented_for_client(
    ds_client: Dataset,
    text_col: str = "text",
    target_duration_sec: float = 30.0,
    sr: int = 16_000,
    max_silence_sec: float = 1.0,
    min_segments: int = 2,
    max_num_sample: int = 20,
    rng: np.random.Generator = None,
) -> Dataset:
    """
    Create an augmented dataset for a single client_id.
    Continues creating augmented samples until max_num_sample is reached.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Randomize sample order
    indices = np.arange(len(ds_client))
    rng.shuffle(indices)

    new_audio = []
    new_texts = []
    new_client_ids = []

    current_audio_chunks = []
    current_text_chunks = []
    current_duration = 0.0
    current_n_segs = 0

    client_id_value = ds_client[0]["client_id"]

    # Continue looping **until we have enough augmented samples**
    i = 0
    while len(new_audio) < max_num_sample:
        idx = indices[i % len(indices)]     # wrap around for repeated sampling
        i += 1

        item = ds_client[int(idx)]
        audio_dict = item["audio"]

        arr = audio_dict["array"]
        this_sr = audio_dict["sampling_rate"]
        if this_sr != sr:
            raise ValueError(f"Sampling rate mismatch: {this_sr} != {sr}")

        dur = len(arr) / sr

        # If adding this would overflow target length and we already have enough segments,
        # finalize the current sample.
        if current_n_segs >= min_segments and current_duration + dur > target_duration_sec:
            concat_audio = concat_with_random_silence(
                current_audio_chunks,
                sr=sr,
                max_silence_sec=max_silence_sec,
                rng=rng,
            )
            concat_text = " ".join(current_text_chunks)

            new_audio.append(concat_audio)
            new_texts.append(concat_text)
            new_client_ids.append(client_id_value)

            # Check if we reached target number of augments
            if len(new_audio) >= max_num_sample:
                break

            # reset buffers for next augmented sample
            current_audio_chunks = []
            current_text_chunks = []
            current_duration = 0.0
            current_n_segs = 0

        # Add current segment
        current_audio_chunks.append(arr)
        current_text_chunks.append(item[text_col])
        current_duration += dur
        current_n_segs += 1

    # Optional: finalize leftover if you want
    # (only if len(new_audio) < max_num_sample)
    if (
        len(new_audio) < max_num_sample
        and current_n_segs >= min_segments
    ):
        concat_audio = concat_with_random_silence(
            current_audio_chunks,
            sr=sr,
            max_silence_sec=max_silence_sec,
            rng=rng,
        )
        concat_text = " ".join(current_text_chunks)

        new_audio.append(concat_audio)
        new_texts.append(concat_text)
        new_client_ids.append(client_id_value)

    # Build dataset
    augmented_ds = Dataset.from_dict(
        {
            "audio": new_audio,
            text_col: new_texts,
            "client_id": new_client_ids,
        }
    )

    return augmented_ds


def wrap_audio(example, sr=16000):
    # example["audio"] is currently a list or np.ndarray
    arr = np.asarray(example["audio"], dtype=np.float32)
    return {
        "audio": {
            "array": arr,
            "sampling_rate": sr,
        }
    }


def build_augmented_cv(ds: Dataset,
                       target_duration_sec: float = 30.0,
                       num_proc: int = 4):
    rng = None
    client_ids = ds.unique("client_id")
    augmented_by_client = []
    for cid in client_ids:
        ds_client = ds.filter(lambda x, cid=cid: x["client_id"] == cid)
        aug_client = build_augmented_for_client(
            ds_client,
            text_col="transcription",
            target_duration_sec=target_duration_sec,
            sr=16_000,
            max_silence_sec=1.0,
            min_segments=2,
            rng=rng,
        )
        augmented_by_client.append(aug_client)

    # augmented_ds = concatenate_datasets(augmented_by_client)
    augmented_ds = concatenate_datasets(augmented_by_client)
    augmented_ds = augmented_ds.map(wrap_audio,
                                    num_proc=num_proc)
    augmented_ds = augmented_ds.cast_column("audio", Audio(sampling_rate=16000))
    return augmented_ds