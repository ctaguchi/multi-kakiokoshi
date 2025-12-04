# multi-kakiokoshi

## Strategy
- Add any additional data (e.g. from the Bible)
- Segment into smaller bits, and combine them for various lengths

## (Future work) More strategies
- Pre-fine-tune a pre-trained multilingual self-supervised speech model with a high/mid-resource similar language with the same writing system.
  - This is for both fitting the model to the language's phonology and also making
- Do additional fine-tuning with the target low-resource language.

## Preprocessing
The text preprocessing includes:
- Segment the audio and the aligned text, using VAD and forced alignment.
- Normalize the text with `uroman` for forced alignment.
- Convert numerals into spelled-out numbers
- Deal with other irregular characters (such as strange punctuation)

Below are some language-specific preprocessing
- In Cypriot Greek, we found that there are ζ̆ and ž depending on transcriptions. These are unified as ζ̆.

### Check if there's any numeral
From the `src` directory, run the following (example with `el-CY`)
- `uv run python utils/check_numerals.py -l el-CY`

### To run segmentation
From the `src` directory, run the following (example with `aln`)
- `uv run python -m segment.segment -d mcv-sps-st-09-2025 -l aln -n 2`

### Upload the completed dataset
From the `src` directory, run the following (example with `aln`)
- `uv run python utils/upload_dataset.py -d data/mcv-sps-st-09-2025/aln/segmented_dataset -n mcv-sps-aln-segmented`

## List of languages
| Code | Language | Script | Similar |
|------| -------- | ------ | ------- |
| aln | Gheg Albanian | Latin | Albanian |
| bew | Betawi | Latin | Indonesian, Malay |
| bxk | Bukusu | Latin | |
| el-CY | Cypriot Greek | Greek | Greek |
| cgg | Chiga | Latin | |
| hch | Wixárika | Latin | |
| kcn | Nubi | Latin | |
| koo | Konzo | Latin | |
| led | Lendu | Latin | |
| lke | Kenyi | Latin | |
| lth | Thur | Latin | |
| meh | Southwestern Tlaxiaco Mixtec | Latin | |
| mmc | Michoacán Mazahua | Latin | |
| pne | Western Penan | Latin | |
| ruc | Ruuli | Latin | |
| rwm | Amba | Latin | |
| sco | Scots | Latin | English |
| tob | Toba Qom | Latin | |
| top | Papantla Totonac | Latin | |
| ttj | Rutoro | Latin | |
| ukv | Kuku | Latin | |

## Before running experiments
Before training ASR models, make sure to:
- remove non-phonetic symbols that don't bare any pronunciations, such as quotation marks, commas, periods
  - regex would be: # TODO
  - Full Latin/Greek alphabet with the special characters
- remove duplicate whitespaces ("a  b" -> "a b")

Note that this text preprocessing is not the same as the one used for the test data officially (https://www.codabench.org/competitions/10820/?ref=community.mozilladatacollective.com#/pages-tab).

## 