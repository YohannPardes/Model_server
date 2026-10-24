# Audio samples

Short recordings of real human speech for trying the API by hand. Tests don't use them.

```bash
curl -X POST http://localhost:8000/api/v1/jobs -F "file=@samples/voices_sp0307_3s.wav"
```

| File | Length | Clips (4 s each) | Format | Source |
|---|---|---|---|---|
| `voices_sp0307_3s.wav` | 3.4 s | 1 | WAV | VOiCES, speaker 0307 (`Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042`) |
| `librispeech_2277_7s.flac` | 6.9 s | 2 | FLAC | LibriSpeech dev-clean, utterance `2277-149896-0008` |
| `librispeech_7902_10s.wav` | 9.7 s | 3 | WAV | LibriSpeech test-other, utterance `7902-96591-0014` |
| `librispeech_374_13s.mp3` | 13.0 s | 3 | MP3 | LibriSpeech train-clean-100, utterance `374-180298-0013` |
| `librispeech_6930_14s.flac` | 14.2 s | 4 | FLAC | LibriSpeech test-clean, utterance `6930-75918-0001` |

All files are 16 kHz mono and contain only **real** (bona fide) speech. There are no synthetic or
cloned voices in this folder.

## Licenses and attribution

- **VOiCES**: "Voices Obscured in Complex Environmental Settings (VOiCES) corpus" by SRI International
  and Lab41 (IQT Labs), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The file is unchanged. It is the copy published in the PyTorch audio tutorial assets.
- **LibriSpeech**: "LibriSpeech ASR corpus" by Vassil Panayotov, Guoguo Chen, Daniel Povey and
  Sanjeev Khudanpur, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  It is derived from public-domain [LibriVox](https://librivox.org) audiobooks. Changes: single
  utterances were taken from the [`sanchit-gandhi/librispeech_asr_dummy`](https://huggingface.co/datasets/sanchit-gandhi/librispeech_asr_dummy)
  copy and re-encoded to WAV, FLAC or MP3.
