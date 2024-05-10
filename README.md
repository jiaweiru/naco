# NACO

**N**eural **A**udio **CO**ding toolkit (NACO) compresses audio/speech into discrete codes for tasks such as codecs or speech synthesis.

## Overview

The aim of NACO is to build a framework for training as well as inference of neural codecs that incorporates state-of-the-art neural audio coding structures as well as quantization methods. In addition, NACO supports inference calls to various open source neural codecs.

- [x] Support official pre-trained model inference for AudioDEC, DAC, Encodec, FunCodec, HiFiCodec, and SNAC.
- [ ] Build the neural codec training framework


## Other pre-trained codecs usage

Install it using:

```bash
conda create -n naco python=3.9
conda activate naco
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install speechbrain descript-audio-codec encodec funcodec snac
pip install git+https://github.com/voidful/AudioDec.git
```

To encode (and decode) audio in Python, use the following code(learn more in `test_codec.py`):

```python
import torchaudio
from naco.codec import dac, encodec, funcodec, hificodec, audiodec, snac

def test_dac(mode, device):
    model_types = ["dac_16k_6kbps", "dac_24k_24kbps", "dac_44k_8kbps", "dac_44k_16kbps"]
    for model_type in model_types:
        codec = dac.DAC(model_type, device)
        print(model_type)
        print(
            f"Samplerate: {codec.sample_rate}, support bitrates: {codec.support_bitrates}"
        )
        waveform, sample_rate = torchaudio.load(f"./demo/0000_{model_type[4:6]}.wav")
        waveform = waveform.to(device)
        assert sample_rate == codec.sample_rate
        for i, nq in enumerate(codec.support_quantizers):
            if mode == "plain":
                resyn_waveform = codec.resyn(waveform, nq)
                code, stuff_for_synth = codec.extract_unit(waveform, nq)
                syn_waveform = codec.synth_unit(stuff_for_synth)
            elif mode == "norm_chunked":
                resyn_waveform = codec.resyn_norm_chunked(waveform, nq)
                code, stuff_for_synth = codec.extract_unit_norm_chunked(waveform, nq)
                syn_waveform = codec.synth_unit_norm_chunked(stuff_for_synth)

            print(
                f"Resyn length: {waveform.shape[-1]} -> {resyn_waveform.shape[-1]} or {syn_waveform.shape[-1]}"
            )
            print(
                f"Code size and dtype: {code.shape}, {code.dtype}, index from {torch.min(code)} to {torch.max(code)}"
            )
            torchaudio.save(
                f"./demo/0000_{model_type[4:6]}_{model_type}_{mode}_{codec.support_bitrates[i]}kbps.wav",
                resyn_waveform.cpu(),
                sample_rate,
                bits_per_sample=16,
            )
```


## Acknowledgements
https://github.com/voidful/Codec-SUPERB

https://github.com/facebookresearch/AudioDec

https://github.com/descriptinc/descript-audio-codec

https://github.com/alibaba-damo-academy/FunCodec

https://github.com/yangdongchao/AcademiCodec

https://github.com/hubertsiuzdak/snac
