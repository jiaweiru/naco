# Warning: An import error occurs when using the current script as the main program.
# Because the filename is the same as the package "funcodec".
import torch
from funcodec.bin.codec_inference import Speech2Token

import math
import typing as tp
from pathlib import Path
from urllib.request import urlretrieve

from .utils import pad_audio

# segment=null, audio_normalize=True for all models.
FUNCODEC_MODEL_URL = {
    # FreqCodec
    "funcodec_en_libritts-16k-gr1nq32ds320": {
        "ckpt": "https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/resolve/main/model.pth",
        "config": "https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/raw/main/config.yaml",
    },
    "funcodec_en_libritts-16k-gr8nq32ds320": {
        "ckpt": "https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch/resolve/main/model.pth",
        "config": "https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch/raw/main/config.yaml",
    },
    # EnCodec
    "funcodec_en_libritts-16k-nq32ds320": {
        "ckpt": "https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch/resolve/main/model.pth",
        "config": "https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch/raw/main/config.yaml",
    },
    "funcodec_en_libritts-16k-nq32ds640": {
        "ckpt": "https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch/resolve/main/model.pth",
        "config": "https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch/raw/main/config.yaml",
    },
    "funcodec_zh_en_general_16k_nq32ds320": {
        "ckpt": "https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch/resolve/main/model.pth",
        "config": "https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch/raw/main/config.yaml",
    },
    "funcodec_zh_en_general_16k_nq32ds640": {
        "ckpt": "https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/resolve/main/model.pth",
        "config": "https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/raw/main/config.yaml",
    },
}


class FunCodec:
    """
    FunCodec. https://github.com/alibaba-damo-academy/FunCodec
    Here we provide the inference implementation of the official repository.
    """

    def __init__(self, model_type: str, device: str = "cpu") -> None:
        model_path = Path.home().joinpath(f".cache/funcodec/{model_type}")
        config_path = model_path.joinpath("config.yaml")
        ckpt_path = model_path.joinpath("model.pth")

        if not (config_path.exists() and ckpt_path.exists()):
            model_path.mkdir(parents=True, exist_ok=True)
            config_url = FUNCODEC_MODEL_URL[model_type]["config"]
            ckpt_url = FUNCODEC_MODEL_URL[model_type]["ckpt"]
            urlretrieve(config_url, config_path)
            urlretrieve(ckpt_url, ckpt_path)

        self.model = Speech2Token(config_path, ckpt_path, device=device)

        self.sample_rate = self.model.model.sample_rate
        self.support_bitrates = self.get_funcodec_bitrates()

    def get_funcodec_bitrates(self) -> tp.List[float]:
        codebook_size = self.model.model_args.quantizer_conf["codebook_size"]
        num_quantizers = self.model.model_args.quantizer_conf["num_quantizers"]
        rand_num_quant = self.model.model_args.quantizer_conf["rand_num_quant"]
        hop_length = self.model.model_args.quantizer_conf["encoder_hop_length"]
        max_bitrate = (
            self.sample_rate / hop_length * num_quantizers * math.log2(codebook_size)
        )
        support_bitrates = [
            round(max_bitrate * num_quant / num_quantizers / 1_000, 1)
            for num_quant in rand_num_quant
        ]
        return support_bitrates

    @torch.inference_mode()
    def resyn(
        self, audio: torch.Tensor, bitrate: tp.Optional[float] = None
    ) -> torch.Tensor:
        if bitrate is None:
            bitrate = self.support_bitrates[-1]
        assert bitrate in self.support_bitrates
        length = audio.shape[-1]
        hop_length = self.model.model_args.quantizer_conf["encoder_hop_length"]
        audio = pad_audio(audio, hop_length)
        code, code_emb, _, _ = self.model(
            audio, bit_width=bitrate * 1_000, run_mod="encode"
        )
        scale = code_emb[0][1]
        _, _, resyn_audio, _ = self.model(code[0].permute(1, 2, 0), run_mod="decode")
        return resyn_audio.squeeze(0)[:, :length] * scale

    @torch.inference_mode()
    def extract_unit(
        self, audio: torch.Tensor, bitrate: tp.Optional[float] = None
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, int, torch.Tensor]]:
        if bitrate is None:
            bitrate = self.support_bitrates[-1]
        assert bitrate in self.support_bitrates
        length = audio.shape[-1]
        hop_length = self.model.model_args.quantizer_conf["encoder_hop_length"]
        audio = pad_audio(audio, hop_length)
        code, code_emb, _, _ = self.model(
            audio, bit_width=bitrate * 1_000, run_mod="encode"
        )
        scale = code_emb[0][1]
        return code[0].permute(1, 0, 2).squeeze(0), (
            code[0].permute(1, 2, 0),
            length,
            scale,
        )

    @torch.inference_mode()
    def synth_unit(
        self, stuff_for_synth: tp.Tuple[torch.Tensor, int, torch.Tensor]
    ) -> torch.Tensor:
        code, length, scale = stuff_for_synth
        _, _, syn_audio, _ = self.model(code, run_mod="decode")
        return syn_audio.squeeze(0)[:, :length] * scale
