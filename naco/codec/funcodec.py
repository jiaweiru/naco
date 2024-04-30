import math
import torch
from funcodec.bin.codec_inference import Speech2Token

from pathlib import Path
from urllib.request import urlretrieve
import typing as tp


# segment=null, audio_normalize=True for all models.
FUNCODEC_MODEL_TYPE = {
    # FreqCodec
    "funcodec_en_libritts-16k-gr1nq32ds320": "https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch",
    "funcodec_en_libritts-16k-gr8nq32ds320": "https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch",
    # EnCodec
    "funcodec_en_libritts-16k-nq32ds320": "https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch",
    "funcodec_en_libritts-16k-nq32ds640": "https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch",
    "funcodec_zh_en_general_16k_nq32ds320": "https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch",
    "funcodec_zh_en_general_16k_nq32ds640": "https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch",
}


class FunCodec:
    """
    FunCodec. https://github.com/alibaba-damo-academy/FunCodec
    Here we provide the inference implementation of CodecSUPERB and the official repository,
    the only difference between them is that the CodecSUPERB implementation directly sets
    `run_mod="inference"` to achieve the normalization before and after coding.
    """

    def __init__(self, model_type: str, device: str = "cpu") -> None:
        model_path = Path.home().joinpath(f".cache/funcodec/{model_type}")
        config_path = model_path.joinpath("config.yaml")
        ckpt_path = model_path.joinpath("model.pth")

        if not (config_path.exists() and ckpt_path.exists()):
            model_path.mkdir(parents=True, exist_ok=True)
            config_url = FUNCODEC_MODEL_TYPE[model_type] + "/raw/main/config.yaml"
            ckpt_url = FUNCODEC_MODEL_TYPE[model_type] + "/resolve/main/model.pth"
            urlretrieve(config_url, config_path)
            urlretrieve(ckpt_url, ckpt_path)

        self.model = Speech2Token(config_path, ckpt_path, device=device)
        self.support_bitrate = self.get_funcodec_bitrate()

    def get_funcodec_bitrate(self) -> tp.List[float]:
        sample_rate = self.model.model.sample_rate
        encoder_hop_length = self.model.model_args.quantizer_conf["encoder_hop_length"]
        codebook_size = self.model.model_args.quantizer_conf["codebook_size"]
        num_quantizers = self.model.model_args.quantizer_conf["num_quantizers"]
        rand_num_quant = self.model.model_args.quantizer_conf["rand_num_quant"]

        max_bitrate = (
            sample_rate / encoder_hop_length * num_quantizers * math.log2(codebook_size)
        )
        support_bitrate = [
            max_bitrate * num_quant / num_quantizers / 1_000
            for num_quant in rand_num_quant
        ]

        return support_bitrate

    @torch.inference_mode()
    def resyn(
        self, audio: torch.Tensor, sample_rate: int, bitrate: float
    ) -> torch.Tensor:
        # Check the sample rate and bitrate
        assert sample_rate == self.model.model.sample_rate
        assert bitrate in self.support_bitrate

        length = audio.shape[-1]
        code, code_emb, _, _ = self.model(
            audio, bit_width=bitrate * 1_000, run_mod="encode"
        )
        scale = code_emb[0][1]
        _, _, resyn_audio, _ = self.model(code[0].permute(1, 2, 0), run_mod="decode")
        return resyn_audio.squeeze(0)[:, :length] * scale

    @torch.inference_mode()
    def extract_unit(
        self, audio: torch.Tensor, sample_rate: int, bitrate: float
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, int, torch.Tensor]]:
        # Check the sample rate and bitrate
        assert sample_rate == self.model.model.sample_rate
        assert bitrate in self.support_bitrate

        length = audio.shape[-1]
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
