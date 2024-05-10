import yaml
import math
import zipfile
import typing as tp
from pathlib import Path
from urllib.request import urlretrieve

import torch
from AudioDec.utils import audiodec

from .utils import pad_audio

AUDIODEC_PRETRAINED_URL = "https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip"
AUDIODEC_MODEL_URL = {
    "AudioDec_v1_symAD_vctk_48000_hop300_clean": {
        "stat": "https://github.com/facebookresearch/AudioDec/raw/main/stats/symAD_vctk_48000_hop300_clean.npy"
    },
    "AudioDec_v1_symAD_libritts_24000_hop300_clean": {
        "stat": "https://github.com/facebookresearch/AudioDec/raw/main/stats/symAD_libritts_24000_hop300_clean.npy"
    },
}
AUDIODEC_MODEL_PATH = {
    "AudioDec_v1_symAD_vctk_48000_hop300_clean": {
        "encoder": {
            "ckpt": "exp/autoencoder/symAD_vctk_48000_hop300/checkpoint-200000steps.pkl"
        },
        "vocoder": {
            "ckpt": "exp/vocoder/AudioDec_v1_symAD_vctk_48000_hop300_clean/checkpoint-500000steps.pkl",
            "config": "exp/vocoder/AudioDec_v1_symAD_vctk_48000_hop300_clean/config.yml",
        },
    },
    "AudioDec_v1_symAD_libritts_24000_hop300_clean": {
        "encoder": {
            "ckpt": "exp/autoencoder/symAD_libritts_24000_hop300/checkpoint-500000steps.pkl"
        },
        "vocoder": {
            "ckpt": "exp/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/checkpoint-500000steps.pkl",
            "config": "exp/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/config.yml",
        },
    },
}


def modify_audiodec_yml(yml_file, key=["generator_params", "stats"], value=None):
    with open(yml_file, "r") as f:
        yml_data = yaml.safe_load(f)

    current_dict = yml_data
    for k in key[:-1]:
        current_dict = yml_data[k]
    current_dict[key[-1]] = value

    with open(yml_file, "w") as f:
        yaml.dump(yml_data, f)


class AudioDEC:
    """
    AudioDEC. https://github.com/facebookresearch/AudioDec
    Supports inference for AudeoDEC v1 models, officially available pre-trained on VCTK &
    LibriTTS respectively.
    """

    def __init__(self, model_type: str, device: str = "cpu") -> None:
        audiodec_path = Path.home().joinpath(f".cache/audiodec")
        if not audiodec_path.exists():
            audiodec_path.mkdir()
            urlretrieve(AUDIODEC_PRETRAINED_URL, audiodec_path.joinpath("exp.zip"))
            with zipfile.ZipFile(audiodec_path.joinpath("exp.zip")) as z:
                z.extractall(audiodec_path)

        stat_path = audiodec_path.joinpath(
            model_type, AUDIODEC_MODEL_URL[model_type]["stat"].split("/")[-1]
        )
        if not stat_path.exists():
            audiodec_path.joinpath(model_type).mkdir()
            urlretrieve(AUDIODEC_MODEL_URL[model_type]["stat"], stat_path)
            modify_audiodec_yml(
                audiodec_path.joinpath(
                    AUDIODEC_MODEL_PATH[model_type]["vocoder"]["config"]
                ),
                value=str(stat_path),
            )
        encoder_ckpt_path = audiodec_path.joinpath(
            AUDIODEC_MODEL_PATH[model_type]["encoder"]["ckpt"]
        )
        vocoder_ckpt_path = audiodec_path.joinpath(
            AUDIODEC_MODEL_PATH[model_type]["vocoder"]["ckpt"]
        )

        self.model = audiodec.AudioDec(tx_device=device, rx_device=device)
        self.model.load_transmitter(encoder_ckpt_path)
        self.model.load_receiver(encoder_ckpt_path, vocoder_ckpt_path)

        self.autoencoder_config = self.model._load_config(encoder_ckpt_path)
        self.sample_rate = self.autoencoder_config["sampling_rate"]
        self.support_bitrates = self.get_audiodec_bitrates()

    def get_audiodec_bitrates(self) -> tp.List[float]:
        codebook_size = self.autoencoder_config["generator_params"]["codebook_size"]
        codebook_num = self.autoencoder_config["generator_params"]["codebook_num"]
        hop_length = math.prod(
            self.autoencoder_config["generator_params"]["enc_strides"]
        )
        bitrate = (
            self.sample_rate / hop_length * codebook_num * math.log2(codebook_size)
        )
        support_bitrates = [round(bitrate / 1_000, 1)]
        return support_bitrates

    @torch.inference_mode()
    def resyn(self, audio: torch.Tensor) -> torch.Tensor:
        length = audio.shape[-1]
        hop_length = math.prod(
            self.autoencoder_config["generator_params"]["enc_strides"]
        )
        audio = pad_audio(audio, hop_length)
        self.model.tx_encoder.reset_buffer()
        z = self.model.tx_encoder.encode(audio.unsqueeze(0))
        zq, _ = self.model.tx_encoder.quantizer.codebook.forward_index(
            z.transpose(2, 1), flatten_idx=False
        )
        resyn_audio = self.model.decoder.decode(zq)

        return resyn_audio.squeeze(0)[:, :length]

    @torch.inference_mode()
    def extract_unit(
        self, audio: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, int]]:
        length = audio.shape[-1]
        hop_length = math.prod(
            self.autoencoder_config["generator_params"]["enc_strides"]
        )
        audio = pad_audio(audio, hop_length)
        self.model.tx_encoder.reset_buffer()
        z = self.model.tx_encoder.encode(audio.unsqueeze(0))
        _, codes = self.model.tx_encoder.quantizer.codebook.forward_index(
            z.transpose(2, 1), flatten_idx=False
        )
        return codes, (codes, length)

    @torch.inference_mode()
    def synth_unit(self, stuff_for_synth: tp.Tuple[torch.Tensor, int]) -> torch.Tensor:
        codes, length = stuff_for_synth
        codebook_size = self.autoencoder_config["generator_params"]["codebook_size"]
        zq = self.model.rx_encoder.lookup(
            codes
            + torch.arange(codes.shape[0]).reshape(-1, 1).to(codes.device)
            * codebook_size
        )
        syn_audio = self.model.decoder.decode(zq)
        return syn_audio.squeeze(0)[:, :length]
