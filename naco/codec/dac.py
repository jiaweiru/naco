# Warning: An import error occurs when using the current script as the main program.
# Because the filename is the same as the package "dac".
import torch
import dac
from audiotools import AudioSignal

import math
import typing as tp

DAC_MODEL_MAP = {
    "dac_16k_8kbps": ("16khz", "8kbps"),
    "dac_24k_8kbps": ("24khz", "8kbps"),
    "dac_44k_8kbps": ("44khz", "8kbps"),
    "dac_44k_16kbps": ("44khz", "16kbps"),
}


class DAC:
    """
    Descript-audio-coding. https://github.com/descriptinc/descript-audio-codec
    Use the two ways recommended by the official repository for DAC, which have the following
    characteristics respectively:
    1. Direct input the audio into the model.
    2. Long audio is processed using chunking, in addition to loudness normalization of the audio
    segments, which is related to the way the training data of the model is preprocessed.
    """

    def __init__(self, model_type: str, device: str = "cpu") -> None:
        self.model_path = dac.utils.download(*DAC_MODEL_MAP[model_type])
        self.model = dac.DAC.load(self.model_path)
        self.model.to(device).eval()

        self.sample_rate = self.model.sample_rate
        self.hop_length = self.model.hop_length
        self.support_bitrate = self.get_dac_bitrates()

    def get_dac_bitrates(self):
        n_codebooks = self.model.n_codebooks
        codebook_size = self.model.codebook_size
        max_bitrate = (
            self.sample_rate / self.hop_length * n_codebooks * math.log2(codebook_size)
        )
        support_bitrates = [
            round(max_bitrate * i / n_codebooks / 1_000, 2) for i in range(n_codebooks)
        ]

        return support_bitrates

    @torch.inference_mode()
    def resyn(self, audio: torch.Tensor) -> torch.Tensor:
        """The DAC currently only supports encoding using all quantizers,
        in fact it supports encoding at any bitrate from 0 to the total number of quantizers
        """
        # Check the sample rate and padding
        padded_audio = self.model.preprocess(audio.unsqueeze(0), self.sample_rate)
        length = audio.shape[-1]
        zq, _, _, _, _ = self.model.encode(padded_audio)
        resyn_audio = self.model.decode(zq)
        return resyn_audio.squeeze(0)[:, :length]

    @torch.inference_mode()
    def extract_unit(
        self, audio: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, int]]:
        # Check the sample rate and padding
        padded_audio = self.model.preprocess(audio.unsqueeze(0), self.sample_rate)
        length = audio.shape[-1]
        zq, codes, _, _, _ = self.model.encode(padded_audio)
        return codes.squeeze(0), (zq, length)

    @torch.inference_mode()
    def synth_unit(self, stuff_for_synth: tp.Tuple[torch.Tensor, int]) -> torch.Tensor:
        zq, length = stuff_for_synth
        synth_audio = self.model.decode(zq)
        return synth_audio.squeeze(0)[:, :length]

    @torch.inference_mode()
    def resyn_norm_chunked(
        self, audio: torch.Tensor, win_duration: float = 15.0
    ) -> torch.Tensor:
        """Avoid OOM by using the official chunking method for long audio."""
        audio = AudioSignal(audio.squeeze(0), self.sample_rate)
        audio_dac = self.model.compress(audio, win_duration)
        resyn_audio = self.model.decompress(audio_dac).audio_data
        return resyn_audio.squeeze(0)

    @torch.inference_mode()
    def extract_unit_norm_chunked(
        self, audio: torch.Tensor, win_duration: float = 15.0
    ) -> tp.Tuple[torch.Tensor, dac.DACFile]:
        # Get the DACFile object from audio
        audio = AudioSignal(audio.squeeze(0), self.sample_rate)
        audio_dac = self.model.compress(audio, win_duration)

        codes = audio_dac.codes.squeeze(0)
        return codes, audio_dac

    @torch.inference_mode()
    def synth_unit_norm_chunked(self, stuff_for_synth: dac.DACFile) -> torch.Tensor:
        synth_audio = self.model.decompress(stuff_for_synth).audio_data
        return synth_audio.squeeze(0)
