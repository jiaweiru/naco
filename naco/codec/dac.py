# Warning: An import error occurs when using the current script as the main program.
# Because the filename is the same as the package "dac".
import torch
import dac
from audiotools import AudioSignal

import typing as tp

DAC_MODEL_TYPE = {
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
        model_sample_rate, model_bitrate = DAC_MODEL_TYPE[model_type]
        self.model_path = dac.utils.download(model_sample_rate, model_bitrate)
        self.model = dac.DAC.load(self.model_path)
        self.model.to(device).eval()

    @torch.inference_mode()
    def resyn(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """The DAC currently only supports encoding using all quantizers,
        in fact it supports encoding at any bitrate from 0 to the total number of quantizers
        """
        # Check the sample rate and padding
        padded_audio = self.model.preprocess(audio.unsqueeze(0), sample_rate)
        length = audio.shape[-1]
        z, _, _, _, _ = self.model.encode(padded_audio)
        resyn_audio = self.model.decode(z)
        return resyn_audio.squeeze(0)[:, :length]

    @torch.inference_mode()
    def extract_unit(
        self, audio: torch.Tensor, sample_rate: int
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, int]]:
        # Check the sample rate and padding
        padded_audio = self.model.preprocess(audio.unsqueeze(0), sample_rate)
        length = audio.shape[-1]
        z, codes, _, _, _ = self.model.encode(padded_audio)
        return codes.squeeze(0), (z, length)

    @torch.inference_mode()
    def synth_unit(self, stuff_for_synth: tp.Tuple[torch.Tensor, int]) -> torch.Tensor:
        z, length = stuff_for_synth
        synth_audio = self.model.decode(z)
        return synth_audio.squeeze(0)[:, :length]

    @torch.inference_mode()
    def resyn_norm_chunked(
        self, audio: torch.Tensor, sample_rate: int, win_duration: float = 15.0
    ) -> torch.Tensor:
        """Avoid OOM by using the official chunking method for long audio."""
        audio = AudioSignal(audio.squeeze(0), sample_rate)
        audio_dac = self.model.compress(audio, win_duration)
        resyn_audio = self.model.decompress(audio_dac).audio_data
        return resyn_audio.squeeze(0)

    @torch.inference_mode()
    def extract_unit_norm_chunked(
        self, audio: torch.Tensor, sample_rate: int, win_duration: float = 15.0
    ) -> tp.Tuple[torch.Tensor, dac.DACFile]:
        # Get the DACFile object from audio
        audio = AudioSignal(audio.squeeze(0), sample_rate)
        audio_dac = self.model.compress(audio, win_duration)

        codes = audio_dac.codes.squeeze(0)
        return codes, audio_dac

    @torch.inference_mode()
    def synth_unit_norm_chunked(self, stuff_for_synth: dac.DACFile) -> torch.Tensor:
        synth_audio = self.model.decompress(stuff_for_synth).audio_data
        return synth_audio.squeeze(0)
