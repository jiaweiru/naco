# Warning: An import error occurs when using the current script as the main program.
# Because the filename is the same as the package "encodec".
import math
import typing as tp

import torch
import encodec

from .utils import pad_audio

ENCODEC_MODEL_MAP = {
    # segment_dur=1.0, audio_normalize=True, non-causal music-only stereo model.
    "encodec_48k_24kbps": encodec.EncodecModel.encodec_model_48khz,
    # segment_dur=None, audio_normalize=False, causal audio mono model.
    "encodec_24k_24kbps": encodec.EncodecModel.encodec_model_24khz,
}


class EnCodec:
    """
    Encodec. https://github.com/facebookresearch/encodec
    EnCodec does not give a solution for OOM caused by long audio, in fact, codecs often use
    streaming in practice rather than chunking into several short pieces of audio.
    In addition, EnCodec takes into account a larger gain range in the preprocessing of the
    training data, which makes it unnecessary to normalize the audio before codec processing.
    """

    def __init__(self, model_type: str, device: str = "cpu") -> None:
        self.model = ENCODEC_MODEL_MAP[model_type]()
        self.model.to(device)

        self.sample_rate = self.model.sample_rate
        self.support_bitrates = self.model.target_bandwidths
        self.channels = self.model.channels

    @torch.inference_mode()
    def resyn(
        self, audio: torch.Tensor, bitrate: tp.Optional[float] = None
    ) -> torch.Tensor:
        if bitrate is None:
            bitrate = self.support_bitrates[-1]
        assert bitrate in self.support_bitrates
        length = audio.shape[-1]
        hop_length = math.prod(self.model.encoder.ratios)
        audio = pad_audio(audio, hop_length)
        self.model.set_target_bandwidth(bitrate)
        encoded_frames = self.model.encode(audio.unsqueeze(0))
        resyn_audio = self.model.decode(encoded_frames).squeeze(0)
        return resyn_audio[:, :length]

    @torch.inference_mode()
    def extract_unit(
        self, audio: torch.Tensor, bitrate: tp.Optional[float] = None
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[encodec.model.EncodedFrame, int]]:
        if bitrate is None:
            bitrate = self.support_bitrates[-1]
        assert bitrate in self.support_bitrates
        length = audio.shape[-1]
        hop_length = math.prod(self.model.encoder.ratios)
        audio = pad_audio(audio, hop_length)
        self.model.set_target_bandwidth(bitrate)
        encoded_frames = self.model.encode(audio.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze(0)
        return codes.squeeze(0), (encoded_frames, length)

    @torch.inference_mode()
    def synth_unit(
        self, stuff_for_synth: tp.Tuple[encodec.model.EncodedFrame, int]
    ) -> torch.Tensor:
        encoded_frames, length = stuff_for_synth
        synth_audio = self.model.decode(encoded_frames).squeeze(0)
        return synth_audio[:, :length]
