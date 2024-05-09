# Warning: An import error occurs when using the current script as the main program.
# Because the filename is the same as the package "snac".
import torch
import snac

import math
import typing as tp

SNAC_MODEL_MAP = {
    "snac_24khz_0.98kbps": "hubertsiuzdak/snac_24khz",
    "snac_32khz_1.9kbps": "hubertsiuzdak/snac_32khz",
    "snac_44khz_2.6kbps": "hubertsiuzdak/snac_44khz",
}


class SNAC:
    """
    SNAC. https://github.com/hubertsiuzdak/snac
    """

    def __init__(self, model_type: str, device: str = "cpu") -> None:
        self.model = snac.SNAC.from_pretrained(SNAC_MODEL_MAP[model_type])
        self.model = self.model.to(device).eval()

        self.sample_rate = self.model.sampling_rate
        # The pad also depends on the parameters of quantization and local attention.
        self.hop_length = self.model.hop_length

    def get_snac_bitrates(self):
        support_bitrates = [
            self.sample_rate
            / self.hop_length
            / max(self.model.vq_strides)
            * sum(self.model.vq_strides)
            * math.log2(self.model.codebook_size)
        ]
        return support_bitrates

    @torch.inference_mode()
    def resyn(self, audio: torch.Tensor) -> torch.Tensor:
        length = audio.shape[-1]
        codes = self.model.encode(audio.unsqueeze(0))
        resyn_audio = self.model.decode(codes)
        return resyn_audio.squeeze(0)[:, :length]

    @torch.inference_mode()
    def extract_unit(
        self, audio: torch.Tensor
    ) -> tp.Tuple[tp.List[torch.Tensor], tp.Tuple[tp.List[torch.Tensor], int]]:
        length = audio.shape[-1]
        codes = self.model.encode(audio.unsqueeze(0))
        return codes, (codes, length)

    @torch.inference_mode()
    def synth_unit(
        self, stuff_for_synth: tp.Tuple[tp.List[torch.Tensor], int]
    ) -> torch.Tensor:
        codes, length = stuff_for_synth
        synth_audio = self.model.decode(codes)
        return synth_audio.squeeze(0)[:, :length]
