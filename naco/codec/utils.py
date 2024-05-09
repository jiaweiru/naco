import torch
import math


def pad_audio(audio: torch.Tensor, hop_length: int):
    length = audio.shape[-1]
    right_pad = math.ceil(length / hop_length) * hop_length - length
    audio = torch.nn.functional.pad(audio, (0, right_pad))
    return audio
