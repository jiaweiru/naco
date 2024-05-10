import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils import weight_norm

# from librosa.util import normalize

import json
import math
import typing as tp
from pathlib import Path
from urllib.request import urlretrieve

from .utils import pad_audio

HIFICODEC_MODEL_URL = {
    "HiFi-Codec-16k-320d-large-universal": {
        "ckpt": "https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-16k-320d-large-universal",
        "config": "https://raw.githubusercontent.com/yangdongchao/AcademiCodec/master/egs/HiFi-Codec-16k-320d/config_16k_320d.json",
    },
    "HiFi-Codec-16k-320d": {
        "ckpt": "https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-16k-320d",
        "config": "https://raw.githubusercontent.com/yangdongchao/AcademiCodec/master/egs/HiFi-Codec-16k-320d/config_16k_320d.json",
    },
    "HiFi-Codec-24k-240d": {
        "ckpt": "https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-24k-240d",
        "config": "https://raw.githubusercontent.com/yangdongchao/AcademiCodec/master/egs/HiFi-Codec-24k-240d/config_24k_240d.json",
    },
    "HiFi-Codec-24k-320d": {
        "ckpt": "https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-24k-320d",
        "config": "https://raw.githubusercontent.com/yangdongchao/AcademiCodec/master/egs/HiFi-Codec-24k-320d/config_24k_320d.json",
    },
}


class HiFiCodec:
    """
    HiFiCodec. https://github.com/yangdongchao/AcademiCodec/tree/master
    HiFiCodec does not train the model on multiple bitrates, only HiFiCodec pre-trained models are
    supported here, not the Encodec model reproduced in the official library
    """

    def __init__(self, model_type: str, device: str = "cpu") -> None:
        model_path = Path.home().joinpath(f".cache/hificodec/{model_type}")
        config_path = model_path.joinpath(
            HIFICODEC_MODEL_URL[model_type]["config"].split("/")[-1]
        )
        ckpt_path = model_path.joinpath(
            HIFICODEC_MODEL_URL[model_type]["ckpt"].split("/")[-1]
        )

        if not (config_path.exists() and ckpt_path.exists()):
            model_path.mkdir(parents=True, exist_ok=True)
            config_url = HIFICODEC_MODEL_URL[model_type]["config"]
            ckpt_url = HIFICODEC_MODEL_URL[model_type]["ckpt"]
            urlretrieve(config_url, config_path)
            urlretrieve(ckpt_url, ckpt_path)

        self.model = VQVAE(config_path, ckpt_path, with_encoder=True, device=device)
        self.model.generator.remove_weight_norm()
        self.model.encoder.remove_weight_norm()
        self.model = self.model.to(device).eval()

        self.sample_rate = self.model.h.sampling_rate
        self.support_bitrates = self.get_audiodec_bitrates()

    def get_audiodec_bitrates(self) -> tp.List[float]:
        n_codes = self.model.h.n_codes
        n_codebooks = self.model.h.n_code_groups * 2
        hop_length = math.prod(self.model.h.upsample_rates)
        bitrate = self.sample_rate / hop_length * n_codebooks * math.log2(n_codes)
        support_bitrates = [round(bitrate / 1_000, 1)]
        return support_bitrates

    @torch.inference_mode()
    def resyn(
        self,
        audio: torch.Tensor,
        # norm_as_hificodec: bool = False
    ) -> torch.Tensor:

        length = audio.shape[-1]
        hop_length = math.prod(self.model.h.upsample_rates)
        audio = pad_audio(audio, hop_length)

        # # https://github.com/yangdongchao/AcademiCodec/blob/master/egs/HiFi-Codec-24k-320d/infer.ipynb
        # if norm_as_hificodec:
        #     audio = torch.tensor(
        #         normalize(audio.cpu().numpy(), axis=1),
        #         dtype=torch.float32,
        #         device=audio.device,
        #     )
        code = self.model.encode(audio)
        resyn_audio = self.model(code)
        return resyn_audio.squeeze(0)[:, :length]

    @torch.inference_mode()
    def extract_unit(
        self,
        audio: torch.Tensor,
        # norm_as_hificodec: bool = False
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, int]]:

        length = audio.shape[-1]
        hop_length = math.prod(self.model.h.upsample_rates)
        audio = pad_audio(audio, hop_length)

        # if norm_as_hificodec:
        #     audio = torch.tensor(
        #         normalize(audio.cpu().numpy(), axis=1),
        #         dtype=torch.float32,
        #         device=audio.device,
        #     )
        code = self.model.encode(audio)
        return code.squeeze(0).permute(1, 0), (code, length)

    @torch.inference_mode()
    def synth_unit(self, stuff_for_synth: tp.Tuple[torch.Tensor, int]) -> torch.Tensor:
        code, length = stuff_for_synth
        synth_audio = self.model(code)
        return synth_audio.squeeze(0)[:, :length]


# Academic Codec (HiFi Codec) Model

LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            nn.Conv1d(512, h.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        # padding=(u//2 + u%2),
                        padding=(k - u) // 2,
                        # output_padding=u%2
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        # print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Encoder(nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(1, 32, 7, 1, padding=3))
        self.normalize = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            list(reversed(list(zip(h.upsample_rates, h.upsample_kernel_sizes))))
        ):
            self.ups.append(
                weight_norm(
                    nn.Conv1d(
                        32 * (2**i),
                        32 * (2 ** (i + 1)),
                        k,
                        u,
                        padding=((k - u) // 2),
                        # padding=(u//2 + u%2)
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 32 * (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(h.resblock_kernel_sizes)),
                    list(reversed(h.resblock_dilation_sizes)),
                )
            ):
                self.resblocks.append(resblock(h, ch, k, d))
                self.normalize.append(nn.GroupNorm(ch // 16, ch, eps=1e-6, affine=True))
        self.conv_post = nn.Conv1d(512, 512, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        # print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)


class Quantizer_module(nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        # compute Euclidean distance
        d = (
            torch.sum(x**2, 1, keepdim=True)
            + torch.sum(self.embedding.weight**2, 1)
            - 2 * torch.matmul(x, self.embedding.weight.T)
        )
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        return z_q, min_indicies


class Quantizer(nn.Module):
    def __init__(self, h):
        super(Quantizer, self).__init__()
        assert 512 % h.n_code_groups == 0
        self.quantizer_modules = nn.ModuleList(
            [
                Quantizer_module(h.n_codes, 512 // h.n_code_groups)
                for _ in range(h.n_code_groups)
            ]
        )
        self.quantizer_modules2 = nn.ModuleList(
            [
                Quantizer_module(h.n_codes, 512 // h.n_code_groups)
                for _ in range(h.n_code_groups)
            ]
        )
        self.h = h
        self.codebook_loss_lambda = self.h.codebook_loss_lambda  # e.g., 1
        self.commitment_loss_lambda = self.h.commitment_loss_lambda  # e.g., 0.25
        self.residul_layer = 2
        self.n_code_groups = h.n_code_groups

    def for_one_step(self, xin, idx):
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, 512)
        x = torch.split(x, 512 // self.h.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        if idx == 0:
            for _x, m in zip(x, self.quantizer_modules):
                _z_q, _min_indicies = m(_x)
                z_q.append(_z_q)
                min_indicies.append(_min_indicies)  # B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean(
                (z_q - xin.detach()) ** 2
            ) + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies
        else:
            for _x, m in zip(x, self.quantizer_modules2):
                _z_q, _min_indicies = m(_x)
                z_q.append(_z_q)
                min_indicies.append(_min_indicies)  # B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean(
                (z_q - xin.detach()) ** 2
            ) + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies

    def forward(self, xin):
        # B, C, T
        quantized_out = 0.0
        residual = xin
        all_losses = []
        all_indices = []
        for i in range(self.residul_layer):
            quantized, loss, indices = self.for_one_step(residual, i)  #
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.extend(indices)  #
            all_losses.append(loss)
        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)
        return quantized_out, loss, all_indices

    def embed(self, x):
        # idx: N, T, 4
        # print('x ', x.shape)
        quantized_out = torch.tensor(0.0, device=x.device)
        x = torch.split(x, 1, 2)
        for i in range(self.residul_layer):
            ret = []
            if i == 0:
                for j in range(self.n_code_groups):
                    q = x[j]
                    embed = self.quantizer_modules[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                # print(ret.shape)
                quantized_out = quantized_out + ret
            else:
                for j in range(self.n_code_groups):
                    q = x[j + self.n_code_groups]
                    embed = self.quantizer_modules2[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                quantized_out = quantized_out + ret
        return quantized_out.transpose(1, 2)  # N, C, T


class VQVAE(nn.Module):
    def __init__(self, config_path, ckpt_path, with_encoder=False, device="cpu"):
        super(VQVAE, self).__init__()
        ckpt = torch.load(ckpt_path, map_location=device)
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        self.generator.load_state_dict(ckpt["generator"])
        self.quantizer.load_state_dict(ckpt["quantizer"])
        if with_encoder:
            self.encoder = Encoder(self.h)
            self.encoder.load_state_dict(ckpt["encoder"])

    def forward(self, x):
        # x is the codebook
        # x.shape (B, T, Nq)
        quant_emb = self.quantizer.embed(x)
        return self.generator(quant_emb)

    def encode(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        c = self.encoder(x.unsqueeze(1))
        q, loss_q, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        # shape: [N, T, 4]
        return torch.stack(c, -1)
