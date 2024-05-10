import torch
import torchaudio
from pathlib import Path
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


def test_encodec(device):
    model_types = ["encodec_24k_24kbps", "encodec_48k_24kbps"]
    for model_type in model_types:
        codec = encodec.EnCodec(model_type, device)
        print(model_type)
        print(
            f"Samplerate: {codec.sample_rate}, support bitrates: {codec.support_bitrates}"
        )
        waveform, sample_rate = (
            torchaudio.load("./demo/0000_24.wav")
            if model_type == "encodec_24k_24kbps"
            else torchaudio.load("./demo/0000_stereo_48.wav")
        )
        waveform = waveform.to(device)
        assert sample_rate == codec.sample_rate
        for br in codec.support_bitrates:
            resyn_waveform = codec.resyn(waveform, br)
            code, stuff_for_synth = codec.extract_unit(waveform, br)
            syn_waveform = codec.synth_unit(stuff_for_synth)

            print(
                f"Resyn length: {waveform.shape[-1]} -> {resyn_waveform.shape[-1]} or {syn_waveform.shape[-1]}"
            )
            print(
                f"Code size and dtype: {code.shape}, {code.dtype}, index from {torch.min(code)} to {torch.max(code)}"
            )
            torchaudio.save(
                (
                    f"./demo/0000_24_{model_type}_{br}kbps.wav"
                    if model_type == "encodec_24k_24kbps"
                    else f"./demo/0000_stereo_48_{model_type}_{br}kbps.wav"
                ),
                resyn_waveform.cpu(),
                sample_rate,
                bits_per_sample=16,
            )


def test_funcodec(device):
    model_types = [
        "funcodec_en_libritts-16k-gr1nq32ds320",
        "funcodec_en_libritts-16k-gr8nq32ds320",
        "funcodec_en_libritts-16k-nq32ds320",
        "funcodec_en_libritts-16k-nq32ds640",
        "funcodec_zh_en_general_16k_nq32ds320",
        "funcodec_zh_en_general_16k_nq32ds640",
    ]

    for model_type in model_types:
        codec = funcodec.FunCodec(model_type, device)
        print(model_type)
        print(
            f"Samplerate: {codec.sample_rate}, support bitrates: {codec.support_bitrates}"
        )
        waveform, sample_rate = torchaudio.load("./demo/0000_16.wav")
        waveform = waveform.to(device)
        assert sample_rate == codec.sample_rate
        for br in codec.support_bitrates:
            resyn_waveform = codec.resyn(waveform, br)
            code, stuff_for_synth = codec.extract_unit(waveform, br)
            syn_waveform = codec.synth_unit(stuff_for_synth)

            print(
                f"Resyn length: {waveform.shape[-1]} -> {resyn_waveform.shape[-1]} or {syn_waveform.shape[-1]}"
            )
            print(
                f"Code size and dtype: {code.shape}, {code.dtype}, index from {torch.min(code)} to {torch.max(code)}"
            )
            torchaudio.save(
                f"./demo/0000_16_{model_type}_{br}kbps.wav",
                resyn_waveform.cpu(),
                sample_rate,
                bits_per_sample=16,
            )


def test_hificodec(device):
    model_types = [
        "HiFi-Codec-16k-320d-large-universal",
        "HiFi-Codec-16k-320d",
        "HiFi-Codec-24k-240d",
        "HiFi-Codec-24k-320d",
    ]

    for model_type in model_types:
        codec = hificodec.HiFiCodec(model_type, device)
        print(model_type)
        print(
            f"Samplerate: {codec.sample_rate}, support bitrates: {codec.support_bitrates}"
        )
        if model_type in ["HiFi-Codec-16k-320d-large-universal", "HiFi-Codec-16k-320d"]:
            waveform, sample_rate = torchaudio.load("./demo/0000_16.wav")
            waveform = waveform.to(device)
        elif model_type in ["HiFi-Codec-24k-240d", "HiFi-Codec-24k-320d"]:
            waveform, sample_rate = torchaudio.load("./demo/0000_24.wav")
            waveform = waveform.to(device)
        assert sample_rate == codec.sample_rate
        resyn_waveform = codec.resyn(waveform)
        code, stuff_for_synth = codec.extract_unit(waveform)
        syn_waveform = codec.synth_unit(stuff_for_synth)

        print(
            f"Resyn length: {waveform.shape[-1]} -> {resyn_waveform.shape[-1]} or {syn_waveform.shape[-1]}"
        )
        print(
            f"Code size and dtype: {code.shape}, {code.dtype}, index from {torch.min(code)} to {torch.max(code)}"
        )
        if model_type in ["HiFi-Codec-16k-320d-large-universal", "HiFi-Codec-16k-320d"]:
            torchaudio.save(
                f"./demo/0000_16_{model_type}_{codec.support_bitrates[0]}kbps.wav",
                resyn_waveform.cpu(),
                sample_rate,
                bits_per_sample=16,
            )
        elif model_type in ["HiFi-Codec-24k-240d", "HiFi-Codec-24k-320d"]:
            torchaudio.save(
                f"./demo/0000_24_{model_type}_{codec.support_bitrates[0]}kbps.wav",
                resyn_waveform.cpu(),
                sample_rate,
                bits_per_sample=16,
            )


def test_audiodec(device):
    model_types = [
        "AudioDec_v1_symAD_vctk_48000_hop300_clean",
        "AudioDec_v1_symAD_libritts_24000_hop300_clean",
    ]

    for model_type in model_types:
        codec = audiodec.AudioDEC(model_type, device)
        print(model_type)
        print(
            f"Samplerate: {codec.sample_rate}, support bitrates: {codec.support_bitrates}"
        )
        if "48000" in model_type:
            waveform, sample_rate = torchaudio.load("./demo/0000_48.wav")
            waveform = waveform.to(device)
        elif "24000" in model_type:
            waveform, sample_rate = torchaudio.load("./demo/0000_24.wav")
            waveform = waveform.to(device)
        assert sample_rate == codec.sample_rate
        resyn_waveform = codec.resyn(waveform)
        code, stuff_for_synth = codec.extract_unit(waveform)
        syn_waveform = codec.synth_unit(stuff_for_synth)

        print(
            f"Resyn length: {waveform.shape[-1]} -> {resyn_waveform.shape[-1]} or {syn_waveform.shape[-1]}"
        )
        print(
            f"Code size and dtype: {code.shape}, {code.dtype}, index from {torch.min(code)} to {torch.max(code)}"
        )

        torchaudio.save(
            (
                f"./demo/0000_48_{model_type}_{codec.support_bitrates[0]}kbps.wav"
                if "48000" in model_type
                else f"./demo/0000_24_{model_type}_{codec.support_bitrates[0]}kbps.wav"
            ),
            resyn_waveform.cpu(),
            sample_rate,
            bits_per_sample=16,
        )


def test_snac(device):
    model_types = ["snac_24khz_0.98kbps", "snac_32khz_1.9kbps", "snac_44khz_2.6kbps"]
    for model_type in model_types:
        codec = snac.SNAC(model_type, device)
        print(model_type)
        print(
            f"Samplerate: {codec.sample_rate}, support bitrates: {codec.support_bitrates}"
        )
        waveform, sample_rate = torchaudio.load(f"./demo/0000_{model_type[5:7]}.wav")
        waveform = waveform.to(device)
        resyn_waveform = codec.resyn(waveform)
        code, stuff_for_synth = codec.extract_unit(waveform)
        syn_waveform = codec.synth_unit(stuff_for_synth)

        print(
            f"Resyn length: {waveform.shape[-1]} -> {resyn_waveform.shape[-1]} or {syn_waveform.shape[-1]}"
        )
        # print(
        #     f"Code size and dtype: {code.shape}, {code.dtype}, index from {torch.min(code)} to {torch.max(code)}"
        # )

        torchaudio.save(
            f"./demo/0000_{model_type[5:7]}_{model_type}_{codec.support_bitrates[0]}kbps.wav",
            resyn_waveform.cpu(),
            sample_rate,
            bits_per_sample=16,
        )


def remove_files_except(folder_path, files_to_keep):
    folder_path_obj = Path(folder_path)

    if not folder_path_obj.is_dir():
        print(f"The directory {folder_path} does not exist.")
        return

    files_to_keep_objs = [folder_path_obj / file for file in files_to_keep]

    for file in folder_path_obj.iterdir():
        if file.is_file() and file not in files_to_keep_objs:
            try:
                file.unlink()
                print(f"Deleted file: {file}")
            except OSError as e:
                print(f"Error: {file} : {e.strerror}")


if __name__ == "__main__":
    remove_files_except(
        "demo",
        [
            "0000_16.wav",
            "0000_24.wav",
            "0000_32.wav",
            "0000_44.wav",
            "0000_stereo_48.wav",
            "0000_48.wav",
        ],
    )
    test_dac(mode="plain", device="cuda:2")
    test_dac(mode="norm_chunked", device="cuda:2")
    test_encodec(device="cuda:2")
    test_funcodec(device="cuda:2")
    test_hificodec(device="cuda:2")
    test_audiodec(device="cuda:2")
    test_snac(device="cuda:2")
