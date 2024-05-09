import torch
import torchaudio
from pathlib import Path
from naco.codec import dac, encodec, funcodec, hificodec, audiodec, snac


def test_dac(mode="norm_chunked", device="cpu"):

    model_types = ["dac_16k_8kbps", "dac_24k_8kbps", "dac_44k_8kbps"]

    for model_type in model_types:
        codec = dac.DAC(model_type, device)
        waveform, sample_rate = torchaudio.load(f"./demo/0000_{model_type[4:6]}.wav")
        waveform = waveform.to(device)
        if mode == "plain":
            resyn_waveform1 = codec.resyn(waveform)
            code, stuff_for_synth = codec.extract_unit(waveform)
            resyn_waveform2 = codec.synth_unit(stuff_for_synth)
        elif mode == "norm_chunked":
            resyn_waveform1 = codec.resyn_norm_chunked(waveform)
            code, stuff_for_synth = codec.extract_unit_norm_chunked(waveform)
            resyn_waveform2 = codec.synth_unit_norm_chunked(stuff_for_synth)

        print(f"DAC 8kbps {mode} inference")
        print(
            f"input shape: {waveform.shape}, output shape: {resyn_waveform1.shape}, {resyn_waveform2.shape}"
        )
        print(
            f"code shape and dtype: {code.shape}, {code.dtype},{torch.min(code)},{torch.max(code)}"
        )
        torchaudio.save(
            f"./demo/0000_{model_type[4:6]}_{model_type}_{mode}.wav",
            resyn_waveform1.cpu(),
            sample_rate,
            bits_per_sample=16,
        )


def test_encodec(device="cpu"):

    model_type = "encodec_24k_24kbps"
    codec = encodec.EnCodec(model_type, device)
    waveform, sample_rate = torchaudio.load("./demo/0000_24.wav")
    waveform = waveform.to(device)
    for br in codec.support_bitrates:
        resyn_waveform1 = codec.resyn(waveform, br)
        code, stuff_for_synth = codec.extract_unit(waveform, br)
        resyn_waveform2 = codec.synth_unit(stuff_for_synth)

        print("EnCodec" + str(br) + "kbps")
        print(
            f"input shape: {waveform.shape}, output shape: {resyn_waveform1.shape}, {resyn_waveform2.shape}"
        )
        print(
            f"code shape and dtype: {code.shape}, {code.dtype},{torch.min(code)},{torch.max(code)}"
        )
        br = str(br).replace(".", "_")
        torchaudio.save(
            f"./demo/0000_24_encodec_{br}kbps.wav",
            resyn_waveform1.cpu(),
            sample_rate,
            bits_per_sample=16,
        )

    model_type = "encodec_48k_24kbps"
    codec = encodec.EnCodec(model_type, device)
    waveform, sample_rate = torchaudio.load("./demo/0001_stereo_48.wav")
    waveform = waveform.to(device)
    for br in codec.support_bitrates:
        resyn_waveform1 = codec.resyn(waveform, br)
        code, stuff_for_synth = codec.extract_unit(waveform, br)
        resyn_waveform2 = codec.synth_unit(stuff_for_synth)

        print("EnCodec" + str(br) + "kbps")
        print(
            f"input shape: {waveform.shape}, output shape: {resyn_waveform1.shape}, {resyn_waveform2.shape}"
        )
        print(
            f"code shape and dtype: {code.shape}, {code.dtype},{torch.min(code)},{torch.max(code)}"
        )
        torchaudio.save(
            f"./demo/0001_stereo_48_encodec_{br}kbps.wav",
            resyn_waveform1.cpu(),
            sample_rate,
            bits_per_sample=16,
        )


def test_funcodec(device="cpu"):
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
        for br in codec.support_bitrates:
            waveform, sample_rate = torchaudio.load("./demo/0000_16.wav")
            waveform = waveform.to(device)
            resyn_waveform1 = codec.resyn(waveform, br)
            code, stuff_for_synth = codec.extract_unit(waveform, br)
            resyn_waveform2 = codec.synth_unit(stuff_for_synth)

            print(f"FunCodec {str(br)} kbps")
            print(
                f"input shape: {waveform.shape}, output shape: {resyn_waveform1.shape}, {resyn_waveform2.shape}"
            )
            print(
                f"code shape and dtype: {code.shape}, {code.dtype},{torch.min(code)},{torch.max(code)}"
            )
            torchaudio.save(
                f"./demo/0000_16_{model_type}_{br}kbps.wav",
                resyn_waveform1.cpu(),
                sample_rate,
                bits_per_sample=16,
            )


def test_hificodec(device="cpu"):
    model_types = [
        "HiFi-Codec-16k-320d-large-universal",  # 2kbps
        "HiFi-Codec-16k-320d",  # 2kbps
        "HiFi-Codec-24k-240d",  # 2kbps
        "HiFi-Codec-24k-320d",  # 3kbps
    ]

    for model_type in model_types:
        codec = hificodec.HiFiCodec(model_type, device)
        if model_type in ["HiFi-Codec-16k-320d-large-universal", "HiFi-Codec-16k-320d"]:
            waveform, sample_rate = torchaudio.load("./demo/0000_16.wav")
            waveform = waveform.to(device)
        elif model_type in ["HiFi-Codec-24k-240d", "HiFi-Codec-24k-320d"]:
            waveform, sample_rate = torchaudio.load("./demo/0000_24.wav")
            waveform = waveform.to(device)
        resyn_waveform1 = codec.resyn(waveform)
        code, stuff_for_synth = codec.extract_unit(waveform)
        resyn_waveform2 = codec.synth_unit(stuff_for_synth)

        print(f"HiFiCodec, {model_type}")
        print(
            f"input shape: {waveform.shape}, output shape: {resyn_waveform1.shape}, {resyn_waveform2.shape}"
        )
        print(
            f"code shape and dtype: {code.shape}, {code.dtype},{torch.min(code)},{torch.max(code)}"
        )
        if model_type in ["HiFi-Codec-16k-320d-large-universal", "HiFi-Codec-16k-320d"]:
            torchaudio.save(
                (
                    f"./demo/0000_16_{model_type}_2kbps.wav"
                    if model_type != "HiFi-Codec-24k-320d"
                    else f"./demo/0000_16_{model_type}_3kbps.wav"
                ),
                resyn_waveform1.cpu(),
                sample_rate,
                bits_per_sample=16,
            )
        elif model_type in ["HiFi-Codec-24k-240d", "HiFi-Codec-24k-320d"]:
            torchaudio.save(
                (
                    f"./demo/0000_24_{model_type}_2kbps.wav"
                    if model_type != "HiFi-Codec-24k-320d"
                    else f"./demo/0000_24_{model_type}_3kbps.wav"
                ),
                resyn_waveform1.cpu(),
                sample_rate,
                bits_per_sample=16,
            )


def test_audiodec(device="cpu"):
    model_types = [
        "AudioDec_v1_symAD_vctk_48000_hop300_clean",
        "AudioDec_v1_symAD_libritts_24000_hop300_clean",
    ]

    for model_type in model_types:
        codec = audiodec.AudioDEC(model_type, device)
        if "48000" in model_type:
            waveform, sample_rate = torchaudio.load("./demo/0002_48.wav")
            waveform = waveform.to(device)
        elif "24000" in model_type:
            waveform, sample_rate = torchaudio.load("./demo/0000_24.wav")
            waveform = waveform.to(device)
        resyn_waveform1 = codec.resyn(waveform)
        code, stuff_for_synth = codec.extract_unit(waveform)
        resyn_waveform2 = codec.synth_unit(stuff_for_synth)

        print(f"AudioDEC, {model_type}")
        print(
            f"input shape: {waveform.shape}, output shape: {resyn_waveform1.shape}, {resyn_waveform2.shape}"
        )
        print(
            f"code shape and dtype: {code.shape}, {code.dtype},{torch.min(code)},{torch.max(code)}"
        )

        torchaudio.save(
            (
                f"./demo/0002_48_{model_type}_12.8kbps.wav"
                if "48000" in model_type
                else f"./demo/0000_24_{model_type}_6.4kbps.wav"
            ),
            resyn_waveform2.cpu(),
            sample_rate,
            bits_per_sample=16,
        )


def test_snac(device="cpu"):
    model_types = ["snac_24khz_0.98kbps", "snac_32khz_1.9kbps", "snac_44khz_2.6kbps"]
    for model_type in model_types:
        codec = snac.SNAC(model_type, device)
        waveform, sample_rate = torchaudio.load(f"./demo/0000_{model_type[5:7]}.wav")
        waveform = waveform.to(device)
        resyn_waveform1 = codec.resyn(waveform)
        code, stuff_for_synth = codec.extract_unit(waveform)
        resyn_waveform2 = codec.synth_unit(stuff_for_synth)

        print(f"SNAC, {model_type}")
        print(
            f"input shape: {waveform.shape}, output shape: {resyn_waveform1.shape}, {resyn_waveform2.shape}"
        )
        torchaudio.save(
            f"./demo/0000_{model_type[5:7]}_{model_type}.wav",
            resyn_waveform1.cpu(),
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
            "0001_stereo_48.wav",
            "0002_48.wav",
        ],
    )
    # test_dac(mode="plain", device="cuda:2")
    # test_dac(mode="norm_chunked", device="cuda:2")
    # test_encodec(device="cuda:2")
    # test_funcodec(device="cuda:2")
    # test_hificodec(device="cuda:2")
    # test_audiodec(device="cuda:2")
    test_snac(device="cuda:2")
