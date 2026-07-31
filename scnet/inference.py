from pdb import main

import yaml
import torch
from SCNet import SCNet
import tqdm
from utils import new_sdr
import os
import torchaudio
import argparse
from ml_collections import ConfigDict
from pathlib import Path
import numpy as np
import time
import random
import typing as tp
import soundfile as sf

def set_seed(seed=42):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # OS environment (some libraries use this)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # CuDNN determinism
    # Warning: This can slow down performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = torch.nn.functional.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)

    
class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return

def apply_model(model, mix, shifts=1, split=True, segment=20, samplerate=44100,
                overlap=0.25, transition_power=1., progress=False, device=None,
                num_workers=0, pool=None):
    """
    Apply model to a given mixture.

    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress (bool): if True, show a progress bar (requires split=True)
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if pool is None:
        if num_workers > 0 and device.type == 'cpu':
            pool = ThreadPoolExecutor(num_workers)
        else:
            pool = DummyPoolExecutor()
    kwargs = {
        'shifts': shifts,
        'split': split,
        'overlap': overlap,
        'transition_power': transition_power,
        'progress': progress,
        'device': device,
        'pool': pool,
    }
    model.to(device)

    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape
    if split:
        kwargs['split'] = False
        out = torch.zeros(batch, len(model.sources), channels, length, device=mix.device)
        sum_weight = torch.zeros(length, device=mix.device)
        segment = int(samplerate * segment)
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        scale = stride / samplerate
        weight = torch.cat([torch.arange(1, segment // 2 + 1, device=device),
                         torch.arange(segment - segment // 2, 0, -1, device=device)])              
        assert len(weight) == segment
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max())**transition_power
        futures = []
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            future = pool.submit(apply_model, model, chunk, **kwargs)
            futures.append((future, offset))
            offset += segment
        if progress:
            futures = tqdm.tqdm(futures, unit_scale=scale, ncols=120, unit='seconds')
        for future, offset in futures:
            chunk_out = future.result()
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += (weight[:chunk_length] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment] += weight[:chunk_length].to(mix.device)
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    elif shifts:
        kwargs['shifts'] = 0
        max_shift = int(0.5 * samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(model, shifted, **kwargs)
            out += shifted_out[..., max_shift - offset:]
        out /= shifts
        return out
    else:
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length).to(device)
        with torch.no_grad():
            out = model(padded_mix)
        return center_trim(out, length)

def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    if wav.ndim == 1:
        src_channels = 1
    else:
        src_channels = wav.shape[-2]

    if src_channels == channels:
        pass
    elif channels == 1:
        if src_channels > 1:
            wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        wav = wav.expand(-1, channels, -1)
    elif src_channels >= channels:
        wav = wav[..., :channels, :]
    else:
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

def convert_audio(wav, from_samplerate, to_samplerate, channels):
    """Convert audio from a given samplerate to a target one and target number of channels."""
    # Convert channels first
    wav = convert_audio_channels(wav, channels)
    
    # Resample audio if necessary
    if from_samplerate != to_samplerate:
        wav = julius.resample_frac(wav, from_samplerate, to_samplerate)
    return wav

def load_model(model, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No model checkpoint file found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        if 'best_state' not in checkpoint:
            #raise KeyError(f"Checkpoint does not contain the state")
            state_dict = dict(checkpoint)
        else:
            state_dict = checkpoint['best_state']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
               new_state_dict[k[7:]] = v
            else:
               new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        return model
    


class Seperator:
    def __init__(self, model, checkpoint_path):
        self.separator = load_model(model, checkpoint_path)
        self.separator.to(DEVICE)

    @property
    def instruments(self):
        if args.tnoisy:
            return ['bass', 'drums', 'other', 'vocals', 'audience']
        else:
            return ['bass', 'drums', 'other', 'vocals']


    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate
        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """
        mix = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))

        # convert audio to GPU
        mix = mix.to(DEVICE)
        mix_channels = mix.shape[0]
        mix = convert_audio(mix, sample_rate, 44100, self.separator.audio_channels)

        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std
        # Separate
        with torch.no_grad():
            estimates = apply_model(self.separator, mix[None], overlap=0.5, progress=False)[0]

        # Printing some sanity checks.
        #print(time.time() - b, mono.shape[-1] / sample_rate, mix.std(), estimates.std())

        estimates = estimates * std + mean

        estimates = convert_audio(estimates, 44100, sample_rate, mix_channels)

        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            idx = self.separator.sources.index(instrument)
            separated_music_arrays[instrument] = torch.squeeze(estimates[idx]).detach().cpu().numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates



if __name__ == "__main__":
    global args, DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--reverberant", dest='reverberant', action='store_true')
    parser.add_argument("--noisy", dest='noisy', action='store_true')
    parser.add_argument("--mix_path", type=str, default='test.wav', help="path to mixture .wav file")
    parser.add_argument("--out_path", type=str, default='..', help="directory where to write the stems")

    args = parser.parse_args()

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    set_seed(42)

    if not args.tnoisy and not args.treverberant:
        print("Inference on clean version of the model")
        model_name = 'M1_clean'
        with open('conf/ismir26_clean.yaml', 'r') as file:
            config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    elif not args.tnoisy and args.treverberant:
        print("Evaluating rev version of the model")
        model_name = 'M2_reverberant'
        with open('conf/ismir26_rev.yaml', 'r') as file:
            config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    elif args.tnoisy and not args.treverberant:
        print("Evaluating noisy version of the model")
        model_name = 'M3_noisy'
        with open('conf/ismir26_noisy.yaml', 'r') as file:
            config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    else:
        print("Evaluating noisyrev version of the model")
        model_name = 'M4_noisyrev'
        with open('conf/ismir26_noisyrev.yaml', 'r') as file:
            config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

    model_path = os.path.join('models', model_name)
    model = SCNet(**config.model)
    model.eval()

    seperator = Seperator(model, os.path.join(model_path, 'model.th'))

    mix = torchaudio.load(args.mix_path)

    with torch.no_grad():
        estimates, _ = seperator.separate_music_file(mix[0].T.numpy(), 44100)

    for estimate in estimates:
        sf.write(os.path.join(args.out_path, estimate + '.wav'), estimates[estimate], mix[1])
    print("Done! Stems written to {}".format(args.out_path))
