import argparse
import os
import shutil
from pathlib import Path

import tqdm
from moisesdb.dataset import MoisesDB
import moisesdb.defaults
from moisesdb.utils import save_audio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build MUSDBMOISES from MUSDB18HQ, MoisesDB and Crowdioset."
    )
    parser.add_argument("--moises_path", type=Path, default=Path("../moisesdb"))
    parser.add_argument("--musdb_path", type=Path, default=Path("../musdb18hq"))
    parser.add_argument("--out_path", type=Path, default=Path("../musdbmoises"))
    parser.add_argument("--crowdioset_path", type=Path, default=Path("../crowdioset"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    moises_path = args.moises_path
    musdb_path = args.musdb_path
    out_path = args.out_path
    crowdioset_path = args.crowdioset_path

    # we make a copy of MUSDB18HQ where to add moisesdb and singalongs
    if not os.path.exists(os.path.join(out_path)):
        shutil.copytree(musdb_path, out_path)

    if not os.path.exists(os.path.join(out_path, "valid")):

        os.makedirs(os.path.join(out_path, "valid"))

        valid_names = ['Actions - One Minute Smile', 'Alexander Ross - Goodbye Bolero', 'ANiMAL - Rockshow',
                 'Clara Berry And Wooldog - Waltz For My Victims', 'Fergessen - Nos Palpitants',
                 'James May - On The Line', 'Johnny Lokke - Promises & Lies', 'Leaf - Summerghost',
                 'Meaxic - Take A Step', 'Patrick Talbot - A Reason To Leave', 'Skelpolu - Human Mistakes',
                 'Traffic Experiment - Sirens', 'Triviul - Angelsaint', 'Young Griffo - Pennies']

        for name in valid_names:
            shutil.move(os.path.join(out_path, 'train', name), os.path.join(out_path, 'valid', name))

    # if not all MOISESDB tracks are in MUSDBMOISES train folder already
    if len([x for x in os.listdir(os.path.join(out_path, 'train')) if ' - ' in x]) != 320:
        # add the MOISESDB 4-sources and mixtures to the train dir
        db = MoisesDB(
            data_path=moises_path,
            sample_rate=44100
        )
        for i, song in enumerate(tqdm.tqdm(db)):
            try:
                folder = os.path.join(out_path, 'train', song.artist + ' - ' + song.name)
                if not os.path.exists(folder):
                    stems = song.mix_stems(moisesdb.defaults.mix_4_stems)
                    mixture = song.audio
                    minlenght = min(stems['vocals'].shape[1], stems['drums'].shape[1], stems['bass'].shape[1], stems['other'].shape[1], mixture.shape[1])
                    os.mkdir(folder)
                    save_audio(os.path.join(folder, 'mixture.wav'), mixture[:, :minlenght])
                    save_audio(os.path.join(folder, 'vocals.wav'), stems['vocals'][:, :minlenght])
                    save_audio(os.path.join(folder, 'drums.wav'), stems['drums'][:, :minlenght])
                    save_audio(os.path.join(folder, 'bass.wav'), stems['bass'][:, :minlenght])
                    save_audio(os.path.join(folder, 'other.wav'), stems['other'][:, :minlenght])
            except:
                print('some error')
                print(i)

    # train: only singalong.wav exists per-song, under crowdioset/train/singalongs/<song>/
    for song_dir in (out_path / 'train').iterdir():
        if song_dir.is_dir():
            src = crowdioset_path / 'train' / 'singalongs' / song_dir.name / 'singalong.wav'
            shutil.copy(src, song_dir / 'singalong.wav')

    # valid: both audience.wav and singalong.wav exist under crowdioset/valid/<song>/
    for song_dir in (out_path / 'valid').iterdir():
        if song_dir.is_dir():
            src_dir = crowdioset_path / 'valid' / song_dir.name
            for fname in ('audience.wav', 'singalong.wav'):
                shutil.copy(src_dir / fname, song_dir / fname)

    # test: both audience.wav and singalong.wav exist under crowdioset/test/<song>/
    for song_dir in (out_path / 'test').iterdir():
        if song_dir.is_dir():
            src_dir = crowdioset_path / 'test' / song_dir.name
            for fname in ('audience.wav', 'singalong.wav'):
                shutil.copy(src_dir / fname, song_dir / fname)


if __name__ == "__main__":
    main()
