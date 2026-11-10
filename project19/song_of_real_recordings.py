"""Song of real recordings: the actual LIGO detector data for every collision with a small data file.

Run from the repo root:   python3 project19/song_of_real_recordings.py
It makes two files:
    plots/19_real_recordings_song.wav       the song: 3 seconds of real detector data per collision
    plots/19_real_recordings_gallery.png    one spectrogram per collision, in the same order as the song

HOW IT WORKS
* Every collision gets exactly the same treatment as GW150914 in starters/19_hear_a_merger.py:
  LIGO Livingston data (Hanford if Livingston wasn't recording), whitened (the noise evened out
  across pitches), only 30-400 Hz kept, 3 seconds from 2 s before the merger to 1 s after,
  scaled so its loudest moment is 1. Same WAV settings too: 4096 samples per second, 32-bit.
* The clips play one after another in the order they were detected, so the merger always comes
  2 seconds into each 3-second clip: listen for a "whoop" on that beat.
* This is real data, NOT a model. Most collisions are too faint to stand out in a single detector,
  so many clips are just hiss; the loud ones (like GW150914 and GW170817) are clearly audible.
* Only collisions with a small (32-second, ~1 MB) data file are included: everything from
  2015-2020 plus one from 2023. Newer collisions only have ~100 MB files, too big to download here.
* Pitches are real (not raised), so headphones help: laptop speakers miss most of 30-400 Hz.

DATA (please cite): LIGO Scientific Collaboration, Virgo Collaboration and KAGRA Collaboration,
    via the Gravitational Wave Open Science Center (GWOSC), https://gwosc.org
    event list: https://gwosc.org/eventapi/jsonfull/allevents/ and /GWTC/
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from gwpy.timeseries import TimeSeries
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo folder
DATA = os.path.join(ROOT, 'data')
CLIPS = os.path.join(DATA, 'gwosc_32s')      # the downloaded detector files go here
PLOTS = os.path.join(ROOT, 'plots')

# Same certificate note as starters/19_hear_a_merger.py: gwosc.org's certificate is signed by
# "emSign Root CA - G1", which the operating system's list includes but certifi's doesn't.
# This keeps the security check on; it only chooses which list of authorities to trust.
SYSTEM_CERTS = '/etc/ssl/certs/ca-certificates.crt'
VERIFY = SYSTEM_CERTS if os.path.exists(SYSTEM_CERTS) else True


def download(url, path):
    """Download a file once and keep it, so reruns don't download it again."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url, verify=VERIFY, timeout=120)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
    return path


def get_json(url, filename):
    with open(download(url, os.path.join(DATA, filename))) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------
# 1. Which collisions have a small data file, and where is it?
# ---------------------------------------------------------------------------------------------
# The confident-detections catalog: one entry per collision, with its time and loudness.
catalog = pd.DataFrame(get_json('https://gwosc.org/eventapi/jsonfull/GWTC/',
                                'gwosc_gwtc_catalog.json')['events'].values())
catalog['GPS'] = pd.to_numeric(catalog.GPS)

# Every version of every event: older versions are often the ones with the small 32-second files.
all_versions = get_json('https://gwosc.org/eventapi/jsonfull/allevents/', 'gwosc_allevents.json')['events']

DETECTORS = ['L1', 'H1']   # Livingston first (like the starter), then Hanford
small_file = {}            # event name -> (detector, url)
for entry in sorted(all_versions.values(), key=lambda e: e['version']):   # newest version wins
    for det in reversed(DETECTORS):                                       # so L1 wins over H1
        for s in entry['strain']:
            if (s['detector'] == det and s['duration'] == 32 and s['sampling_rate'] == 4096
                    and s['format'] == 'hdf5'):
                small_file[entry['commonName']] = (det, s['url'])

events = catalog[catalog.commonName.isin(small_file)].sort_values('GPS').reset_index(drop=True)
print(f'{len(catalog)} collisions in the catalog; {len(events)} have a small data file -> one clip each')

# ---------------------------------------------------------------------------------------------
# 2. Process each one exactly like the starter does for GW150914
# ---------------------------------------------------------------------------------------------
RATE = 4096          # samples per second, same as the data and the GW150914 WAV
CLIP_START, CLIP_END = -2, 1       # seconds around the merger, same as the starter
FADE = int(0.02 * RATE)            # 20 ms fade at each end so clips join without a click

clips, spectrograms, used, problems = [], [], [], []
for i, ev in events.iterrows():
    name, gps = ev.commonName, ev.GPS
    det, url = small_file[name]
    try:
        path = download(url, os.path.join(CLIPS, os.path.basename(url)))
        data = TimeSeries.read(path, format='hdf5.gwosc')
        sound = data.whiten(4, 2).bandpass(30, 400).crop(gps + CLIP_START, gps + CLIP_END)
        clip = sound.value / np.abs(sound.value).max()           # loudest moment = 1, like gwpy does
        if not np.all(np.isfinite(clip)):
            raise ValueError('data contains gaps (NaN)')
        q = data.q_transform(outseg=(gps - 0.5, gps + 0.2), frange=(20, 500))   # starter's picture
    except Exception as err:   # say what failed instead of hiding it
        problems.append(f'{name}: {err}')
        continue
    ramp = np.linspace(0, 1, FADE)
    clip[:FADE] *= ramp
    clip[-FADE:] *= ramp[::-1]
    clips.append(clip.astype('float32'))
    spectrograms.append((q.times.value - gps, q.frequencies.value, q.value.T))
    used.append((name, det))
    print(f'{len(used):3d}  {name:18} {det}', flush=True)

if problems:
    print('\nLeft out because of a problem:')
    print('\n'.join(problems))

song = np.concatenate(clips)
os.makedirs(PLOTS, exist_ok=True)
wav_path = os.path.join(PLOTS, '19_real_recordings_song.wav')
wavfile.write(wav_path, RATE, song)
clip_len = (CLIP_END - CLIP_START)
print(f'\nSaved {wav_path}: {len(used)} clips, {len(song) / RATE / 60:.1f} minutes')

# ---------------------------------------------------------------------------------------------
# 3. The gallery: one spectrogram per clip, in song order, labelled with when it plays
# ---------------------------------------------------------------------------------------------
cols = 13
rows = int(np.ceil(len(used) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.1 * rows + 1.2), sharex=True, sharey=True)
axes = axes.ravel()
for k, ((name, det), (t, f, power)) in enumerate(zip(used, spectrograms)):
    ax = axes[k]
    mesh = ax.pcolormesh(t, f, power, cmap='viridis', vmin=0, vmax=25, shading='auto')
    starts = k * clip_len                           # when this clip starts in the song (seconds)
    ax.set_title(f"{starts // 60}:{starts % 60:02d}  {name.split('_')[0]}", fontsize=9)
    ax.set_yscale('log')
for ax in axes[len(used):]:
    ax.axis('off')                                  # hide empty panels in the last row
axes[0].set_ylim(20, 500)
axes[0].set_yticks([30, 100, 300], labels=['30', '100', '300'])
axes[0].set_xticks([-0.4, 0, 0.2], labels=['-0.4', '0', '0.2'])
for ax in axes[::cols]:
    ax.set_ylabel('Hz', fontsize=9)
for ax in axes[(rows - 1) * cols:]:
    ax.set_xlabel('s from merger', fontsize=9)

fig.suptitle(f'Song of real recordings: {len(used)} collisions in LIGO data, 2015-2023 '
             f'(the order and start times of 19_real_recordings_song.wav)', fontsize=16, x=0.01, ha='left')
fig.text(0.01, 1 - 0.75 / fig.get_figheight(),
         'Each panel: real LIGO data around one merger, processed like the GW150914 plot '
         '(Livingston detector, or Hanford when Livingston was off). A bright curve rising to the '
         'right is a chirp; plain speckle means the collision is too faint to see in one detector.\n'
         'Colour scale fixed at 0-25 for all panels so they can be compared. '
         'Data: LIGO/Virgo/KAGRA via the Gravitational Wave Open Science Center (gwosc.org).',
         fontsize=11, va='top')
fig.colorbar(mesh, ax=axes.tolist(), label='Normalized energy', fraction=0.015, pad=0.01)
fig.subplots_adjust(top=1 - 1.4 / fig.get_figheight(), left=0.03, right=0.94, bottom=0.04,
                    hspace=0.45, wspace=0.08)
png_path = os.path.join(PLOTS, '19_real_recordings_gallery.png')
fig.savefig(png_path, dpi=100)
print(f'Saved {png_path}')
