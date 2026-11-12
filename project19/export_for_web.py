"""Prepare the files for the interactive "Listening to Black Holes" web page (project19/web/).

Run from the repo root, AFTER the two song scripts:   python3 project19/export_for_web.py

It reads the sounds that already exist in plots/ and writes, into project19/web/:
    audio/gw150914.wav                the starter's 3-second GW150914 sound
    audio/real_recordings_song.wav    the song of 91 real recordings
        (same samples as plots/; converted to 16-bit at 8192 samples per second, because
         some web browsers can't play the original 32-bit, 4096-per-second files)
    gw150914_spectrogram.png          spectrogram of the GW150914 clip, no axes (the page draws them)
    gw150914_zoom.png                 the same, for the starter plot's 0.7 s window around the merger
    clip_spectrograms.jpg             one small spectrogram per clip of the real-recordings song
    data.json                         facts the page needs: event names, masses, times in each song

Nothing here is invented: every number comes from the GWOSC catalog or from the LIGO data files.
It also saves plots/19_gw150914_formula_vs_data.png: the textbook spiral-in formula drawn over
the real GW150914 spectrogram.

It also CHECKS that it has the clips in the right order, by rebuilding each 3-second clip from the
LIGO data and comparing it with the matching part of plots/19_real_recordings_song.wav.

DATA (please cite): LIGO Scientific Collaboration, Virgo Collaboration and KAGRA Collaboration,
via the Gravitational Wave Open Science Center (GWOSC), https://gwosc.org
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.cosmology import Planck15
from astropy.time import Time
from gwpy.timeseries import TimeSeries
from PIL import Image
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CLIPS = os.path.join(DATA, 'gwosc_32s')
PLOTS = os.path.join(ROOT, 'plots')
WEB = os.path.join(ROOT, 'project19', 'web')
os.makedirs(os.path.join(WEB, 'audio'), exist_ok=True)

# Same certificate note as starters/19_hear_a_merger.py (gwosc.org uses the emSign authority).
SYSTEM_CERTS = '/etc/ssl/certs/ca-certificates.crt'
VERIFY = SYSTEM_CERTS if os.path.exists(SYSTEM_CERTS) else True


def download(url, path):
    """Download a file once and keep it in data/, so reruns don't download it again."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url, verify=VERIFY, timeout=180)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
    return path


def get_json(url, filename):
    with open(download(url, os.path.join(DATA, filename))) as f:
        return json.load(f)


def number(x, digits=1):
    """Round a number for the web page; missing values become None (null in JSON)."""
    return None if x is None or pd.isna(x) else round(float(x), digits)


# The catalog of confident detections, as used by both song scripts
catalog = pd.DataFrame(get_json('https://gwosc.org/eventapi/jsonfull/GWTC/',
                                'gwosc_gwtc_catalog.json')['events'].values())
for col in ['GPS', 'mass_1_source', 'mass_2_source', 'chirp_mass_source', 'total_mass_source',
            'final_mass_source', 'redshift', 'network_matched_filter_snr']:
    catalog[col] = pd.to_numeric(catalog[col], errors='coerce')

NS_LIMIT = 3.0   # anything lighter than about 3 Suns is treated as a neutron star (as in the song)


def kind_of(m1, m2):
    if pd.isna(m1) or pd.isna(m2):
        return 'unknown'
    if m1 < NS_LIMIT:
        return 'two neutron stars'
    if m2 < NS_LIMIT:
        return 'black hole + neutron star'
    return 'two black holes'


def travel_time_gyr(z):
    """How long the waves travelled to reach us, in billions of years (from the redshift)."""
    return None if pd.isna(z) else round(float(Planck15.lookback_time(z).value), 2)


def date_of(gps):
    return Time(gps, format='gps').to_datetime().strftime('%Y-%m-%d %H:%M UTC')


def to_web_wav(samples, rate, path):
    """Save a sound as 16-bit, 8192 samples per second: playable in every browser.
    Doubling 4096 -> 8192 adds no new sound (the clips only contain 30-400 Hz)."""
    if rate == 4096:
        samples, rate = resample_poly(samples, 2, 1), 8192
    samples = samples / max(1.0, np.abs(samples).max())        # keep within -1..1
    wavfile.write(path, rate, (samples * 32767).astype(np.int16))


# Spectrogram settings shared by every picture the page shows
F_LOW, F_HIGH = 20, 500        # Hz, same range as the starter's plot
CLIP_START, CLIP_END = -2, 1   # seconds around the merger, same as the sounds
VMAX = 25                      # colour scale 0-25 "normalized energy", same as the gallery
VIRIDIS = matplotlib.colormaps['viridis']


def spectrogram_image(data, gps, n_times, n_freqs):
    """Q-transform (the starter's spectrogram) of the 3 s around the merger, as an RGB image array.
    Rows run from high pitch (top) to low pitch (bottom), on a log scale."""
    q = data.q_transform(gps=gps, search=0.5, outseg=(gps + CLIP_START, gps + CLIP_END),
                         frange=(F_LOW, F_HIGH), logf=True, fres=n_freqs,
                         tres=(CLIP_END - CLIP_START) / n_times)
    power = np.asarray(q.value)[:n_times].T[::-1]          # (freq, time), high pitch first
    rgb = VIRIDIS(np.clip(power / VMAX, 0, 1))[..., :3]
    return (rgb * 255).astype(np.uint8), float(q.frequencies.value[0]), float(q.frequencies.value[-1])


# ---------------------------------------------------------------------------------------------
# 1. GW150914: the starter's sound and a matching spectrogram
# ---------------------------------------------------------------------------------------------
GPS_150914 = 1126259462.4
starter_file = download('https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/'
                        'L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5',
                        os.path.join(DATA, 'L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5'))
data_150914 = TimeSeries.read(starter_file, format='hdf5.gwosc')
image, f_lo, f_hi = spectrogram_image(data_150914, GPS_150914, n_times=600, n_freqs=160)
Image.fromarray(image).save(os.path.join(WEB, 'gw150914_spectrogram.png'))

rate, sound = wavfile.read(os.path.join(PLOTS, '19_gw150914_livingston.wav'))
to_web_wav(sound.astype(float), rate, os.path.join(WEB, 'audio', 'gw150914.wav'))

# The same 0.7-second window as the starter's plot (0.5 s before to 0.2 s after), sharper
ZOOM = (-0.5, 0.2)
q = data_150914.q_transform(gps=GPS_150914, search=0.5, outseg=(GPS_150914 + ZOOM[0], GPS_150914 + ZOOM[1]),
                            frange=(F_LOW, F_HIGH), logf=True, fres=180, tres=(ZOOM[1] - ZOOM[0]) / 560)
zoom_power = np.asarray(q.value)[:560]
rgb = VIRIDIS(np.clip(zoom_power.T[::-1] / VMAX, 0, 1))[..., :3]
Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(WEB, 'gw150914_zoom.png'))

# When exactly did the black holes merge? Use the loudest moment of the cleaned-up signal.
# (The catalog time 1126259462.4 is rounded to 0.1 s.)
clean = data_150914.whiten(4, 2).bandpass(30, 400).crop(GPS_150914 - 0.3, GPS_150914 + 0.2)
loudest = float(clean.times.value[np.argmax(np.abs(hilbert(clean.value)))] - GPS_150914)
print(f'GW150914: loudest moment is {loudest:+.3f} s from the catalog time')

ev = catalog[catalog.commonName == 'GW150914'].iloc[0]
gw150914 = dict(name='GW150914', date=date_of(GPS_150914), m1=number(ev.mass_1_source),
                m2=number(ev.mass_2_source), final=number(ev.final_mass_source),
                total=number(ev.total_mass_source), chirp_mass=number(ev.chirp_mass_source, 2),
                redshift=number(ev.redshift, 2), travel_gyr=travel_time_gyr(ev.redshift),
                snr=number(ev.network_matched_filter_snr), merger_offset=round(loudest, 4),
                f_low=f_lo, f_high=f_hi, clip_start=CLIP_START, clip_end=CLIP_END, zoom=ZOOM, vmax=VMAX)

# A labelled figure for slides: the textbook formula drawn over the real data
T_SUN = 4.925e-6                                    # the Sun's mass as a time, G*M_sun/c^3 (s)
heard_chirp_mass = ev.chirp_mass_source * (1 + ev.redshift)
tau = np.geomspace(0.6, 0.0004, 400)                # seconds before the merger
formula_pitch = (5 / (256 * tau)) ** (3 / 8) * (T_SUN * heard_chirp_mass) ** (-5 / 8) / np.pi
fig, ax = plt.subplots(figsize=(8, 4))
mesh = ax.pcolormesh(q.times.value[:560] - GPS_150914, q.frequencies.value, zoom_power.T,
                     vmin=0, vmax=VMAX, cmap='viridis', shading='auto')
ax.plot(loudest - tau, formula_pitch, '--', color='white', lw=1.4,
        label=f'Textbook formula for {ev.mass_1_source:.1f} + {ev.mass_2_source:.1f} Suns')
ax.set_yscale('log')
ax.set_ylim(F_LOW, F_HIGH)
ax.set_xlim(*ZOOM)
ax.set_yticks([20, 30, 50, 100, 200, 300, 500], labels=['20', '30', '50', '100', '200', '300', '500'])
ax.set_xlabel('Seconds from 2015-09-14 09:50:45.4 UTC')
ax.set_ylabel('Pitch (Hz)')
ax.set_title('GW150914 at LIGO Livingston: the spiral-in formula follows the real chirp')
ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
fig.colorbar(mesh, ax=ax, label='Normalized energy')
ax.text(0, -0.32, 'Data: LIGO via GWOSC (gwosc.org). Formula: leading-order inspiral, '
        'lined up with the loudest moment in the data.', transform=ax.transAxes, fontsize=8, color='0.35')
fig.savefig(os.path.join(PLOTS, '19_gw150914_formula_vs_data.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('GW150914 done')

# ---------------------------------------------------------------------------------------------
# 2. Song of the collisions: rebuild its timetable (same rules as song_of_collisions.py)
# ---------------------------------------------------------------------------------------------
events = catalog.dropna(subset=['mass_1_source', 'mass_2_source', 'chirp_mass_source',
                                'redshift', 'network_matched_filter_snr']).copy()
events = events.sort_values('GPS').reset_index(drop=True)
events['kind'] = [kind_of(a, b) for a, b in zip(events.mass_1_source, events.mass_2_source)]
events['total_mass_source'] = events.total_mass_source.fillna(events.mass_1_source + events.mass_2_source)
events['final_mass_source'] = events.final_mass_source.fillna(0.95 * events.total_mass_source)
stretch = 1 + events.redshift

runs = pd.DataFrame(get_json('https://gwosc.org/api/v2/runs', 'gwosc_runs.json')['results'])
runs = runs.sort_values('gps_start').reset_index(drop=True)
events['run'] = None
for _, run in runs.iterrows():
    inside = (events.GPS >= run.gps_start) & (events.GPS <= run.gps_end)
    events.loc[inside, 'run'] = run['name']
events = events[events.run.notna()].reset_index(drop=True)
runs = runs[runs['name'].isin(events.run)].reset_index(drop=True)

WEEK, SONG_SEC_PER_WEEK, PAUSE, LEAD_IN, ENDING = 7 * 24 * 3600, 0.5, 2.0, 3.0, 3.0
clock = LEAD_IN
runs['song_start'] = 0.0
runs['song_end'] = 0.0
for i, run in runs.iterrows():
    runs.loc[i, 'song_start'] = clock
    clock += (run.gps_end - run.gps_start) / WEEK * SONG_SEC_PER_WEEK
    runs.loc[i, 'song_end'] = clock
    clock += PAUSE
song_length = clock - PAUSE + ENDING
info = runs.set_index('name')
events['song_time'] = [info.song_start[r] + (g - info.gps_start[r]) / WEEK * SONG_SEC_PER_WEEK
                       for g, r in zip(events.GPS, events.run)]

# Check against the real WAV file: the song lengths must agree
rate, song = wavfile.read(os.path.join(PLOTS, '19_song_of_collisions.wav'))
assert abs(len(song) / rate - song_length) < 0.01, (len(song) / rate, song_length)

T_SUN, PITCH_UP, F_START, MAX_CHIRP = 4.925e-6, 4, 20.0, 3.0
loudness = np.sqrt(events.network_matched_filter_snr / events.network_matched_filter_snr.max())
collision_rows = []
for i, e in events.iterrows():
    mc, mt, mf = (e.chirp_mass_source * stretch[i], e.total_mass_source * stretch[i],
                  e.final_mass_source * stretch[i])
    if e.kind == 'two black holes':
        f_end = 0.5349 / (2 * np.pi * T_SUN * mf)
    else:
        f_end = 1 / (6 ** 1.5 * np.pi * T_SUN * mt)
    duration = min(5 / 256 * (np.pi * F_START) ** (-8 / 3) * (T_SUN * mc) ** (-5 / 3), MAX_CHIRP)
    collision_rows.append(dict(
        name=e.commonName, kind=e.kind, run=e.run, date=date_of(e.GPS)[:10],
        t=round(e.song_time, 3), dur=round(duration, 3), pitch=round(PITCH_UP * f_end, 1),
        loud=round(float(loudness[i]), 3), m1=number(e.mass_1_source), m2=number(e.mass_2_source),
        snr=number(e.network_matched_filter_snr), travel_gyr=travel_time_gyr(e.redshift),
        mc=round(mc, 3), mt=round(mt, 3), mf=round(mf, 3)))   # masses as heard: lets the page replay one chirp

run_rows = []
for _, run in runs.iterrows():
    y0, y1 = Time([run.gps_start, run.gps_end], format='gps').to_datetime()
    run_rows.append(dict(name=run['name'], start=round(run.song_start, 3), end=round(run.song_end, 3),
                         years=f'{y0.year}' if y0.year == y1.year else f'{y0.year}-{str(y1.year)[2:]}',
                         n=int((events.run == run['name']).sum())))
print(f'Song of the collisions: {len(collision_rows)} chirps, {song_length:.1f} s')

# ---------------------------------------------------------------------------------------------
# 3. Song of real recordings: find the 91 clips (same rules as song_of_real_recordings.py),
#    check each against the WAV file, and draw a small spectrogram for each one
# ---------------------------------------------------------------------------------------------
all_versions = get_json('https://gwosc.org/eventapi/jsonfull/allevents/', 'gwosc_allevents.json')['events']
USE_HANFORD = {'GW170817'}


def small_file_url(entries, detector):
    url = None
    for entry in sorted(entries, key=lambda e: e['version']):
        for s in entry['strain']:
            if (s['detector'] == detector and s['duration'] == 32 and s['sampling_rate'] == 4096
                    and s['format'] == 'hdf5'):
                url = s['url']
    return url


versions_of = {}
for entry in all_versions.values():
    versions_of.setdefault(entry['commonName'], []).append(entry)
small_file = {}
for name, entries in versions_of.items():
    for det in (['H1', 'L1'] if name in USE_HANFORD else ['L1', 'H1']):
        url = small_file_url(entries, det)
        if url:
            small_file[name] = (det, url)
            break

candidates = catalog[catalog.commonName.isin(small_file)].sort_values('GPS').reset_index(drop=True)
print(f'{len(candidates)} candidate clips; downloading their 32-second files (about 1 MB each)...')
with ThreadPoolExecutor(8) as pool:
    list(pool.map(lambda n: download(small_file[n][1], os.path.join(CLIPS, os.path.basename(small_file[n][1]))),
                  candidates.commonName))

rate, real_song = wavfile.read(os.path.join(PLOTS, '19_real_recordings_song.wav'))
CLIP = (CLIP_END - CLIP_START) * rate          # samples per clip
FADE = int(0.02 * rate)
TILE_T, TILE_F = 300, 64                       # each small spectrogram: 300 times x 64 pitches
clip_rows, tiles = [], []
for _, e in candidates.iterrows():
    det, url = small_file[e.commonName]
    data = TimeSeries.read(os.path.join(CLIPS, os.path.basename(url)), format='hdf5.gwosc')
    try:
        sound = data.whiten(4, 2).bandpass(30, 400).crop(e.GPS + CLIP_START, e.GPS + CLIP_END)
        clip = sound.value / np.abs(sound.value).max()
        if not np.all(np.isfinite(clip)):
            raise ValueError('data contains gaps (NaN)')
    except Exception as err:     # the song script left these out too
        print(f'  left out (as in the song): {e.commonName}: {err}')
        continue
    ramp = np.linspace(0, 1, FADE)
    clip[:FADE] *= ramp
    clip[-FADE:] *= ramp[::-1]
    k = len(clip_rows)
    in_song = real_song[k * CLIP:(k + 1) * CLIP]
    assert np.allclose(clip, in_song, atol=1e-4), f'clip {k} ({e.commonName}) does not match the song'
    image, f_lo, f_hi = spectrogram_image(data, e.GPS, TILE_T, TILE_F)
    tiles.append(image)
    clip_rows.append(dict(
        name=e.commonName, det={'L1': 'Livingston', 'H1': 'Hanford'}[det], date=date_of(e.GPS),
        start=k * (CLIP_END - CLIP_START), kind=kind_of(e.mass_1_source, e.mass_2_source),
        m1=number(e.mass_1_source), m2=number(e.mass_2_source),
        snr=number(e.network_matched_filter_snr), travel_gyr=travel_time_gyr(e.redshift)))
    print(f'  {k + 1:3d} {e.commonName:18} {det}  matches the song', flush=True)
assert len(clip_rows) * CLIP == len(real_song), 'clip count does not match the song length'

# All small spectrograms in one picture: 7 across, in song order
COLS = 7
rows = int(np.ceil(len(tiles) / COLS))
sprite = np.zeros((rows * TILE_F, COLS * TILE_T, 3), np.uint8)
for k, tile in enumerate(tiles):
    r, c = divmod(k, COLS)
    sprite[r * TILE_F:(r + 1) * TILE_F, c * TILE_T:(c + 1) * TILE_T] = tile
Image.fromarray(sprite).save(os.path.join(WEB, 'clip_spectrograms.jpg'), quality=88)
to_web_wav(real_song.astype(float), rate, os.path.join(WEB, 'audio', 'real_recordings_song.wav'))

# ---------------------------------------------------------------------------------------------
# 4. Everything the page needs, in one file
# ---------------------------------------------------------------------------------------------
with open(os.path.join(WEB, 'data.json'), 'w') as f:
    json.dump(dict(
        source='LIGO/Virgo/KAGRA via the Gravitational Wave Open Science Center (GWOSC), https://gwosc.org',
        gw150914=gw150914,
        collisions=dict(length=round(song_length, 3), pitch_up=PITCH_UP, events=collision_rows, runs=run_rows),
        real=dict(clip_seconds=CLIP_END - CLIP_START, merger_at=-CLIP_START, tile=[TILE_T, TILE_F],
                  sprite_cols=COLS, f_low=f_lo, f_high=f_hi, vmax=VMAX, clips=clip_rows),
    ), f, separators=(',', ':'))
print(f'\nWrote {len(clip_rows)} clips and {len(collision_rows)} chirps to {WEB}/data.json')
