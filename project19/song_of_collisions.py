"""Song of the collisions: every measured merger in the LIGO-Virgo-KAGRA catalog, turned into sound.

Run from the repo root:   python3 project19/song_of_collisions.py
It makes two files:
    plots/19_song_of_collisions.wav         the song (about 1 minute 50 seconds)
    plots/19_song_of_collisions_score.png   a picture of the song, to follow along while listening

HOW THE SONG WORKS
* One chirp per merger, for every event in the catalog whose masses have been measured.
  (The catalog also lists weaker candidates with no masses yet; with no masses there is
  nothing to turn into sound, so they are left out.)
* Each chirp is a MODEL computed from that event's measured masses with the textbook
  formula for two objects spiralling together. It is NOT the detector recording: most real
  recordings are buried in noise and would just sound like hiss.
  Heavier pairs give a short, low "thump"; lighter pairs give a long, rising whistle.
* Every pitch is raised 2 octaves (x4) so laptop speakers can play it. The timing of each
  chirp is left at its real speed.
* A louder chirp means the event stood out more clearly from the detector noise
  (a higher signal-to-noise ratio, or SNR).
* The song's clock only runs while the detectors were observing. One soft tick = one week
  of observing, with a slightly louder tick every 4 weeks. Each chirp peaks at the moment
  the merger was detected. A bell and a short pause mark the start of each observing run
  (O1, O2, ...); between runs the detectors were switched off for upgrades.

DATA (please cite): Gravitational Wave Open Science Center (GWOSC), https://gwosc.org
    event catalog:  https://gwosc.org/eventapi/jsonfull/GWTC/
    observing runs: https://gwosc.org/api/v2/runs
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.time import Time
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo folder
DATA = os.path.join(ROOT, 'data')
PLOTS = os.path.join(ROOT, 'plots')

# Same certificate note as starters/19_hear_a_merger.py: gwosc.org's certificate is signed by
# "emSign Root CA - G1", which the operating system's list includes but certifi's doesn't.
# This keeps the security check on; it only chooses which list of authorities to trust.
SYSTEM_CERTS = '/etc/ssl/certs/ca-certificates.crt'
VERIFY = SYSTEM_CERTS if os.path.exists(SYSTEM_CERTS) else True


def get_json(url, filename):
    """Download a JSON file once and keep a copy in data/, so reruns don't download it again."""
    path = os.path.join(DATA, filename)
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        r = requests.get(url, verify=VERIFY, timeout=120)
        r.raise_for_status()
        with open(path, 'w') as f:
            f.write(r.text)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------
# 1. Load the catalog and the observing runs
# ---------------------------------------------------------------------------------------------
catalog = get_json('https://gwosc.org/eventapi/jsonfull/GWTC/', 'gwosc_gwtc_catalog.json')
events = pd.DataFrame(catalog['events'].values())
numbers = ['GPS', 'mass_1_source', 'mass_2_source', 'chirp_mass_source', 'total_mass_source',
           'final_mass_source', 'redshift', 'network_matched_filter_snr']
for col in numbers:
    events[col] = pd.to_numeric(events[col], errors='coerce')   # the API sends numbers as text
n_catalog = len(events)

# Keep events whose masses, distance (redshift) and loudness (SNR) have all been measured.
events = events.dropna(subset=['mass_1_source', 'mass_2_source', 'chirp_mass_source',
                               'redshift', 'network_matched_filter_snr'])
events = events.sort_values('GPS').reset_index(drop=True)
print(f'{n_catalog} events in the catalog; {len(events)} have measured masses -> one chirp each')

# What kind of objects merged? Anything lighter than about 3 Suns is treated as a neutron star.
NS_LIMIT = 3.0
events['kind'] = 'two black holes'
events.loc[events.mass_2_source < NS_LIMIT, 'kind'] = 'black hole + neutron star'
events.loc[events.mass_1_source < NS_LIMIT, 'kind'] = 'two neutron stars'

# Fill gaps: total mass = sum of the two; the final black hole is about 5% lighter than the pair
# (the missing mass left as gravitational-wave energy).
events['total_mass_source'] = events.total_mass_source.fillna(events.mass_1_source + events.mass_2_source)
events['final_mass_source'] = events.final_mass_source.fillna(0.95 * events.total_mass_source)

# The waves are stretched on their way to us by the expansion of the universe, so the detector
# "hears" the masses multiplied by (1 + redshift). That's what sets the pitch we hear.
stretch = 1 + events.redshift
events['chirp_mass_heard'] = events.chirp_mass_source * stretch
events['total_mass_heard'] = events.total_mass_source * stretch
events['final_mass_heard'] = events.final_mass_source * stretch

runs = pd.DataFrame(get_json('https://gwosc.org/api/v2/runs', 'gwosc_runs.json')['results'])
runs = runs.sort_values('gps_start').reset_index(drop=True)

# Which observing run was each event detected in?
events['run'] = None
for _, run in runs.iterrows():
    inside = (events.GPS >= run.gps_start) & (events.GPS <= run.gps_end)
    events.loc[inside, 'run'] = run['name']
missing = events.run.isna()
if missing.any():
    print('Warning: these events fall outside every observing run and are left out:',
          events.loc[missing, 'commonName'].tolist())
    events = events[~missing].reset_index(drop=True)
runs = runs[runs['name'].isin(events.run)].reset_index(drop=True)   # only runs with events

# ---------------------------------------------------------------------------------------------
# 2. The song's clock: only count time while the detectors were observing
# ---------------------------------------------------------------------------------------------
WEEK = 7 * 24 * 3600         # seconds in a week
SONG_SEC_PER_WEEK = 0.5      # one week of observing = half a second of song (120 ticks a minute)
PAUSE = 2.0                  # seconds of song between observing runs
LEAD_IN = 3.0                # seconds of song before the first run starts
ENDING = 3.0                 # seconds after the last run, so the last chirp can ring out

clock = LEAD_IN
runs['song_start'] = 0.0
runs['song_end'] = 0.0
for i, run in runs.iterrows():
    runs.loc[i, 'song_start'] = clock
    clock += (run.gps_end - run.gps_start) / WEEK * SONG_SEC_PER_WEEK
    runs.loc[i, 'song_end'] = clock
    clock += PAUSE
song_length = clock - PAUSE + ENDING

run_info = runs.set_index('name')
events['song_time'] = [run_info.song_start[r] + (gps - run_info.gps_start[r]) / WEEK * SONG_SEC_PER_WEEK
                       for gps, r in zip(events.GPS, events.run)]

# ---------------------------------------------------------------------------------------------
# 3. The sound of one merger (a model chirp)
# ---------------------------------------------------------------------------------------------
RATE = 22050          # samples per second in the WAV file
PITCH_UP = 4          # raise every pitch 2 octaves so laptop speakers can play it
F_START = 20.0        # Hz: the lowest gravitational-wave pitch LIGO can pick up
MAX_CHIRP = 3.0       # s: longest chirp we play (neutron-star chirps really last minutes)
T_SUN = 4.925e-6      # the Sun's mass written as a time: G * M_sun / c^3, in seconds


def chirp_sound(chirp_mass, total_mass, final_mass, black_holes):
    """Model chirp for one merger. Masses are in Suns, as heard by the detector.

    Returns (wave, merger_index): the samples, and which sample is the moment of merger.
    """
    mass_time = T_SUN * chirp_mass

    # Textbook (leading-order) formula for two objects spiralling together: the wave's pitch f
    # at a time tau before they merge, and the reverse (how long before merger the pitch is f).
    def pitch(tau):
        return (5 / (256 * tau)) ** (3 / 8) * mass_time ** (-5 / 8) / np.pi

    def time_before_merger(f):
        return 5 / 256 * (np.pi * f) ** (-8 / 3) * mass_time ** (-5 / 3)

    if black_holes:
        # The new, single black hole "rings" like a struck bell at one pitch, then goes quiet.
        # Standard fit from black-hole theory, assuming a typical final spin of 0.7.
        f_end = 0.5349 / (2 * np.pi * T_SUN * final_mass)
        ring_time = 3.29 / (np.pi * f_end)
    else:
        # With a neutron star, stop at the closest stable orbit: after that it is torn apart or
        # swallowed, and the signal fades quickly.
        f_end = 1 / (6 ** 1.5 * np.pi * T_SUN * total_mass)
        ring_time = 0.005

    # Inspiral: count down from the start of the chirp to the moment the pitch reaches f_end.
    tau_start = min(time_before_merger(F_START), MAX_CHIRP)
    tau = np.arange(tau_start, time_before_merger(f_end), -1 / RATE)
    f_inspiral = pitch(tau)
    amp_inspiral = (f_inspiral / f_end) ** (2 / 3)        # it gets louder as it speeds up

    # Ringdown: one steady pitch that dies away.
    t_ring = np.arange(0, 6 * ring_time, 1 / RATE)
    f = np.concatenate([f_inspiral, np.full(len(t_ring), f_end)]) * PITCH_UP
    amp = np.concatenate([amp_inspiral, np.exp(-t_ring / ring_time)])

    phase = 2 * np.pi * np.cumsum(f) / RATE              # add up the pitch to get the wave's phase
    wave = amp * np.sin(phase)
    fade_in = np.clip(np.arange(len(wave)) / (0.02 * RATE), 0, 1)   # 20 ms fade-in avoids a click
    return wave * fade_in, len(f_inspiral)


# ---------------------------------------------------------------------------------------------
# 4. Mix everything into one track
# ---------------------------------------------------------------------------------------------
song = np.zeros(int(song_length * RATE) + 1)


def add(sound, at_time, merger_index=0):
    """Add a sound to the song so that its merger_index sample lands at at_time (seconds)."""
    start = int(round(at_time * RATE)) - merger_index
    lo, hi = max(start, 0), min(start + len(sound), len(song))
    if hi > lo:
        song[lo:hi] += sound[lo - start:hi - start]


# Loudness: brighter detections are louder. Square root, so the quieter ones are still audible.
loudness = np.sqrt(events.network_matched_filter_snr / events.network_matched_filter_snr.max())
events['ending_pitch'] = 0.0
for i, ev in events.iterrows():
    wave, merger = chirp_sound(ev.chirp_mass_heard, ev.total_mass_heard, ev.final_mass_heard,
                               ev.kind == 'two black holes')
    add(loudness[i] * wave, ev.song_time, merger)
    # Remember the highest pitch we play for this event (for the picture)
    if ev.kind == 'two black holes':
        events.loc[i, 'ending_pitch'] = PITCH_UP * 0.5349 / (2 * np.pi * T_SUN * ev.final_mass_heard)
    else:
        events.loc[i, 'ending_pitch'] = PITCH_UP / (6 ** 1.5 * np.pi * T_SUN * ev.total_mass_heard)

# A soft tick for every week of observing, a little louder every 4 weeks.
t = np.arange(0, 0.03, 1 / RATE)
tick = np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.004)
for _, run in runs.iterrows():
    for k, tick_time in enumerate(np.arange(run.song_start, run.song_end, SONG_SEC_PER_WEEK)):
        add((0.10 if k % 4 == 0 else 0.05) * tick, tick_time)

# A bell just before each observing run starts.
t = np.arange(0, 2.5, 1 / RATE)
bell = sum(a * np.sin(2 * np.pi * f * t) for f, a in [(523, 1.0), (1046, 0.5), (1569, 0.25)]) * np.exp(-t / 0.6)
for _, run in runs.iterrows():
    add(0.25 * bell, run.song_start - 1.5)

song *= 0.9 / np.abs(song).max()                            # loudest moment at 90% volume
song[-int(RATE):] *= np.linspace(1, 0, int(RATE))           # 1-second fade-out at the end
os.makedirs(PLOTS, exist_ok=True)
wav_path = os.path.join(PLOTS, '19_song_of_collisions.wav')
wavfile.write(wav_path, RATE, (song * 32767).astype(np.int16))
print(f'Saved {wav_path}: {len(song) / RATE:.0f} s long')

# ---------------------------------------------------------------------------------------------
# 5. The "score": a picture of the song
# ---------------------------------------------------------------------------------------------
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#898781'
SURFACE, GRID, BAND = '#fcfcfb', '#e1e0d9', '#f0efec'
COLORS = {'two black holes': '#2a78d6', 'black hole + neutron star': '#eb6834',
          'two neutron stars': '#1baf7a'}

fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

# Shade each observing run and label it with its name and years
for _, run in runs.iterrows():
    ax.axvspan(run.song_start, run.song_end, color=BAND, zorder=0, lw=0)
    y0, y1 = Time([run.gps_start, run.gps_end], format='gps').to_datetime()
    years = f'{y0.year}' if y0.year == y1.year else f'{y0.year}-{str(y1.year)[2:]}'
    n = (events.run == run['name']).sum()
    ax.text((run.song_start + run.song_end) / 2, 1.02, f"{run['name']}  {years}\n{n} mergers",
            transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9, color=INK_2)

# One dot per chirp: when it lands, the pitch it ends on, sized by loudness
for kind, color in COLORS.items():
    sel = events[events.kind == kind]
    ax.scatter(sel.song_time, sel.ending_pitch, s=8 + 260 * loudness[sel.index] ** 2,
               color=color, edgecolor=SURFACE, linewidth=1.2, alpha=0.9, zorder=3,
               label=f'{kind} ({len(sel)})')

# Label a few famous ones, with their two masses (in Suns) taken from the catalog
famous = {'GW150914': 'the first', 'GW170817': 'two neutron stars, also seen in light',
          'GW190521': 'very heavy', 'GW231123_135430': 'heaviest so far',
          'GW250114_082203': 'loudest so far'}
offsets = {'GW150914': (8, -22), 'GW170817': (8, -14), 'GW190521': (8, -14),
           'GW231123_135430': (-10, -16), 'GW250114_082203': (-20, 40)}   # where each label sits


def suns(m):
    return f'{m:.1f}' if m < 10 else f'{m:.0f}'


for name, description in famous.items():
    row = events[events.commonName == name]
    if len(row):
        r = row.iloc[0]
        label = f"{name.split('_')[0]}: {description} ({suns(r.mass_1_source)} + {suns(r.mass_2_source)} Suns)"
        dx, dy = offsets[name]
        ax.annotate(label, (r.song_time, r.ending_pitch), xytext=(dx, dy), textcoords='offset points',
                    fontsize=9, color=INK, ha='left' if dx > 0 else 'right', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', fc=SURFACE, ec='none', alpha=0.85),
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))

ax.set_yscale('log')
pitch_ticks = [250, 500, 1000, 2000, 4000]
ax.set_yticks(pitch_ticks, labels=[str(p) for p in pitch_ticks])   # plain numbers, not 10^3
ax.minorticks_off()
ax.set_ylim(events.ending_pitch.min() * 0.75, events.ending_pitch.max() * 1.25)   # room for labels
ax.set_xlim(0, song_length)
ax.set_xlabel('Time in the song (seconds)', color=INK_2)
ax.set_ylabel('Highest pitch of the chirp, as played (Hz)', color=INK_2)
ax.grid(True, which='major', color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
for side in ['left', 'bottom']:
    ax.spines[side].set_color('#c3c2b7')
ax.tick_params(colors=MUTED)
legend = ax.legend(loc='lower left', frameon=True, fontsize=9, title='What merged', title_fontsize=9)
legend.get_frame().set_edgecolor(GRID)
fig.suptitle(f'Song of the collisions: {len(events)} mergers heard by LIGO, Virgo and KAGRA',
             x=0.06, ha='left', fontsize=14, color=INK, y=0.99)
fig.text(0.06, 0.915, 'Each dot is one model chirp in 19_song_of_collisions.wav. Lower dot = heavier pair '
         '= lower, shorter chirp. Bigger dot = louder (clearer detection).\n'
         'Shaded = detectors observing (0.5 s of song per week). Pitches raised 2 octaves. '
         'Data: GWOSC event catalog (gwosc.org).', ha='left', va='top', fontsize=9, color=INK_2)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.09, top=0.78)
png_path = os.path.join(PLOTS, '19_song_of_collisions_score.png')
fig.savefig(png_path, dpi=150, facecolor=SURFACE)
print(f'Saved {png_path}')

print('\nMergers per observing run:')
print(events.groupby('run', sort=False).size().to_string())
