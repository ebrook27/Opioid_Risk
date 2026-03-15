"""Plot county-level unemployment trends by state.

Creates one plot per state (by two-digit FIPS) showing each county
as a grey line and the state average as a black line.

Usage:
    python scripts/plot_unemployment_trends.py \
        --input data/Processed/Unemployment/Unemployment_rates_CT_fixed.csv \
        --state-codes data/Raw/STATE_FIPS_CODES.txt \
        --outdir plots/unemployment_by_state --save

"""
from pathlib import Path
import re
import argparse
import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_state_codes(path):
    df = pd.read_csv(path, dtype=str)
    df['FIPS'] = df['FIPS'].str.zfill(2)
    df = df.set_index('FIPS')['STATE'].to_dict()
    return df


def find_year_cols(columns):
    years = []
    for c in columns:
        m = re.search(r'(20\d{2})', c)
        if m:
            years.append((c, int(m.group(1))))
    years_sorted = [c for c, y in sorted(years, key=lambda t: t[1])]
    return years_sorted


def prepare_long_df(df, year_cols):
    id_vars = [c for c in df.columns if c not in year_cols]
    long = df.melt(id_vars=id_vars, value_vars=year_cols,
                   var_name='year_label', value_name='unemployment')
    # extract year numeric
    long['year'] = long['year_label'].str.extract(r'(20\d{2})').astype(float)
    # ensure numeric unemployment
    long['unemployment'] = pd.to_numeric(long['unemployment'], errors='coerce')
    return long


def plot_state(df_long, state_fips, state_name, outdir=None, save=False, dpi=150, fmt='png', ylim=None):
    s = df_long[df_long['state_fips'] == state_fips]
    if s.empty:
        return

    # compute state average by year
    state_avg = s.groupby('year', as_index=False)['unemployment'].mean()

    years = sorted(s['year'].dropna().unique())
    years = [int(y) for y in years]

    plt.figure(figsize=(9, 6))
    sns.set_style('whitegrid')

    # plot each county
    for county, grp in s.groupby('FIPS'):
        plt.plot(grp['year'], grp['unemployment'], color='grey', alpha=0.6, linewidth=0.8)

    # plot state average
    plt.plot(state_avg['year'], state_avg['unemployment'], color='black', linewidth=2.2, label='State average')

    plt.xlabel('Year')
    plt.ylabel('Unemployment Rate')
    title = f"{state_name.title()} Unemployment Trends, 2010-2022"
    plt.title(title)
    plt.xlim(min(years), max(years))
    plt.xticks(years, [str(y) for y in years], rotation=45)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()

    if save and outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fname = outdir / f"unemployment_trends_state_{state_fips}.{fmt}"
        plt.savefig(fname, dpi=dpi)
        plt.close()
    else:
        plt.show()


def main(args):
    input_path = Path(args.input)
    state_codes_path = Path(args.state_codes)

    df = pd.read_csv(input_path, dtype=str)

    # determine year columns (columns that contain a 20XX year)
    year_cols = find_year_cols(df.columns)
    if not year_cols:
        raise SystemExit('No year columns detected in input file')

    df_long = prepare_long_df(df, year_cols)

    # derive county 5-digit FIPS and state 2-digit
    df_long['FIPS'] = df_long['FIPS'].astype(str).str.zfill(5)
    df_long['state_fips'] = df_long['FIPS'].str[:2]

    # numeric conversions
    df_long['year'] = df_long['year'].astype(float)
    df_long['unemployment'] = pd.to_numeric(df_long['unemployment'], errors='coerce')

    state_map = load_state_codes(state_codes_path)

    # compute global y-limits so plots are comparable across states
    if df_long['unemployment'].notna().any():
        global_min = float(df_long['unemployment'].min(skipna=True))
        global_max = float(df_long['unemployment'].max(skipna=True))
        ymin = max(0.0, math.floor(global_min * 10) / 10.0)
        # add 5% headroom and round up to one decimal place
        ymax = math.ceil((global_max * 1.05) * 10) / 10.0
        ylim = (ymin, ymax)
    else:
        ylim = None

    states = sorted(df_long['state_fips'].unique())

    for st in states:
        name = state_map.get(st, f'State {st}')
        plot_state(df_long, st, name, outdir=args.outdir, save=args.save, dpi=args.dpi, fmt=args.format, ylim=ylim)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/Processed/Unemployment/Unemployment_rates_CT_fixed.csv')
    p.add_argument('--state-codes', default='data/Raw/STATE_FIPS_CODES.txt')
    p.add_argument('--outdir', default='plots/unemployment_by_state')
    p.add_argument('--save', action='store_true', help='Save plots to outdir instead of showing')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--format', default='png', help='Image format when saving (png, pdf, etc)')
    args = p.parse_args()
    main(args)
