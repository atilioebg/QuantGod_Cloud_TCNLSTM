import sys, yaml
from pathlib import Path
sys.path.insert(0, '.')
import polars as pl
import pandas as pd
from collections import Counter

with open('src/cloud/base_model/configs/master_config.yaml') as f:
    config = yaml.safe_load(f)

sell_th = 0.003
buy_th  = 0.003
mins    = 15
lookahead_bars = 3  # 15min / 5min

pre_file = Path('data/L2/pre_processed/2023-02-09_BTCUSDT_ob500.data.parquet')
df = pl.read_parquet(pre_file)
print(f'Rows: {len(df)} | island_id present: {"island_id" in df.columns}')

island_col = 'island_id' if 'island_id' in df.columns else None
total_valid = len(df) - lookahead_bars

# === LÓGICA ANTIGA: High/Low rolling ===
if island_col:
    df_a = df.with_columns([
        pl.col('high').rolling_max(window_size=lookahead_bars).shift(-lookahead_bars).over(island_col).alias('fmh'),
        pl.col('low').rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).over(island_col).alias('fml'),
        pl.col('tick_count').rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).over(island_col).alias('ftk'),
    ])
else:
    df_a = df.with_columns([
        pl.col('high').rolling_max(window_size=lookahead_bars).shift(-lookahead_bars).alias('fmh'),
        pl.col('low').rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).alias('fml'),
        pl.col('tick_count').rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).alias('ftk'),
    ])

df_a = df_a.slice(0, total_valid).with_columns([
    pl.when(pl.col('ftk').is_null() | (pl.col('ftk') == 0)).then(None)
    .when(pl.col('fmh') >= pl.col('close') * (1 + buy_th)).then(pl.lit(2, dtype=pl.Int8))
    .when(pl.col('fml') <= pl.col('close') * (1 - sell_th)).then(pl.lit(0, dtype=pl.Int8))
    .otherwise(pl.lit(1, dtype=pl.Int8)).alias('old_t')
])

# === LÓGICA NOVA: Close shift + island boundary ===
if island_col:
    df_n = df.with_columns([
        pl.col('close').shift(-lookahead_bars).over(island_col).alias('fc'),
        pl.col(island_col).shift(-lookahead_bars).over(island_col).alias('fi'),
        pl.col('tick_count').rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).over(island_col).alias('ftk2'),
    ])
    df_n = df_n.slice(0, total_valid).with_columns([
        pl.when(pl.col('ftk2').is_null() | (pl.col('ftk2') == 0)).then(None)
        .when(pl.col('fi').is_null() | (pl.col('fi') != pl.col(island_col))).then(None)
        .when(pl.col('fc') >= pl.col('close') * (1 + buy_th)).then(pl.lit(2, dtype=pl.Int8))
        .when(pl.col('fc') <= pl.col('close') * (1 - sell_th)).then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(1, dtype=pl.Int8)).alias('new_t')
    ])
else:
    df_n = df.with_columns([
        pl.col('close').shift(-lookahead_bars).alias('fc'),
        pl.col('tick_count').rolling_min(window_size=lookahead_bars).shift(-lookahead_bars).alias('ftk2'),
    ])
    df_n = df_n.slice(0, total_valid).with_columns([
        pl.when(pl.col('ftk2').is_null() | (pl.col('ftk2') == 0)).then(None)
        .when(pl.col('fc') >= pl.col('close') * (1 + buy_th)).then(pl.lit(2, dtype=pl.Int8))
        .when(pl.col('fc') <= pl.col('close') * (1 - sell_th)).then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(1, dtype=pl.Int8)).alias('new_t')
    ])

# === COMPARAÇÃO ===
cmp = df_a.select(['old_t']).hstack(df_n.select(['new_t']))
changed = cmp.filter(
    pl.col('old_t').is_not_null() &
    pl.col('new_t').is_not_null() &
    (pl.col('old_t') != pl.col('new_t'))
)

lm = {0: 'SELL', 1: 'NEUTRAL', 2: 'BUY', None: 'NaN'}

print()
print('=' * 60)
print(f'AUDIT 2023-02-09  lookahead={lookahead_bars} barras / {mins}min')
print('=' * 60)
print(f'Total valido             : {total_valid}')
print(f'NaN antigas              : {cmp["old_t"].null_count()}')
print(f'NaN novas                : {cmp["new_t"].null_count()}')
print(f'NaN delta                : {cmp["new_t"].null_count() - cmp["old_t"].null_count():+d}')
print(f'Labels MUDARAM           : {len(changed)} ({len(changed)/total_valid*100:.2f}%)')
print()

trans = Counter((r['old_t'], r['new_t']) for r in changed.to_dicts())
if trans:
    print('Transicoes (ANTIGO -> NOVO):')
    for (o, n), c in sorted(trans.items(), key=lambda x: -x[1]):
        print(f'  {lm.get(o):<8} -> {lm.get(n):<8}: {c:>5} samples')
else:
    print('Nenhuma mudanca de label detectada!')

ov = {r['old_t']: r['count'] for r in cmp.drop_nulls('old_t')['old_t'].value_counts().to_dicts()}
nv = {r['new_t']: r['count'] for r in cmp.drop_nulls('new_t')['new_t'].value_counts().to_dicts()}
print()
print(f'  {"Label":<10} {"ANTIGO":>8} {"NOVO":>8} {"DELTA":>8}')
for lbl, nm in [(2, 'BUY'), (0, 'SELL'), (1, 'NEUTRAL')]:
    o = ov.get(lbl, 0)
    n = nv.get(lbl, 0)
    print(f'  {nm:<10} {o:>8} {n:>8} {n-o:>+8}')
print('=' * 60)
