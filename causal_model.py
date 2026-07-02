import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from xml.etree import ElementTree as ET
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from econml.dml import CausalForestDML

BASE_DIR     = Path(__file__).parent
RIDER_DIR    = BASE_DIR / 'rider_data'
GC_DIR       = BASE_DIR / 'riders_gc'

# Lookup (course, year) → startlist_quality pour calculer N-1
_sq_lookup_path  = BASE_DIR / 'race_data' / 'startlist_quality_lookup.csv'
_pts1_lookup_path = BASE_DIR / 'race_data' / 'race_pts1_lookup.csv'
_SQ_LOOKUP: pd.DataFrame | None = None
_PTS1_LOOKUP: pd.Series | None = None

def _get_sq_lookup() -> pd.DataFrame:
    global _SQ_LOOKUP
    if _SQ_LOOKUP is None and _sq_lookup_path.exists():
        _SQ_LOOKUP = pd.read_csv(_sq_lookup_path)
        _SQ_LOOKUP['year'] = _SQ_LOOKUP['year'].astype(int)
    return _SQ_LOOKUP

def _get_pts1_lookup() -> pd.Series | None:
    global _PTS1_LOOKUP
    if _PTS1_LOOKUP is None and _pts1_lookup_path.exists():
        df = pd.read_csv(_pts1_lookup_path)
        _PTS1_LOOKUP = df.set_index('course')['pts_uci_rank1']
    return _PTS1_LOOKUP
OUTCOME      = 'pts_uci_equipe_stage'
TREATMENT    = 'selected'
N_FOLDS      = 5
N_BOOT       = 200
N_TREES      = 300
RANDOM_STATE = 42

# Noms canoniques → toutes les variantes historiques (WorldTour uniquement)
TEAM_GROUPS: dict[str, list[str]] = {
    "Visma | Lease a Bike":         ["Team LottoNL-Jumbo", "Team Jumbo-Visma", "Jumbo-Visma", "Team Visma | Lease a Bike"],
    "INEOS Grenadiers":             ["Team Sky", "Team INEOS", "INEOS Grenadiers"],
    "UAE Team Emirates":            ["UAE Team Emirates", "UAE Team Emirates - XRG"],
    "Soudal Quick-Step":            ["Quick-Step Floors", "Deceuninck - Quick Step", "Quick-Step Alpha Vinyl Team", "Soudal Quick-Step"],
    "Red Bull - BORA - hansgrohe":  ["BORA - hansgrohe", "Red Bull - BORA - hansgrohe"],
    "Lidl - Trek":                  ["Trek - Segafredo", "Lidl - Trek"],
    "EF Education - EasyPost":      ["Team EF Education First - Drapac p/b Cannondale", "EF Education First", "EF Pro Cycling", "EF Education - Nippo", "EF Education - EasyPost"],
    "Groupama - FDJ":               ["Groupama - FDJ"],
    "Lotto Dstny":                  ["Lotto Soudal", "Lotto Dstny", "Lotto"],
    "Decathlon AG2R":               ["AG2R La Mondiale", "AG2R Citroën Team", "Decathlon AG2R La Mondiale Team"],
    "Astana Qazaqstan":             ["Astana Pro Team", "Astana - Premier Tech", "Astana Qazaqstan Team", "XDS Astana Team"],
    "Team dsm-firmenich":           ["Team Sunweb", "Team DSM", "Team dsm - firmenich", "Team dsm-firmenich PostNL"],
    "Bahrain Victorious":           ["Bahrain Merida", "Bahrain Merida Pro Cycling Team", "Bahrain - McLaren", "Bahrain - Victorious"],
    "Jayco AlUla":                  ["Mitchelton-Scott", "Team BikeExchange", "Team BikeExchange - Jayco", "Team Jayco AlUla"],
    "Intermarché - Wanty":          ["Wanty - Groupe Gobert", "Wanty - Gobert Cycling Team", "Circus - Wanty Gobert", "Intermarché - Wanty - Gobert Matériaux", "Intermarché - Circus - Wanty", "Intermarché - Wanty"],
    "Alpecin - Deceuninck":         ["Alpecin - Fenix", "Alpecin - Deceuninck"],
    "Cofidis":                      ["Cofidis", "Cofidis, Solutions Crédits"],
    "Movistar Team":                ["Movistar Team"],
    "Uno-X Mobility":               ["Uno-X Pro Cycling Team", "Uno-X Mobility"],
    "Arkéa":                        ["Team Fortuneo - Samsic", "Team Arkéa Samsic", "Arkéa - B&B Hotels"],
}

def expand_team(name: str) -> list[str]:
    """Retourne toutes les variantes historiques d'une équipe (ou [name] si inconnu)."""
    return TEAM_GROUPS.get(name, [name])

FEATURES_RACE = [
    'distance_gpx_km', 'denivele_pos', 'denivele_neg',
    'altitude_max', 'altitude_min',
    'n_cols_cat4', 'n_cols_cat3', 'n_cols_cat2', 'n_cols_cat1', 'n_cols_hc',
    'loc_last_col_cat2', 'loc_last_col_cat1', 'loc_last_col_hc',
    'gradient_last_1km', 'gradient_last_3km', 'gradient_last_5km',
    'denivele_last_5km', 'gradient_first_50km', 'denivele_first_50km',
    'cobblestones_km', 'compacted_gravel_km',
    'cobblestones_last_10km', 'compacted_gravel_last_10km',
    'startlist_quality', 'startlist_quality_prev',
]
FEATURES_DYNAMIC = ['forme_equipe', 'n_races_30d', 'km_30d', 'leader_played', 'is_team_leader', 'year']
ALL_FEATURES     = FEATURES_RACE + FEATURES_DYNAMIC

AVAILABLE_OUTCOMES = [
    'pts_uci_equipe_stage',
    'pts_uci_pct_max',   # pts_uci_equipe_stage normalisé par le max historique de la course
    'pts_uci',
    'pts_uci_teammates', # pts équipe hors le coureur lui-même (effet domestique/leadership sur les autres)
    # pts_uci_gc / pts_uci_kom / pts_uci_points sont des totaux de course dupliqués sur toutes les étapes
    # → modèle race-level séparé (run_analysis_race_level)
]

# Outcomes du modèle race-level — colonnes de riders_gc/ (une ligne par course)
AVAILABLE_OUTCOMES_RACE = ['pts_uci_equipe_gc', 'pts_uci_equipe_kom', 'pts_uci_equipe_points']

# Features agrégées au niveau course
_RACE_SUM_FEATS   = [
    'distance_gpx_km', 'denivele_pos', 'denivele_neg',
    'n_cols_cat4', 'n_cols_cat3', 'n_cols_cat2', 'n_cols_cat1', 'n_cols_hc',
    'cobblestones_km', 'compacted_gravel_km',
]
_RACE_FIRST_FEATS = [
    'startlist_quality', 'forme_equipe', 'n_races_30d', 'km_30d',
    'leader_played', 'is_team_leader',
]
FEATURES_RACE_LEVEL = _RACE_SUM_FEATS + _RACE_FIRST_FEATS

# ── Leader lookup (2024-2025) ─────────────────────────────────────────────────
_TEAM_NAME_MAP = {
    'Team Visma | Lease a Bike': 'Team Visma',
    'Team Visma | Lease a Bike Development': 'Jumbo-Visma Development Team',
}
_LOOKUP_CACHE = BASE_DIR / 'leader_played_lookup.parquet'


def _build_leader_played_lookup():
    if _LOOKUP_CACHE.exists():
        df = pd.read_parquet(_LOOKUP_CACHE)
        return {
            (row.equipe, int(row.year), row.course, row.stage_num): int(row.is_leader)
            for row in df.itertuples()
        }
    tl_path = BASE_DIR / 'team_leaders.csv'
    if not tl_path.exists():
        return {}
    tl = pd.read_csv(tl_path)[['team', 'year', 'leader_1_rider']]
    rows = []
    for f in sorted(RIDER_DIR.glob('*.csv')):
        try:
            df = pd.read_csv(f, usecols=['year', 'equipe', 'course', 'stage_num', 'selected'], low_memory=False)
            df = df[df['year'].isin([2024, 2025]) & (df['selected'] == 1)].copy()
            if len(df):
                df['rider'] = f.stem
                rows.append(df)
        except Exception:
            pass
    if not rows:
        return {}
    sel = pd.concat(rows, ignore_index=True)
    sel['team_key'] = sel['equipe'].map(_TEAM_NAME_MAP).fillna(sel['equipe'])
    sel = sel.merge(tl, left_on=['team_key', 'year'], right_on=['team', 'year'], how='left')
    sel['is_leader'] = (sel['rider'] == sel['leader_1_rider']).astype(int)
    lp = sel.groupby(['equipe', 'year', 'course', 'stage_num'])['is_leader'].max().reset_index()
    lp['year'] = lp['year'].astype(int)
    lp.to_parquet(_LOOKUP_CACHE, index=False)
    return {(r.equipe, int(r.year), r.course, r.stage_num): int(r.is_leader) for r in lp.itertuples()}


LEADER_PLAYED_LOOKUP = _build_leader_played_lookup()


# ── Riders index ──────────────────────────────────────────────────────────────
_INDEX_CACHE = BASE_DIR / 'riders_index.parquet'


def _build_riders_index():
    if _INDEX_CACHE.exists():
        return pd.read_parquet(_INDEX_CACHE)
    rows = []
    for f in sorted(RIDER_DIR.glob('*.csv')):
        try:
            df = pd.read_csv(f, usecols=['year', 'equipe', TREATMENT], low_memory=False)
            for (equipe, year), grp in df.groupby(['equipe', 'year']):
                rows.append({
                    'rider':    f.stem,
                    'equipe':   equipe,
                    'year':     int(year),
                    'n_sel':    int((grp[TREATMENT] == 1).sum()),
                    'n_total':  len(grp),
                })
        except Exception:
            pass
    idx = pd.DataFrame(rows)
    idx.to_parquet(_INDEX_CACHE, index=False)
    return idx


_INDEX = None


def _get_index() -> pd.DataFrame:
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_riders_index()
    return _INDEX


def list_all_teams() -> list[str]:
    return sorted(_get_index()['equipe'].unique())


def list_all_riders() -> list[str]:
    return sorted(_get_index()['rider'].unique())


def find_team_riders(equipe, min_selections: int = 10, years=None) -> list[str]:
    idx = _get_index()
    equipe_list = equipe if isinstance(equipe, list) else [equipe]
    sub = idx[idx['equipe'].isin(equipe_list)]
    if years is not None and 'year' in sub.columns:
        sub = sub[sub['year'].between(int(years[0]), int(years[1]))]
    totals = sub.groupby('rider')['n_sel'].sum()
    return sorted(totals[totals >= min_selections].index.tolist())


def rider_teams(rider_name: str) -> list[str]:
    idx = _get_index()
    return sorted(idx[idx['rider'] == rider_name]['equipe'].unique())


def run_team_analysis(
    equipe,
    min_selections: int = 20,
    years=None,
    outcome: str = OUTCOME,
    features=None,
    n_boot: int = N_BOOT,
) -> pd.DataFrame | None:
    """Run DML for all riders in a team. Returns a DataFrame with one row per rider."""
    riders = find_team_riders(equipe, min_selections=min_selections, years=years)
    rows = []
    for rider_name in riders:
        # Load ALL seasons for training
        df = load_rider(rider_name, equipe=equipe, years=None)
        if df is None:
            continue
        prep = prepare_features(df, features=features, outcome=outcome)
        if prep is None:
            continue
        X, T, Y, df_clean, _ = prep
        dml = run_dml(X, T, Y, n_boot=n_boot)
        if dml is None:
            continue
        brut = (
            df_clean[df_clean[TREATMENT] == 1][outcome].mean()
            - df_clean[df_clean[TREATMENT] == 0][outcome].mean()
        )
        rows.append({
            'rider':       rider_name,
            'n_obs':       len(T),
            'n_selected':  int(T.sum()),
            'taux_sel':    round(float(T.mean()), 3),
            'ate_orig':    dml['ate_orig'],
            'ci_low':      dml['ci_low'],
            'ci_high':     dml['ci_high'],
            'significant': dml['significant'],
            'effet_brut':  float(brut),
            'r2_t':        dml['r2_t'],
            'r2_y':        dml['r2_y'],
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values('ate_orig', ascending=False)


def get_team_roster_by_year(equipe) -> pd.DataFrame:
    """Returns a DataFrame (rider × year) with n_sel and n_total for a given team."""
    idx = _get_index()
    equipe_list = equipe if isinstance(equipe, list) else [equipe]
    sub = idx[idx['equipe'].isin(equipe_list)].copy()
    sub = sub.sort_values(['year', 'rider'])
    return sub[['rider', 'equipe', 'year', 'n_sel', 'n_total']]


# ── GPX index & profile loading ───────────────────────────────────────────────
GPX_DIR_ELE2  = BASE_DIR / 'data' / 'gpx_files_ele2'   # {day}_{month}_{year}_{Name}.gpx  (2024-2025)
GPX_DIR_2     = BASE_DIR / 'data' / 'gpx_files_2'       # {year} {race_name}.gpx           (2017-2025)
_GPX_INDEX_CACHE = BASE_DIR / 'gpx_index.parquet'
_GPX_INDEX = None


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _build_gpx_index() -> pd.DataFrame:
    rows = []
    # ── gpx_files_ele2 : {day}_{month}_{year}_{Name}.gpx ─────────────────────
    if GPX_DIR_ELE2.exists():
        for f in sorted(GPX_DIR_ELE2.glob('*.gpx')):
            parts = f.stem.split('_', 3)
            if len(parts) < 4:
                continue
            try:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            race_name = parts[3]
            rows.append({
                'day': day, 'month': month, 'year': year,
                'race_name': race_name,
                'name_norm': _norm(race_name),
                'has_day': True,
                'filepath': str(f),
            })
    # ── gpx_files_2 : {year} {race_name}.gpx ─────────────────────────────────
    if GPX_DIR_2.exists():
        for f in sorted(GPX_DIR_2.glob('*.gpx')):
            stem = f.stem  # "2022 Tour de France Stage 17"
            parts = stem.split(' ', 1)
            if len(parts) < 2:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            race_name = parts[1]
            rows.append({
                'day': -1, 'month': -1, 'year': year,
                'race_name': race_name,
                'name_norm': _norm(race_name),
                'has_day': False,
                'filepath': str(f),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(_GPX_INDEX_CACHE, index=False)
    return df


def _get_gpx_index() -> pd.DataFrame:
    global _GPX_INDEX
    if _GPX_INDEX is None:
        if _GPX_INDEX_CACHE.exists():
            _GPX_INDEX = pd.read_parquet(_GPX_INDEX_CACHE)
        elif GPX_DIR_ELE2.exists() or GPX_DIR_2.exists():
            _GPX_INDEX = _build_gpx_index()
        else:
            _GPX_INDEX = pd.DataFrame()
    return _GPX_INDEX


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


_SECONDARY_RACE_WORDS = {'femmes', 'women', 'junior', 'juniors', 'espoirs', 'u23',
                         'amateur', 'neopro', 'under23', 'dames'}


def _score_name(course_norm: str, name_norm: str, stage_str: str) -> int:
    best = 0
    for length in range(min(len(course_norm), len(name_norm), 12), 2, -1):
        for start in range(len(course_norm) - length + 1):
            if course_norm[start:start + length] in name_norm:
                best = length
                break
        if best:
            break
    # Stage bonus only when base name similarity is already meaningful (≥5 chars)
    # Prevents "Tour de France Stage 7" from beating "Tour Auvergne" just by stage number
    if stage_str and stage_str in name_norm and best >= 5:
        best += 5
    # Penalise women/junior/espoirs races when not explicitly requested
    for word in _SECONDARY_RACE_WORDS:
        if word in name_norm and word not in course_norm:
            best -= 4
            break
    # Prefer shorter names (less extra content beyond the match)
    best -= len(name_norm) // 20
    return best


# Known slug renames: course slug in rider data → name used in GPX files
_COURSE_ALIASES: dict[str, str] = {
    'tour-auvergne-rhone-alpes':  'criterium-du-dauphine',
    'heist-op-den-berg':          'heistse-pijl',
    'world-championship':         'uci-road-world-championships',
    'world-championship-itt':     'uci-road-world-championships-itt',
}


def _raw_match(course_norm: str, name_norm: str) -> int:
    """Longest common substring length (no bonuses/penalties)."""
    for length in range(min(len(course_norm), len(name_norm), 15), 2, -1):
        for start in range(len(course_norm) - length + 1):
            if course_norm[start:start + length] in name_norm:
                return length
    return 0


def find_gpx_path(course: str, date_val, stage_num=None) -> str | None:
    """Return filepath of the best-matching GPX for a given race row."""
    idx = _get_gpx_index()
    if idx is None or len(idx) == 0:
        return None
    try:
        d = pd.to_datetime(date_val)
        day, month, year = d.day, d.month, d.year
    except Exception:
        return None

    # Apply known aliases (renamed races)
    course = _COURSE_ALIASES.get(course, course)
    course_norm = _norm(course)
    stage_str = _norm(f'stage{int(stage_num)}') if stage_num and pd.notna(stage_num) else ''

    def _is_good(best_row: pd.Series) -> bool:
        """Score must be > 4 AND raw substring must cover ≥35% of the course slug."""
        if best_row['score'] <= 4:
            return False
        raw = _raw_match(course_norm, best_row['name_norm'])
        return len(course_norm) > 0 and raw / len(course_norm) >= 0.35

    # 1. Try exact-date candidates from gpx_files_ele2
    cands_exact = idx[
        idx['has_day'] & (idx['day'] == day) & (idx['month'] == month) & (idx['year'] == year)
    ].copy()
    if len(cands_exact) == 1:
        # Single candidate on exact date — still validate ratio
        row = cands_exact.iloc[0]
        row['score'] = _score_name(course_norm, row['name_norm'], stage_str)
        if _is_good(row):
            return row['filepath']
    elif len(cands_exact) > 1:
        cands_exact['score'] = cands_exact.apply(
            lambda r: _score_name(course_norm, r['name_norm'], stage_str), axis=1
        )
        best = cands_exact.nlargest(1, 'score').iloc[0]
        if _is_good(best):
            return best['filepath']

    # 2. Fall back to year-only candidates from gpx_files_2
    cands_yr = idx[~idx['has_day'] & (idx['year'] == year)].copy()
    if len(cands_yr) == 0:
        return None
    cands_yr['score'] = cands_yr.apply(
        lambda r: _score_name(course_norm, r['name_norm'], stage_str), axis=1
    )
    # Départage : pénalité suffixe → raw_match non-plafonné → longueur nom
    _query_has = {k: k in course_norm for k in ('itt', 'ttt', 'relay')}
    def _suffix_penalty(n):
        return sum(1 for k in ('itt', 'ttt', 'relay') if k in n and not _query_has[k])
    def _raw_uncapped(n):
        for length in range(min(len(course_norm), len(n)), 2, -1):
            for start in range(len(course_norm) - length + 1):
                if course_norm[start:start + length] in n:
                    return length
        return 0
    cands_yr['raw']     = cands_yr['name_norm'].apply(_raw_uncapped)
    cands_yr['penalty'] = cands_yr['name_norm'].apply(_suffix_penalty)
    cands_yr['name_len']= cands_yr['name_norm'].str.len()
    best = cands_yr.sort_values(['score', 'penalty', 'raw', 'name_len'],
                                 ascending=[False, True, False, True]).iloc[0]
    return best['filepath'] if _is_good(best) else None


_GPX_PROFILES: pd.DataFrame | None = None

def _get_gpx_profiles() -> pd.DataFrame | None:
    """Load precomputed profiles parquet once, cache in module-level variable."""
    global _GPX_PROFILES
    if _GPX_PROFILES is not None:
        return _GPX_PROFILES
    parquet_path = BASE_DIR / 'gpx_profiles.parquet'
    if parquet_path.exists():
        _GPX_PROFILES = pd.read_parquet(parquet_path)
    return _GPX_PROFILES


def load_gpx_profile(course: str, date_val, stage_num=None,
                     max_points: int = 600) -> pd.DataFrame | None:
    """Return a DataFrame with distance_km and elevation for the given race.

    Uses the precomputed gpx_profiles.parquet if available (fast, no GPX files
    needed). Falls back to live GPX parsing if the parquet is missing.
    """
    filepath = find_gpx_path(course, date_val, stage_num)
    if filepath is None:
        return None

    gpx_key = Path(filepath).stem

    # Fast path: precomputed parquet
    profiles = _get_gpx_profiles()
    if profiles is not None:
        sub = profiles[profiles['gpx_key'] == gpx_key][['distance_km', 'elevation']]
        return sub.reset_index(drop=True) if len(sub) > 0 else None

    # Fallback: live parsing from raw GPX file
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = {'g': 'http://www.topografix.com/GPX/1/1'}
        pts = root.findall('.//g:trkpt', ns)
        if not pts:
            return None
        lats = [float(p.get('lat')) for p in pts]
        lons = [float(p.get('lon')) for p in pts]
        ele_nodes = [p.find('g:ele', ns) for p in pts]
        eles = [float(e.text) if e is not None else None for e in ele_nodes]

        dists = [0.0]
        for i in range(1, len(lats)):
            dists.append(dists[-1] + _haversine_km(lats[i-1], lons[i-1], lats[i], lons[i]))

        df = pd.DataFrame({'distance_km': dists, 'elevation': eles}).dropna()
        if len(df) > max_points:
            step = max(1, len(df) // max_points)
            df = df.iloc[::step].reset_index(drop=True)
        return df
    except Exception:
        return None


def get_team_year_leaders() -> dict:
    """Returns {(team, year): leader_1_rider} from team_leaders.csv."""
    path = BASE_DIR / 'team_leaders.csv'
    if not path.exists():
        return {}
    tl = pd.read_csv(path)[['team', 'year', 'leader_1_rider']]
    return {(row.team, int(row.year)): row.leader_1_rider for row in tl.itertuples()}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_rider(rider_name: str, equipe=None, years=None) -> pd.DataFrame | None:
    path = RIDER_DIR / f'{rider_name}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df['date']      = pd.to_datetime(df['date'], errors='coerce')
    df['stage_num'] = pd.to_numeric(df['stage_num'], errors='coerce')
    df['rider']     = rider_name
    if equipe is not None:
        equipe_list = equipe if isinstance(equipe, list) else [equipe]
        df = df[df['equipe'].isin(equipe_list)]
    if years is not None:
        df = df[df['year'].between(years[0], years[1])]
    if 'leader_played' in df.columns and LEADER_PLAYED_LOOKUP:
        mask = df['leader_played'].isna()
        if mask.any():
            keys = list(zip(
                df.loc[mask, 'equipe'],
                df.loc[mask, 'year'].astype(int),
                df.loc[mask, 'course'],
                df.loc[mask, 'stage_num'],
            ))
            df.loc[mask, 'leader_played'] = [LEADER_PLAYED_LOOKUP.get(k, 0) for k in keys]
    # Fix CLM misclassification: physical clustering can't distinguish flat ITTs
    # from flat road stages — use won_how / type / course name as ground truth
    if 'stage_cluster_label' in df.columns:
        is_clm = (
            (df.get('won_how', pd.Series(dtype=str)) == 'Time trial')
            | (df.get('type', pd.Series(dtype=str)) == 'prologue')
            | df['course'].str.contains(
                r'(?:^|-)(?:clm|chrono|prologue|itt)(?:-|$)',
                case=False, regex=True, na=False,
            )
        )
        df.loc[is_clm, 'stage_cluster']       = 0.0
        df.loc[is_clm, 'stage_cluster_label'] = '⏱️  CLM'
        _CLUSTER_LABELS_EN = {
            '🟢  Plat/Sprint':    '🟢  Flat/Sprint',
            '⛰️  Moy. montagne':  '⛰️  Medium mountain',
            '🏔️  Haute montagne': '🏔️  High mountain',
            '⏱️  CLM':            '⏱️  TT',
        }
        df['stage_cluster_label'] = df['stage_cluster_label'].replace(_CLUSTER_LABELS_EN)

    # Jointure startlist_quality de l'année précédente
    sq_lookup = _get_sq_lookup()
    if sq_lookup is not None and 'course' in df.columns and 'year' in df.columns:
        df['year'] = df['year'].astype(int)
        prev = sq_lookup.rename(columns={'startlist_quality': 'startlist_quality_prev', 'year': '_prev_year'})
        prev['_join_year'] = prev['_prev_year'] + 1
        df = df.merge(
            prev[['course', '_join_year', 'startlist_quality_prev']],
            left_on=['course', 'year'], right_on=['course', '_join_year'], how='left'
        ).drop(columns=['_join_year'])
        # Fallback : moyenne historique de la course quand N-1 absent (2018, première édition…)
        hist_mean = sq_lookup.groupby('course')['startlist_quality'].median().rename('_sq_hist')
        df = df.join(hist_mean, on='course')
        mask = df['startlist_quality_prev'].isna()
        df.loc[mask, 'startlist_quality_prev'] = df.loc[mask, '_sq_hist']
        df.drop(columns=['_sq_hist'], inplace=True)

    # Compute normalised outcome : pts équipe / pts 1ère place (barème UCI fixe)
    # → 100% = performance de la 1ère place, peut dépasser 100% si plusieurs coureurs scorent
    if 'pts_uci_equipe_stage' in df.columns:
        pts1 = _get_pts1_lookup()
        if pts1 is not None:
            rank1_pts = df['course'].map(pts1)
        else:
            rank1_pts = df.groupby('course')['pts_uci_equipe_stage'].transform('max')
        df['pts_uci_pct_max'] = (
            df['pts_uci_equipe_stage']
            / rank1_pts.replace(0, float('nan')) * 100
        ).fillna(0).round(2)
    # Points coéquipiers = total équipe - points du coureur lui-même
    if 'pts_uci_equipe_stage' in df.columns and 'pts_uci' in df.columns:
        df['pts_uci_teammates'] = (
            pd.to_numeric(df['pts_uci_equipe_stage'], errors='coerce').fillna(0)
            - pd.to_numeric(df['pts_uci'], errors='coerce').fillna(0)
        ).clip(lower=0)
    return df if len(df) > 0 else None


# ── Feature preparation ───────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame, features=None, outcome=OUTCOME, essential_features=None):
    if features is None:
        features = ALL_FEATURES
    df = df.copy()
    feats = [f for f in features if f in df.columns]
    df[outcome]   = pd.to_numeric(df[outcome],   errors='coerce')
    df[TREATMENT] = pd.to_numeric(df[TREATMENT], errors='coerce')
    if essential_features is None:
        essential_features = [f for f in FEATURES_RACE if f in df.columns and f != 'startlist_quality']
    df_clean = df.dropna(subset=essential_features + [outcome, TREATMENT]).reset_index(drop=True)
    if len(df_clean) < 20:
        return None
    for col in feats:
        if col in df_clean.columns and df_clean[col].isna().any():
            med = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(med if not pd.isna(med) else 0)
    X = df_clean[feats].values.astype(float)
    T = df_clean[TREATMENT].values.astype(float)
    Y = np.log1p(np.clip(df_clean[outcome].values.astype(float), 0, None))
    return X, T, Y, df_clean, feats


# ── DML ───────────────────────────────────────────────────────────────────────
def run_dml(X, T, Y, n_boot=N_BOOT, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    if T.sum() < 5 or (T == 0).sum() < 5:
        return None
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    T_resid = np.zeros(len(T))
    Y_resid = np.zeros(len(Y))
    r2_t_folds, r2_y_folds = [], []
    for train_idx, val_idx in kf.split(X):
        m_t = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=random_state)
        m_t.fit(X[train_idx], T[train_idx])
        T_pred = m_t.predict(X[val_idx])
        T_resid[val_idx] = T[val_idx] - T_pred
        r2_t_folds.append(r2_score(T[val_idx], T_pred))
        m_y = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=random_state)
        m_y.fit(X[train_idx], Y[train_idx])
        Y_pred = m_y.predict(X[val_idx])
        Y_resid[val_idx] = Y[val_idx] - Y_pred
        r2_y_folds.append(r2_score(Y[val_idx], Y_pred))
    ate = np.cov(Y_resid, T_resid)[0, 1] / (np.var(T_resid) + 1e-10)
    rng = np.random.default_rng(random_state)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(T_resid), len(T_resid), replace=True)
        cov_b = np.cov(Y_resid[idx], T_resid[idx])
        boots.append(cov_b[0, 1] / (np.var(T_resid[idx]) + 1e-10))
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return {
        'ate_log':   ate,
        'ate_orig':  float(np.expm1(ate)),
        'ci_low':    float(np.expm1(ci_low)),
        'ci_high':   float(np.expm1(ci_high)),
        'significant': bool(ci_low > 0 or ci_high < 0),
        'r2_t':      float(np.mean(r2_t_folds)),
        'r2_y':      float(np.mean(r2_y_folds)),
        'T_resid':   T_resid,
        'Y_resid':   Y_resid,
    }


# ── Causal Forest ─────────────────────────────────────────────────────────────
def run_causal_forest(X, T, Y, n_trees=N_TREES, random_state=RANDOM_STATE):
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=random_state),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=random_state),
        n_estimators=n_trees,
        min_samples_leaf=10,
        max_depth=5,
        random_state=random_state,
        cv=N_FOLDS,
    )
    cf.fit(Y, T, X=X)
    return cf, cf.effect(X)


# ── High-level entry point ────────────────────────────────────────────────────
def run_analysis(
    rider_name: str,
    equipe,
    years=None,
    outcome: str = OUTCOME,
    features=None,
    n_boot: int = N_BOOT,
    n_trees: int = N_TREES,
    run_cf: bool = True,
) -> dict | None:
    # Load ALL seasons so nuisance models train on the full history
    df = load_rider(rider_name, equipe=equipe, years=None)
    if df is None:
        return None
    prep = prepare_features(df, features=features, outcome=outcome)
    if prep is None:
        return None
    X, T, Y, df_clean, feats = prep
    dml = run_dml(X, T, Y, n_boot=n_boot)
    if run_cf:
        cf_model, cate = run_causal_forest(X, T, Y, n_trees=n_trees)
        df_clean = df_clean.copy()
        df_clean['cate'] = cate
    else:
        cf_model = cate = None

    # Filter df_clean to the requested years for display only
    if years is not None and 'year' in df_clean.columns:
        df_clean = df_clean[df_clean['year'].between(years[0], years[1])]

    result = {
        'dml':        dml,
        'df_clean':   df_clean,
        'df_raw':     df,
        'features':   feats,
        'n_obs':      len(T),
        'n_selected': int(T.sum()),
        'outcome':    outcome,
        'rider':      rider_name,
        'equipe':     equipe,
        'years':      years,
    }
    if run_cf:
        result['cf_model'] = cf_model
        result['cate']     = cate
        result['X']        = X
    return result


# ── Race-level model (GC / KOM / Sprint classements annexes) ──────────────────
def load_rider_race_level(
    rider_name: str, equipe=None, years=None, outcome: str = 'pts_uci_equipe_gc',
) -> pd.DataFrame | None:
    """Charge riders_gc/{rider}.csv (une ligne par course) et joint les features GPX agrégées."""
    gc_path = GC_DIR / f'{rider_name}.csv'
    if not gc_path.exists():
        return None
    gc = pd.read_csv(gc_path, low_memory=False)
    gc = gc[gc['type'] == 'gc'].copy()   # une ligne par course (les 3 outcomes sont déjà là)
    # Keep only multi-stage races (classification 2.X) — one-day races have no meaningful GC
    if 'classification' in gc.columns:
        gc = gc[gc['classification'].str.startswith('2', na=False)]

    if equipe is not None:
        equipe_list = equipe if isinstance(equipe, list) else [equipe]
        gc = gc[gc['equipe'].isin(equipe_list)]
    if years is not None:
        gc = gc[gc['year'].between(years[0], years[1])]
    if outcome not in gc.columns or len(gc) == 0:
        return None
    # NaN pour pts KOM/sprint = course sans ce classement → 0 pts, observation valide
    gc[outcome] = pd.to_numeric(gc[outcome], errors='coerce').fillna(0)

    # Joindre les features GPX agrégées depuis rider_data/
    stage_df = load_rider(rider_name, equipe=equipe, years=years)
    if stage_df is not None:
        agg_spec: dict = {}
        for f in _RACE_SUM_FEATS:
            if f in stage_df.columns:
                agg_spec[f] = (f, 'sum')
        for f in _RACE_FIRST_FEATS:
            if f in stage_df.columns:
                agg_spec[f] = (f, 'first')
        if agg_spec:
            feat_df = stage_df.groupby(['equipe', 'year', 'course']).agg(**agg_spec).reset_index()
            gc = gc.merge(feat_df, on=['equipe', 'year', 'course'], how='left')

    return gc if len(gc) > 0 else None


def run_analysis_race_level(
    rider_name: str,
    equipe,
    years=None,
    outcome: str = 'pts_uci_equipe_gc',
    n_boot: int = N_BOOT,
    n_trees: int = N_TREES,
    run_cf: bool = True,
) -> dict | None:
    """DML + Causal Forest au niveau course pour les classements GC/KOM/sprint."""
    # Load ALL seasons for training
    df = load_rider_race_level(rider_name, equipe=equipe, years=None, outcome=outcome)
    if df is None:
        return None
    # essential_features=[] : on n'exclut pas les lignes sans GPX, on impute par la médiane
    prep = prepare_features(df, features=FEATURES_RACE_LEVEL, outcome=outcome, essential_features=[])
    if prep is None:
        return None
    X, T, Y, df_clean, feats = prep
    dml = run_dml(X, T, Y, n_boot=n_boot)
    if run_cf:
        cf_model, cate = run_causal_forest(X, T, Y, n_trees=n_trees)
        df_clean = df_clean.copy()
        df_clean['cate'] = cate
    else:
        cf_model = cate = None

    # Filter df_clean to the requested years for display only
    if years is not None and 'year' in df_clean.columns:
        df_clean = df_clean[df_clean['year'].between(years[0], years[1])]

    result = {
        'dml':        dml,
        'df_clean':   df_clean,
        'df_raw':     df,
        'features':   feats,
        'n_obs':      len(T),
        'n_selected': int(T.sum()),
        'outcome':    outcome,
        'rider':      rider_name,
        'equipe':     equipe,
        'years':      years,
    }
    if run_cf:
        result['cf_model'] = cf_model
        result['cate']     = cate
        result['X']        = X
    return result


def run_team_analysis_race_level(
    equipe,
    min_selections: int = 10,
    years=None,
    outcome: str = 'pts_uci_equipe_gc',
    n_boot: int = N_BOOT,
) -> pd.DataFrame | None:
    """DML race-level pour tous les coureurs d'une équipe."""
    riders = find_team_riders(equipe, min_selections=min_selections, years=years)
    rows = []
    for rider_name in riders:
        df = load_rider_race_level(rider_name, equipe=equipe, years=None, outcome=outcome)
        if df is None:
            continue
        prep = prepare_features(df, features=FEATURES_RACE_LEVEL, outcome=outcome, essential_features=[])
        if prep is None:
            continue
        X, T, Y, df_clean, _ = prep
        dml = run_dml(X, T, Y, n_boot=n_boot)
        if dml is None:
            continue
        rows.append({
            'rider':       rider_name,
            'n_obs':       len(T),
            'n_selected':  int(T.sum()),
            'taux_sel':    round(float(T.mean()), 3),
            'ate_orig':    dml['ate_orig'],
            'ci_low':      dml['ci_low'],
            'ci_high':     dml['ci_high'],
            'significant': dml['significant'],
            'r2_t':        dml['r2_t'],
            'r2_y':        dml['r2_y'],
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values('ate_orig', ascending=False)
