import sys
import unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import causal_model as cm

st.set_page_config(
    page_title="Causal Cycling",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stToolbar"]       {visibility: hidden;}
[data-testid="stDecoration"]    {display: none;}
[data-testid="stStatusWidget"]  {visibility: hidden;}
#MainMenu                       {visibility: hidden;}
footer                          {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🚴 Causal Cycling — Rider Evaluation")

CLASSIFICATION_RANK = {
    '2.UWT': 1, '1.UWT': 2, 'WC': 3, 'Olympics': 4,
    '2.Pro': 5, '1.Pro': 6, '2.HC': 7, '1.HC': 8,
    '2.1': 9, '1.1': 10, '1.2': 11,
    'NC': 12, 'CC': 13, '1.Ncup': 14,
}

def fmt_classification(c):
    rank = CLASSIFICATION_RANK.get(c)
    return f"{c} ({rank})" if rank else c

# ── Cached helpers ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building rider index…")
def get_all_riders():
    return cm.list_all_riders()

@st.cache_data(show_spinner=False)
def load_gpx_profile(course, date_val, stage_num):
    return cm.load_gpx_profile(course, date_val, stage_num)

@st.cache_data(show_spinner=False)
def get_all_teams():
    return cm.list_all_teams()

@st.cache_data(show_spinner=False)
def get_rider_teams(rider_name):
    return cm.rider_teams(rider_name)

@st.cache_data(show_spinner=False)
def get_team_riders(equipe_tuple, min_sel, years_tuple=None):
    years = list(years_tuple) if years_tuple else None
    return cm.find_team_riders(list(equipe_tuple), min_selections=min_sel, years=years)

@st.cache_data(show_spinner=False)
def get_leaders():
    return cm.get_team_year_leaders()

@st.cache_data(show_spinner=False)
def cached_analysis(rider, equipe_tuple, years, outcome, n_boot, n_trees, run_cf):
    return cm.run_analysis(
        rider, list(equipe_tuple),
        years=years, outcome=outcome,
        n_boot=n_boot, n_trees=n_trees, run_cf=run_cf,
    )

@st.cache_data(show_spinner=False)
def cached_analysis_race(rider, equipe_tuple, years, outcome, n_boot, n_trees, run_cf):
    return cm.run_analysis_race_level(
        rider, list(equipe_tuple),
        years=years, outcome=outcome,
        n_boot=n_boot, n_trees=n_trees, run_cf=run_cf,
    )

@st.cache_data(show_spinner=False)
def cached_team_analysis_race(equipe_tuple, years, outcome, n_boot):
    return cm.run_team_analysis_race_level(
        list(equipe_tuple), years=years, outcome=outcome, n_boot=n_boot,
    )

@st.cache_data(show_spinner=False)
def cached_load_raw(rider, equipe_tuple, years):
    return cm.load_rider(rider, equipe=list(equipe_tuple) if equipe_tuple else None, years=years)

@st.cache_data(show_spinner=False)
def cached_gc_pts(rider, equipe_tuple, years):
    """Load riders_gc pts (GC/KOM/Sprint team pts, one row per race)."""
    df = cm.load_rider_race_level(
        rider,
        equipe=list(equipe_tuple) if equipe_tuple else None,
        years=list(years) if years else None,
        outcome='pts_uci_equipe_gc',
    )
    if df is None:
        return None
    keep = ['course', 'year', 'pts_uci', 'pts_uci_equipe_stage',
            'pts_uci_equipe_gc', 'pts_uci_equipe_kom', 'pts_uci_equipe_points']
    return df[[c for c in keep if c in df.columns]]

@st.cache_data(show_spinner=False)
def cached_roster(equipe_tuple):
    return cm.get_team_roster_by_year(list(equipe_tuple))

def _race_equipe_kw(race_row) -> str:
    """Return a short keyword for the rider's team at a specific race (from equipe field)."""
    eq = str(race_row.get('equipe', '')).lower()
    # Take the first meaningful token (e.g. 'jumbo' from 'jumbo-visma', 'cofidis' from 'cofidis, solutions')
    token = eq.replace('-', ' ').replace(',', ' ').split()[0] if eq.strip() else ''
    return token


def _inject_team_uci_pts(df_res: pd.DataFrame, race_row) -> pd.DataFrame:
    """Replace Team UCI pts (which are PCS pts in the parquet) with the correct UCI team pts
    from pts_uci_equipe_stage for the analyzed rider's team. Other teams keep their parquet sum."""
    correct_pts = race_row.get('pts_uci_equipe_stage', None)
    kw = _race_equipe_kw(race_row)
    if correct_pts is not None and kw:
        mask = df_res['Team'].str.lower().str.contains(kw, regex=False)
        df_res = df_res.copy()
        df_res.loc[mask, 'Team UCI pts'] = correct_pts
    return df_res


@st.cache_data(show_spinner=False)
def load_race_results(course: str, year: int, stage_num):
    """Load race results for a specific race from precomputed stage_results.parquet.
    The full DB is loaded, filtered, then freed — only the small result is cached."""
    path = Path(cm.BASE_DIR) / 'stage_results.parquet'
    if not path.exists():
        return None
    db = pd.read_parquet(path)
    sn = '' if (stage_num is None or (isinstance(stage_num, float) and pd.isna(stage_num))) else str(int(float(stage_num)))
    mask = (db['course'] == course) & (db['year'] == str(int(year))) & (db['stage_num'] == sn)
    df = db[mask][['Rank', 'Rider', 'Team', 'UCI pts']].copy()
    del db  # free the full DB immediately
    if len(df) == 0:
        return None
    df = df.drop_duplicates(subset=['Rank', 'Rider', 'Team'])
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    # PCS data appends team name to rider name — strip it
    def _clean(row):
        rider, team = str(row['Rider']).strip(), str(row['Team']).strip()
        if rider.endswith(' ' + team):
            return rider[:-len(team) - 1].strip()
        if rider.endswith(team):
            return rider[:-len(team)].strip()
        return rider
    df['Rider'] = df.apply(_clean, axis=1)
    df = df.sort_values('Rank').reset_index(drop=True)
    # Add team total UCI pts column
    df['Team UCI pts'] = df.groupby('Team')['UCI pts'].transform('sum')
    return df

@st.cache_data(show_spinner=False)
def load_gc_classification(course: str, year: int):
    """Load GC classification from gc_results.parquet (one row per rider, type='gc')."""
    path = Path(cm.BASE_DIR) / 'gc_results.parquet'
    if not path.exists():
        return None
    db = pd.read_parquet(path)
    mask = (db['course'] == course) & (db['year'] == str(int(year)))
    df = db[mask][['rang', 'rider', 'equipe', 'pts_uci']].copy()
    del db
    if len(df) == 0:
        return None
    df = df.sort_values('rang').reset_index(drop=True)
    df['rider'] = df['rider'].apply(fmt_rider)
    df.columns = ['GC Rank', 'Rider', 'Team', 'UCI pts (GC)']
    df['Team UCI pts (GC)'] = df.groupby('Team')['UCI pts (GC)'].transform('sum')
    return df


@st.cache_data(show_spinner=False)
def load_all_race_stages(course: str, year: int):
    """Load all stage + GC results for a stage race (course + year), grouped by stage_num."""
    path = Path(cm.BASE_DIR) / 'stage_results.parquet'
    if not path.exists():
        return None
    db = pd.read_parquet(path)
    mask = (db['course'] == course) & (db['year'] == str(int(year)))
    df = db[mask][['Rank', 'Rider', 'Team', 'UCI pts', 'stage_num']].copy()
    del db
    if len(df) == 0:
        return None
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    def _clean(row):
        rider, team = str(row['Rider']).strip(), str(row['Team']).strip()
        if rider.endswith(' ' + team):
            return rider[:-len(team) - 1].strip()
        if rider.endswith(team):
            return rider[:-len(team)].strip()
        return rider
    df['Rider'] = df.apply(_clean, axis=1)
    df['UCI pts'] = pd.to_numeric(df['UCI pts'], errors='coerce').fillna(0)
    return df


def _stage_sort_key(s: str) -> tuple:
    """Sort key: numeric stages first (by number), then GC/empty last."""
    try:
        return (0, int(s))
    except (ValueError, TypeError):
        return (1, s or 'zzz')


def fmt_rider(name: str) -> str:
    return unicodedata.normalize('NFC', name.replace('_', ' ')).title()

# ── Sidebar ───────────────────────────────────────────────────────────────────
all_riders = get_all_riders()
all_teams  = get_all_teams()
leaders    = get_leaders()

N_BOOT  = 200
N_TREES = 300

with st.sidebar:
    st.header("Configuration")
    mode = st.radio("Mode", ["Single analysis", "Comparison"], horizontal=True)

    years = st.slider("Years", 2018, 2025, (2018, 2025))

    def rider_selector(key_prefix: str, subtitle: str):
        st.subheader(subtitle)

        # Derive team list from session state BEFORE rendering rider dropdown.
        # Must check both the manual multiselect (_teams) and the canonical
        # selectbox (_canon), because the canonical path doesn't write to _teams.
        teams = st.session_state.get(f"{key_prefix}_teams", [])
        if not teams:
            canon_prev = st.session_state.get(f"{key_prefix}_canon")
            if canon_prev:
                teams = [t for t in cm.expand_team(canon_prev) if t in all_teams]
        rider_list = (
            get_team_riders(tuple(sorted(teams)), 1, tuple(years)) if teams else all_riders
        )
        rider = st.selectbox(
            "Rider (optional)",
            [None] + rider_list,
            format_func=lambda x: "— All riders —" if x is None else fmt_rider(x),
            key=f"{key_prefix}_rider",
        )

        # Teams: filtered on the rider's teams if a rider is selected
        if rider:
            rider_equipes = get_rider_teams(rider)
            teams = st.multiselect(
                "Team(s)",
                rider_equipes,
                default=rider_equipes,
                key=f"{key_prefix}_teams",
                placeholder="All of the rider's teams",
            )
            if not teams:
                teams = rider_equipes
        else:
            canon_names = list(cm.TEAM_GROUPS.keys())
            canon_choice = st.selectbox(
                "Team (canonical name)",
                [None] + canon_names,
                format_func=lambda x: "— Search manually —" if x is None else x,
                key=f"{key_prefix}_canon",
            )
            if canon_choice:
                teams = [t for t in cm.expand_team(canon_choice) if t in all_teams]
                st.caption(
                    f"{', '.join(teams)}"
                    f" · {len(get_team_riders(tuple(sorted(teams)), 1, tuple(years)))} riders"
                )
            else:
                teams = st.multiselect(
                    "Team(s)", all_teams,
                    default=[],
                    key=f"{key_prefix}_teams",
                    placeholder="Search for a team…",
                )
                if teams:
                    st.caption(
                        f"{len(get_team_riders(tuple(sorted(teams)), 1, tuple(years)))} riders in the team"
                    )

        return rider, teams

    rider1, teams1 = rider_selector("c1", "Config 1" if mode == "Comparison" else "Rider")

    if mode == "Comparison":
        st.divider()
        rider2, teams2 = rider_selector("c2", "Config 2")
    else:
        rider2, teams2 = None, None

    st.divider()

    niveau  = st.radio("Analysis level", ["By stage", "By race (GC/KOM/Sprint)"], index=0)
    _race_level = (niveau == "By race (GC/KOM/Sprint)")
    _outcomes = cm.AVAILABLE_OUTCOMES_RACE if _race_level else cm.AVAILABLE_OUTCOMES
    _outcome_labels = {
        'pts_uci_equipe_stage': 'Team pts (stage)',
        'pts_uci_pct_max':      "Team pts (% of winner's pts)",
        'pts_uci':              "Rider's own pts",
        'pts_uci_teammates':    'Teammates pts (team − rider)',
        'pts_uci_equipe_gc':    'Team pts GC',
        'pts_uci_equipe_kom':   'Team pts KOM',
        'pts_uci_equipe_points':'Team pts Sprint',
    }
    outcome = st.selectbox(
        "Outcome (Y)", _outcomes,
        format_func=lambda o: _outcome_labels.get(o, o),
    )

    n_boot  = N_BOOT
    n_trees = N_TREES

    with st.expander("Advanced options"):
        run_cf = st.checkbox("Causal Forest", value=True)

    st.divider()
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# ── Minimal guard ────────────────────────────────────────────────────────────
if not teams1 and not rider1:
    st.info("👈 Select a team or rider in the sidebar.")
    st.stop()

# ── Helper: enrich df_clean with leader name ──────────────────────────────────
def enrich_with_leader(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['leader_name'] = df.apply(
        lambda r: (
            fmt_rider(leaders.get((r['equipe'], int(r['year'])), '—'))
            if r.get('leader_played', 0) == 1
            else '—'
        ) if 'equipe' in df.columns else '—',
        axis=1,
    )
    return df

@st.cache_data(show_spinner=False)
def cached_team_analysis(equipe_tuple, years, outcome, n_boot):
    return cm.run_team_analysis(
        list(equipe_tuple), years=years, outcome=outcome, n_boot=n_boot,
    )

def make_label(rider, teams):
    base = fmt_rider(rider) if rider else "Whole team"
    suffix = ", ".join(t.split("|")[0].strip() for t in teams[:2])
    if len(teams) > 2:
        suffix += "…"
    return f"{base} — {suffix}"

# ── Tabs always visible ────────────────────────────────────────────────────
team_mode1 = (rider1 is None)
label1 = make_label(rider1, teams1)
team_mode2 = (rider2 is None) if mode == "Comparison" else False
label2 = make_label(rider2, teams2) if mode == "Comparison" and teams2 else None

tab_stats, tab_dml, tab_cf, tab_rank = st.tabs(["📈 Descriptive stats", "📊 DML — ATE", "🌲 Causal Forest — CATE", "🏆 WorldTour Rankings"])

# ── Persist the button state via session_state ───────────────────────────────
# run_btn is True only on the frame where button is clicked.
# We store the run params so interacting with CF controls doesn't reset the tab.
if run_btn:
    st.session_state['run_params'] = dict(
        rider1=rider1, teams1=tuple(sorted(teams1)), years=years,
        outcome=outcome, n_boot=n_boot, n_trees=n_trees, run_cf=run_cf,
        rider2=rider2, teams2=tuple(sorted(teams2)) if teams2 else (),
        team_mode1=team_mode1, team_mode2=team_mode2, mode=mode,
        label1=label1, label2=label2,
        race_level=_race_level,
    )

p = st.session_state.get('run_params')
_ran = p is not None

# ── Run analyses (cached — nothing recomputes if parameters are identical) ────
res1 = team_res1 = res2 = team_res2 = None
if _ran:
    _t1_mode = p['team_mode1']
    _r1, _teams1, _years, _out = p['rider1'], p['teams1'], p['years'], p['outcome']
    _nboot, _ntrees, _rcf = p['n_boot'], p['n_trees'], p['run_cf']
    _label1, _label2 = p['label1'], p['label2']
    _race_level = p.get('race_level', False)

    if _t1_mode:
        with st.spinner(f"Team analysis: {_label1}…"):
            if _race_level:
                team_res1 = cached_team_analysis_race(_teams1, _years, _out, _nboot)
            else:
                team_res1 = cached_team_analysis(_teams1, _years, _out, _nboot)
    else:
        with st.spinner(f"Analysis {_label1}…"):
            if _race_level:
                res1 = cached_analysis_race(_r1, _teams1, _years, _out, _nboot, _ntrees, _rcf)
            else:
                res1 = cached_analysis(_r1, _teams1, _years, _out, _nboot, _ntrees, _rcf)

    if p['mode'] == "Comparison" and p['teams2']:
        _r2, _teams2 = p['rider2'], p['teams2']
        if p['team_mode2']:
            with st.spinner(f"Team analysis: {_label2}…"):
                if _race_level:
                    team_res2 = cached_team_analysis_race(_teams2, _years, _out, _nboot)
                else:
                    team_res2 = cached_team_analysis(_teams2, _years, _out, _nboot)
        elif _r2:
            with st.spinner(f"Analysis {_label2}…"):
                if _race_level:
                    res2 = cached_analysis_race(_r2, _teams2, _years, _out, _nboot, _ntrees, _rcf)
                else:
                    res2 = cached_analysis(_r2, _teams2, _years, _out, _nboot, _ntrees, _rcf)
else:
    _t1_mode = team_mode1
    _label1, _label2 = label1, label2

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DML
# ════════════════════════════════════════════════════════════════════════════════
with tab_dml:
  if not _ran:
    st.info("Click **Run analysis** in the sidebar to see the DML results.")
  elif _t1_mode and team_res1 is None:
    st.error(f"No rider with enough data for {_label1}.")
  elif not _t1_mode and res1 is None:
    st.error(f"Insufficient data for {_label1} (not enough valid observations after filtering).")
  else:
    def show_dml_metrics(res, label):
        dml = res['dml']
        st.subheader(label)
        if dml is None:
            st.warning("DML not available (not enough variance in T).")
            return
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ATE", f"{dml['ate_orig']:+.2f} UCI pts")
        c2.metric("95% CI", f"[{dml['ci_low']:+.1f}, {dml['ci_high']:+.1f}]")
        c3.metric("Significant", "✓ Yes" if dml['significant'] else "✗ No")
        c4.metric("R² model T", f"{dml['r2_t']:.3f}", help="Quality of the selection model")
        c5.metric("R² model Y", f"{dml['r2_y']:.3f}", help="Quality of the performance model")
        st.caption(
            f"{res['n_obs']} observations · {res['n_selected']} selections · outcome: **{res['outcome']}**"
        )

    # ── Team view: forest plot for all riders ────────────────────────────
    def show_team_forest(df_team, label):
        st.subheader(label)
        df = df_team.sort_values('ate_orig', ascending=True)
        colors = ['#2271B3' if s else '#BDBDBD' for s in df['significant']]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['ate_orig'],
            y=df['rider'].apply(fmt_rider),
            orientation='h',
            marker_color=colors,
            error_x=dict(
                type='data', symmetric=False,
                array=(df['ci_high'] - df['ate_orig']).clip(lower=0).tolist(),
                arrayminus=(df['ate_orig'] - df['ci_low']).clip(lower=0).tolist(),
                color='#444',
            ),
            customdata=df[['n_selected', 'taux_sel', 'r2_t', 'r2_y']].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "ATE: %{x:+.2f} UCI pts<br>"
                "Selections: %{customdata[0]} (%{customdata[1]:.0%})<br>"
                "R²_T: %{customdata[2]:.3f} | R²_Y: %{customdata[3]:.3f}"
                "<extra></extra>"
            ),
        ))
        fig.add_vline(x=0, line_dash='dash', line_color='red', opacity=0.6)
        fig.update_layout(
            title="ATE by rider (blue = significant, grey = not significant)",
            xaxis_title="ATE (team UCI pts)",
            template='plotly_white',
            height=max(350, len(df) * 28),
            margin=dict(l=160),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df[['rider', 'n_selected', 'taux_sel', 'ate_orig', 'ci_low', 'ci_high', 'significant', 'r2_t', 'r2_y']]
            .assign(rider=df['rider'].apply(fmt_rider))
            .reset_index(drop=True),
            use_container_width=True,
        )

    if _t1_mode:
        show_team_forest(team_res1, label1)
    else:
        show_dml_metrics(res1, label1)

    if mode == "Comparison" and res2:
        st.divider()
        show_dml_metrics(res2, label2)
        st.divider()
        st.subheader("Direct comparison")
        d1, d2 = res1['dml'], res2['dml']
        if d1 and d2:
            fig = go.Figure()
            for lbl, dml, color in [(label1, d1, '#2271B3'), (label2, d2, '#E8824B')]:
                fig.add_trace(go.Bar(
                    name=lbl, x=[lbl], y=[dml['ate_orig']],
                    error_y=dict(
                        type='data', symmetric=False,
                        array=[dml['ci_high'] - dml['ate_orig']],
                        arrayminus=[dml['ate_orig'] - dml['ci_low']],
                    ),
                    marker_color=color,
                ))
            fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.5)
            fig.update_layout(title="ATE compared (95% CI)", yaxis_title="ATE (UCI pts)",
                              template='plotly_white', showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    if res1 and res1['dml']:
        with st.expander("DML residuals"):
            T_r, Y_r = res1['dml']['T_resid'], res1['dml']['Y_resid']
            fig_res = go.Figure(go.Scatter(
                x=T_r, y=Y_r, mode='markers',
                marker=dict(size=4, opacity=0.4, color='#2271B3'),
            ))
            fig_res.update_layout(
                xaxis_title='Residual T̃', yaxis_title='Residual Ỹ',
                title='DML residuals — slope = ATE',
                template='plotly_white', height=350,
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with st.expander("Predicted vs Actual — outcome model"):
            _outcome_col = res1.get('outcome', 'pts_uci_equipe_stage')
            _df_diag = res1['df_clean'].copy()
            _Y_resid = np.array(res1['dml']['Y_resid'])
            _n = len(_Y_resid)

            if _outcome_col in _df_diag.columns:
                _Y_log_actual = np.log1p(_df_diag[_outcome_col].values[:_n])
                _Y_log_pred   = _Y_log_actual - _Y_resid
                _abs_resid    = np.abs(_Y_resid)

                _df_diag = _df_diag.iloc[:_n].copy()
                _df_diag['_Y_actual'] = _Y_log_actual
                _df_diag['_Y_pred']   = _Y_log_pred
                _df_diag['_resid']    = _Y_resid
                _df_diag['_abs_resid']= _abs_resid

                # Keep only races where the rider was selected
                if 'selected' in _df_diag.columns:
                    _df_diag = _df_diag[_df_diag['selected'] == 1].copy().reset_index(drop=True)
                    _Y_log_actual = _df_diag['_Y_actual'].values
                    _Y_log_pred   = _df_diag['_Y_pred'].values
                    _abs_resid    = _df_diag['_abs_resid'].values
                # ── Scatter predicted vs actual ─────────────────────────
                _clusters = (
                    _df_diag['stage_cluster_label'].fillna('Unknown')
                    if 'stage_cluster_label' in _df_diag.columns
                    else pd.Series(['All'] * len(_df_diag))
                )
                _CLUSTER_COLORS = {
                    '⏱️  TT':               '#9b59b6',
                    '🟢  Flat/Sprint':       '#27ae60',
                    '⛰️  Medium mountain':  '#e67e22',
                    '🏔️  High mountain':    '#e74c3c',
                }
                fig_pred = go.Figure()
                for _cl in _clusters.unique():
                    _mask = (_clusters == _cl).values
                    _color = _CLUSTER_COLORS.get(_cl, '#2271B3')
                    _hover = (
                        _df_diag.loc[_mask, 'course'].str.replace('-', ' ').apply(
                            lambda s: unicodedata.normalize('NFC', s).title()
                        ) + ' ' + _df_diag.loc[_mask, 'year'].astype(int).astype(str)
                        if 'course' in _df_diag.columns else pd.Series([''] * _mask.sum())
                    )
                    fig_pred.add_trace(go.Scatter(
                        x=_Y_log_pred[_mask],
                        y=_Y_log_actual[_mask],
                        mode='markers',
                        name=str(_cl),
                        text=_hover,
                        hovertemplate='<b>%{text}</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>',
                        marker=dict(size=5, opacity=0.6, color=_color),
                    ))

                _lim = max(_Y_log_actual.max(), _Y_log_pred.max()) * 1.05
                fig_pred.add_trace(go.Scatter(
                    x=[0, _lim], y=[0, _lim],
                    mode='lines', name='Perfect prediction',
                    line=dict(color='red', dash='dash', width=1.5),
                    showlegend=True,
                ))
                fig_pred.add_annotation(
                    x=0.02, y=0.98, xref='paper', yref='paper',
                    text=f"R² = {res1['dml']['r2_y']:.3f}",
                    showarrow=False, font=dict(size=13),
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#ccc', borderwidth=1,
                    xanchor='left', yanchor='top',
                )
                fig_pred.update_layout(
                    title='Predicted vs Actual — outcome model Y (log scale)',
                    xaxis_title='Predicted  log(1 + UCI pts)',
                    yaxis_title='Actual  log(1 + UCI pts)',
                    template='plotly_white', height=420,
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                st.caption(
                    "Points on the diagonal = perfect predictions. "
                    "Above = team scored more than expected; below = less than expected. "
                    f"R² = {res1['dml']['r2_y']:.3f}."
                )

                # ── Prediction quality by race type ─────────────────────
                if 'stage_cluster_label' in _df_diag.columns:
                    st.markdown("**Prediction quality by race type**")
                    _by_cl = []
                    for _cl, _g in _df_diag.groupby('stage_cluster_label'):
                        _ya = _g['_Y_actual'].values
                        _yp = _g['_Y_pred'].values
                        _ss_res = ((_ya - _yp) ** 2).sum()
                        _ss_tot = ((_ya - _ya.mean()) ** 2).sum()
                        _r2 = 1 - _ss_res / _ss_tot if _ss_tot > 0 else float('nan')
                        _mae = float(np.abs(_ya - _yp).mean())
                        _by_cl.append({'Race type': _cl, 'N': len(_g), 'R²': round(_r2, 3), 'MAE (log)': round(_mae, 3)})
                    _df_cl = pd.DataFrame(_by_cl).sort_values('R²', ascending=False).reset_index(drop=True)
                    st.dataframe(_df_cl.style.format({'R²': '{:.3f}', 'MAE (log)': '{:.3f}'}),
                                 use_container_width=True)
                    st.caption("R² per race type — higher = the model predicts well for this profile. MAE in log(1+pts) units.")

                # ── Worst predicted stages ──────────────────────────────
                st.markdown("**Stages with largest prediction error**")
                _diag_cols = ['course', 'year', 'stage_cluster_label', '_Y_actual', '_Y_pred', '_resid']
                _diag_cols = [c for c in _diag_cols if c in _df_diag.columns]
                _worst = (
                    _df_diag.nlargest(10, '_abs_resid')[_diag_cols]
                    .rename(columns={
                        '_Y_actual': 'Y actual (log)',
                        '_Y_pred':   'Y predicted (log)',
                        '_resid':    'Residual',
                    })
                    .reset_index(drop=True)
                )
                if 'course' in _worst.columns:
                    _worst['course'] = _worst['course'].str.replace('-', ' ').apply(
                        lambda s: unicodedata.normalize('NFC', s).title()
                    )
                st.dataframe(_worst.style.format({
                    'Y actual (log)': '{:.3f}',
                    'Y predicted (log)': '{:.3f}',
                    'Residual': '{:+.3f}',
                }), use_container_width=True)
                st.caption(
                    "Positive residual = the team scored more than the model expected (positive surprise). "
                    "Negative residual = the team underperformed relative to race context."
                )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Causal Forest
# ════════════════════════════════════════════════════════════════════════════════
def _render_cf():
    if not _ran:
        st.info("Click **Run analysis** in the sidebar to see the Causal Forest results.")
        return
    if not p.get('run_cf', True):
        st.info("Causal Forest disabled. Enable it in advanced options.")
        return
    if _t1_mode:
        st.info("In team mode, select a specific rider to see their Causal Forest.")
        return
    if res1 is None or 'cate' not in res1:
        st.warning("Causal Forest not computed.")
        return

    # Controls
    # Features used in the model (in order of importance if CF is available)
    available_features = res1['features']
    if 'cf_model' in res1:
        imp_order = list(
            pd.Series(res1['cf_model'].feature_importances_, index=available_features)
            .sort_values(ascending=False).index
        )
    else:
        imp_order = available_features

    # ── Controls ─────────────────────────────────────────────────────────────
    col_x, col_color, col_trend, col_yr, col_filter = st.columns([2, 2, 2, 2, 2])
    with col_x:
        x_col = st.selectbox(
            "Variable X",
            imp_order,
            index=imp_order.index('denivele_pos') if 'denivele_pos' in imp_order else 0,
        )
    with col_color:
        FEAT_LABELS = {
            'denivele_pos': 'D+ (m)', 'denivele_neg': 'D− (m)',
            'distance_gpx_km': 'Distance (km)', 'startlist_quality': 'Startlist quality',
            'n_cols_hc': 'HC climbs', 'n_cols_cat1': 'Cat1 climbs',
            'n_cols_cat2': 'Cat2 climbs', 'n_cols_cat3': 'Cat3 climbs',
            'n_cols_cat4': 'Cat4 climbs', 'cobblestones_km': 'Cobblestones (km)',
            'compacted_gravel_km': 'Gravel (km)', 'forme_equipe': 'Team form',
            'n_races_30d': 'Races/30d', 'km_30d': 'Km/30d',
            'is_team_leader': 'Team leader', 'leader_played': 'Leader present',
            'gradient_last_5km': 'Final gradient (5km)',
            'altitude_max': 'Max altitude (m)', 'altitude_min': 'Min altitude (m)',
            'loc_last_col_hc': 'Last HC climb position',
            'loc_last_col_cat1': 'Last Cat1 climb position',
            'deniv_last_5km': 'D+ last 5km',
            'top_score_in_team': 'Top scorer in team',
            'forme_coureur': 'Rider form',
        }
        COLOR_OPTIONS = {
            'Selected': 'selected',
            'Leader present': 'leader',
            'Year': 'year',
            'Stage type': 'stage_cluster',
            'Cobblestones': 'cobbles_cat',
            'CATE +/-': 'cate_sign',
        }
        COLOR_OPTIONS.update({
            FEAT_LABELS.get(f, f): f
            for f in cm.ALL_FEATURES
        })
        color_choice = st.selectbox("Color", list(COLOR_OPTIONS.keys()))
        color_col = COLOR_OPTIONS[color_choice]
    with col_trend:
        trend_choice = st.selectbox("Trend", ['None', 'Linear (OLS)', 'LOWESS'])
        trendline = {'None': None, 'Linear (OLS)': 'ols', 'LOWESS': 'lowess'}[trend_choice]
    with col_yr:
        available_years = sorted(res1['df_clean']['year'].dropna().astype(int).unique())
        year_filter = st.multiselect("Filter by year", available_years, default=[])
    with col_filter:
        show_selected_only = st.checkbox("Selected only", value=True)

    def enrich_cf_cols(df):
        df = enrich_with_leader(df.copy())
        if 'leader_played' in df.columns:
            df['leader'] = df['leader_played'].map(
                {1.0: 'Leader present', 0.0: 'No leader'}
            ).fillna('No leader')
        else:
            df['leader'] = 'N/A'
        df['year_str'] = df['year'].astype(int).astype(str)
        df['cobbles_cat'] = (df['cobblestones_km'].fillna(0) > 0).map(
            {True: 'Cobbles', False: 'Asphalt'}
        ) if 'cobblestones_km' in df.columns else 'N/A'
        df['cate_sign'] = (df['cate'] > 0).map({True: 'CATE > 0', False: 'CATE ≤ 0'})
        df['course_label'] = (
            df['course'].str.replace('-', ' ').apply(lambda s: unicodedata.normalize('NFC', s).title())
            + ' ' + df['year'].astype(int).astype(str)
            + (df['stage_num'].apply(lambda s: f' st.{int(s)}' if pd.notna(s) else '') if 'stage_num' in df.columns else '')
        )
        return df

    def _show_course_card(row, df_ref=None, features=None, compare_df=None, compare_label=None, cf_model=None, X_train=None, key_suffix=''):
        with st.container(border=True):
            st.markdown(f"### {row.get('course_label', row.get('course', '?'))}")
            # CATE row — primary + comparison if Comparison mode
            cate_compare = None
            if compare_df is not None:
                same = compare_df[
                    (compare_df['course'] == row.get('course'))
                    & (compare_df['year'].astype(int) == int(row.get('year', 0)))
                ]
                if 'stage_num' in row.index and pd.notna(row.get('stage_num')):
                    same = same[same['stage_num'].astype(float) == float(row['stage_num'])]
                if len(same) > 0:
                    cate_compare = same.iloc[0]['cate']
            if cate_compare is not None:
                cc1, cc2 = st.columns(2)
                cc1.metric("CATE", f"{row['cate']:+.3f} UCI pts")
                cc2.metric(
                    f"CATE — {compare_label or 'comparison'}",
                    f"{cate_compare:+.3f} UCI pts",
                    delta=f"{cate_compare - row['cate']:+.3f}",
                )
            else:
                st.metric("CATE", f"{row['cate']:+.3f} UCI pts")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Distance",
                      f"{row.get('distance_gpx_km', '?'):.0f} km"
                      if pd.notna(row.get('distance_gpx_km')) else "—")
            c2.metric("D+",
                      f"{row.get('denivele_pos', '?'):.0f} m"
                      if pd.notna(row.get('denivele_pos')) else "—")
            c3.metric("Max alt.",
                      f"{row.get('altitude_max', '?'):.0f} m"
                      if pd.notna(row.get('altitude_max')) else "—")
            c4.metric("Startlist",
                      f"{row.get('startlist_quality', '?'):.0f}"
                      if pd.notna(row.get('startlist_quality')) else "—")

            cob_km = float(row.get('cobblestones_km') or 0)
            grav_km = float(row.get('compacted_gravel_km') or 0)

            gpx_df = None if _race_level else load_gpx_profile(
                row.get('course', ''), row.get('date'), row.get('stage_num'),
            )

            if gpx_df is not None and len(gpx_df) > 10:
                fig_p = go.Figure()

                # ── Slope gradient (green → red) ────────────────────
                dx = gpx_df['distance_km'].diff().fillna(0.001).clip(lower=0.001)
                dy = gpx_df['elevation'].diff().fillna(0)
                grad_pct = (dy / (dx * 10)).clip(-20, 20)

                _GRAD_BINS   = [-999, 3, 6, 9, 999]
                _GRAD_COLORS = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
                grad_cat = pd.cut(grad_pct.clip(lower=0), bins=_GRAD_BINS,
                                  labels=False, right=False).fillna(0).astype(int)

                # Blue fill area (no own line color)
                fig_p.add_trace(go.Scatter(
                    x=gpx_df['distance_km'], y=gpx_df['elevation'],
                    fill='tozeroy', fillcolor='rgba(34,113,179,0.10)',
                    line=dict(color='rgba(0,0,0,0)', width=0), mode='lines',
                    showlegend=False, hoverinfo='skip',
                ))

                # Colored segments grouped by category (N traces << N points)
                xs = gpx_df['distance_km'].tolist()
                ys = gpx_df['elevation'].tolist()
                cs = grad_cat.tolist()
                seg_x = [xs[0]]; seg_y = [ys[0]]; cur_c = cs[0]
                for i in range(1, len(xs)):
                    seg_x.append(xs[i]); seg_y.append(ys[i])
                    if cs[i] != cur_c or i == len(xs) - 1:
                        fig_p.add_trace(go.Scatter(
                            x=seg_x, y=seg_y, mode='lines',
                            line=dict(color=_GRAD_COLORS[cur_c], width=2.5),
                            showlegend=False,
                            hovertemplate='%{x:.1f} km — %{y:.0f} m<extra></extra>',
                        ))
                        seg_x = [xs[i]]; seg_y = [ys[i]]; cur_c = cs[i]

                # ── Climbs: vertical line at the position of the last climb ──────
                total_km = gpx_df['distance_km'].max()
                for col_name, loc_col in [('HC', 'loc_last_col_hc'),
                                           ('Cat 1', 'loc_last_col_cat1'),
                                           ('Cat 2', 'loc_last_col_cat2')]:
                    loc = row.get(loc_col)
                    if pd.notna(loc) and 0 < float(loc) < 1:
                        fig_p.add_vline(
                            x=float(loc) * total_km,
                            line_dash='dot', line_color='#555', line_width=1.2,
                            annotation_text=col_name,
                            annotation_position='top left',
                            annotation_font_size=11,
                        )

                # ── Y axis: minimum range 500m, enough top margin ───────
                _elev_min = gpx_df['elevation'].min()
                _elev_max = gpx_df['elevation'].max()
                _half = max((_elev_max - _elev_min) / 2, 250)
                _mid  = (_elev_min + _elev_max) / 2
                _y_low  = max(0, _mid - _half - 30)
                _y_high = _mid + _half + 120  # top margin for climb annotations

                # Surface legend + cobblestones/gravel text
                surf_parts = []
                if cob_km > 0:
                    surf_parts.append(f"🟫 Cobbles {cob_km:.1f} km")
                if grav_km > 0:
                    surf_parts.append(f"🟤 Gravel {grav_km:.1f} km")

                fig_p.update_layout(
                    xaxis_title='Distance (km)', yaxis_title='Elevation (m)',
                    template='plotly_white', height=280,
                    margin=dict(t=10 if not surf_parts else 30, b=40, l=50, r=20),
                    yaxis=dict(range=[_y_low, _y_high]),
                    title=dict(text=' | '.join(surf_parts), font=dict(size=12), x=0) if surf_parts else None,
                )
                st.plotly_chart(fig_p, use_container_width=True, key=f'gpx_{key_suffix}')
            elif not _race_level:
                st.caption("GPX profile not available for this race.")

            # Counts are cumulative (≥ threshold), display exclusive counts
            n_hc   = int(row.get('n_cols_hc')   or 0)
            n_cat1 = int(row.get('n_cols_cat1') or 0) - n_hc
            n_cat2 = int(row.get('n_cols_cat2') or 0) - int(row.get('n_cols_cat1') or 0)
            n_cat3 = int(row.get('n_cols_cat3') or 0) - int(row.get('n_cols_cat2') or 0)
            n_cat4 = int(row.get('n_cols_cat4') or 0) - int(row.get('n_cols_cat3') or 0)
            col_info = [(n_hc,'HC'),(n_cat1,'Cat1'),(n_cat2,'Cat2'),(n_cat3,'Cat3'),(n_cat4,'Cat4')]
            col_str = ", ".join(f"{n}×{c}" for n, c in col_info if n > 0)
            if col_str:
                st.caption("Climbs: " + col_str)
            surf = []
            if float(row.get('cobblestones_km') or 0) > 0:
                surf.append(f"Cobbles {row['cobblestones_km']:.1f} km")
            if float(row.get('compacted_gravel_km') or 0) > 0:
                surf.append(f"Gravel {row['compacted_gravel_km']:.1f} km")
            if surf:
                st.caption("Surface: " + " | ".join(surf))

            # ── CATE explanation (SHAP) ────────────────────────────────────
            if cf_model is not None and X_train is not None and features:
                labels = {
                    'denivele_pos': 'D+ (m)', 'denivele_neg': 'D− (m)',
                    'distance_gpx_km': 'Distance (km)', 'startlist_quality': 'Startlist quality',
                    'n_cols_hc': 'HC climbs', 'n_cols_cat1': 'Cat1 climbs',
                    'n_cols_cat2': 'Cat2 climbs', 'n_cols_cat3': 'Cat3 climbs',
                    'n_cols_cat4': 'Cat4 climbs', 'cobblestones_km': 'Cobblestones (km)',
                    'compacted_gravel_km': 'Gravel (km)', 'forme_equipe': 'Team form',
                    'n_races_30d': 'Races/30d', 'km_30d': 'Km/30d',
                    'is_team_leader': 'Team leader', 'leader_played': 'Leader present',
                    'gradient_last_5km': 'Final gradient (5km)',
                    'altitude_max': 'Max altitude (m)', 'altitude_min': 'Min altitude (m)',
                    'loc_last_col_hc': 'Last HC climb position',
                    'loc_last_col_cat1': 'Last Cat1 climb position',
                    'deniv_last_5km': 'D+ last 5km',
                    'top_score_in_team': 'Top scorer in team',
                    'forme_coureur': 'Rider form',
                    'year': 'Year',
                }
                with st.expander("🔍 Why this CATE?", expanded=False):
                    with st.spinner("Computing SHAP values…"):
                        try:
                            import shap as _shap
                            x_vals = np.array(
                                [float(row.get(f, 0.0) or 0.0) for f in features]
                            ).reshape(1, -1)

                            def _cf_predict(X_in):
                                return cf_model.effect(X_in, T0=0, T1=1).flatten()

                            rng = np.random.RandomState(42)
                            n_bg = min(30, len(X_train))
                            bg = X_train[rng.choice(len(X_train), n_bg, replace=False)]

                            explainer = _shap.KernelExplainer(_cf_predict, bg)
                            sv = explainer.shap_values(x_vals, nsamples=200, silent=True)
                            shap_arr = np.array(sv[0] if isinstance(sv, list) else sv[0])

                            _base = float(explainer.expected_value)
                            _cate = float(row.get('cate', _base + shap_arr.sum()))
                            st.markdown(
                                f"**Average CATE across dataset**: **{_base:+.3f} pts** → "
                                f"**This race**: **{_cate:+.3f} pts**\n\n"
                                f"Each bar shows how much a feature pushes this race's CATE "
                                f"above or below the dataset average ({_base:+.3f} pts). "
                                "🟢 **Green** = favourable vs average (e.g. harder race, better startlist). "
                                "🔴 **Red** = unfavourable. "
                                "All-red bars with a positive CATE simply means the effect exists "
                                "but is weaker than the rider's usual — every feature is below average."
                            )

                            shap_s = pd.Series(dict(zip(features, shap_arr)))
                            shap_s = shap_s.sort_values(key=abs, ascending=False)
                            # Keep features contributing ≥1% of total absolute impact
                            _abs_total = shap_s.abs().sum()
                            _thresh = _abs_total * 0.01 if _abs_total > 0 else 0
                            shap_s = shap_s[shap_s.abs() >= _thresh].head(20)
                            names  = [labels.get(f, f) for f in shap_s.index]
                            colors = ['#2d7a3a' if v > 0 else '#c0392b' for v in shap_s.values]

                            fig_shap = go.Figure(go.Bar(
                                x=shap_s.values, y=names,
                                orientation='h',
                                marker_color=colors,
                                hovertemplate='%{y}: %{x:+.4f} pts<extra></extra>',
                            ))
                            fig_shap.add_vline(x=0, line_color='#888', line_width=1)
                            fig_shap.update_layout(
                                template='plotly_white',
                                height=max(320, 28 * len(shap_s)),
                                xaxis_title='SHAP value (impact on CATE, UCI pts)',
                                margin=dict(t=10, b=40, l=10, r=20),
                                yaxis=dict(autorange='reversed'),
                            )
                            st.plotly_chart(fig_shap, use_container_width=True, key=f'shap_{key_suffix}')
                        except Exception as _e:
                            st.caption(f"SHAP unavailable: {_e}")

    def build_cf_df(res, selected_only, year_filter, course_max_pts=None):
        df = enrich_cf_cols(res['df_clean'].copy())
        if course_max_pts is not None:
            df = df.join(course_max_pts, on='course')
            df['cate_pct_max'] = (df['cate'] / df['pts_course_max'].clip(lower=1) * 100).round(1)
        if 'pts_uci_equipe_stage' in df.columns and 'max_pts' in df.columns:
            df['team_pct_winner'] = (
                df['pts_uci_equipe_stage'] / df['max_pts'].replace(0, float('nan')) * 100
            ).round(1)
        if selected_only:
            df = df[df['selected'] == 1]
        if year_filter:
            df = df[df['year'].astype(int).isin(year_filter)]
        return df

    def make_scatter(df, label, x_col, color_col, key_suffix='', symbol_col=None):
        df = df.copy()
        # Binary columns → readable labels (categorical treatment, not a gradient)
        _BINARY_LABELS = {'selected': {0: 'Not selected', 1: 'Selected'}}
        if color_col in _BINARY_LABELS:
            df[color_col] = df[color_col].map(_BINARY_LABELS[color_col]).fillna('?')
        hover_extra = {k: True for k in ['n_cols_hc', 'startlist_quality', 'leader_name']
                       if k in df.columns}
        hover_extra['cate'] = ':.3f'
        if 'cate_pct_max' in df.columns:
            hover_extra['cate_pct_max'] = ':.1f'
        if 'team_pct_winner' in df.columns:
            hover_extra['team_pct_winner'] = ':.1f'
        if 'pts_uci_equipe_stage' in df.columns:
            hover_extra['pts_uci_equipe_stage'] = ':.0f'
        if x_col in df.columns:
            hover_extra[x_col] = ':.1f'
        hover_extra[color_col] = False
        fig = px.scatter(
            df, x=x_col, y='cate',
            color=color_col,
            color_continuous_scale='Blues' if pd.api.types.is_numeric_dtype(df[color_col]) else None,
            hover_name='course_label',
            hover_data=hover_extra,
            labels={x_col: x_col, 'cate': 'CATE (estimated UCI pts)'},
            title=f"{label} — CATE × {x_col}",
            template='plotly_white',
            trendline=trendline,
            trendline_scope='overall',
            trendline_color_override='#222',
        )
        fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.4)
        # Mark shared races (diamond) without going through the legend system
        if symbol_col and symbol_col in df.columns:
            shared_mask = df[symbol_col].str.contains('◆', na=False)
            if shared_mask.any():
                shared = df[shared_mask]
                fig.add_trace(go.Scatter(
                    x=shared[x_col], y=shared['cate'],
                    mode='markers',
                    marker=dict(symbol='diamond', size=9, color='rgba(0,0,0,0)',
                                line=dict(color='#333', width=1.5)),
                    hoverinfo='skip', showlegend=False, name='',
                ))
        fig.update_traces(selector=dict(mode='markers', showlegend=True),
                          marker=dict(size=7, opacity=0.75))
        if trendline == 'ols':
            try:
                r2 = px.get_trendline_results(fig).iloc[0]['px_fit_results'].rsquared
                fig.add_annotation(
                    x=1, y=1, xref='paper', yref='paper',
                    text=f"R² = {r2:.3f}", showarrow=False,
                    font=dict(size=13, color='#222'),
                    bgcolor='rgba(255,255,255,0.75)',
                    bordercolor='#ccc', borderwidth=1,
                    xanchor='right', yanchor='top',
                )
            except Exception:
                pass
        return fig

    df_cf_all = enrich_cf_cols(res1['df_clean'].copy())  # all races, for missed opportunities
    # Attach DML residuals (aligned row-for-row with df_clean)
    _dml_res = res1.get('dml') or {}
    if 'Y_resid' in _dml_res and len(_dml_res['Y_resid']) == len(df_cf_all):
        df_cf_all['Y_resid'] = np.array(_dml_res['Y_resid'])
        df_cf_all['T_resid'] = np.array(_dml_res['T_resid'])
    # % impact: CATE as % of max pts ever scored by the team at that specific course
    course_max = df_cf_all.groupby('course')['pts_uci_equipe_stage'].max().rename('pts_course_max')
    df_cf_all = df_cf_all.join(course_max, on='course')
    df_cf_all['cate_pct_max'] = (
        df_cf_all['cate'] / df_cf_all['pts_course_max'].replace(0, float('nan')) * 100
    ).round(1)
    if 'pts_uci_equipe_stage' in df_cf_all.columns and 'max_pts' in df_cf_all.columns:
        df_cf_all['team_pct_winner'] = (
            df_cf_all['pts_uci_equipe_stage'] / df_cf_all['max_pts'].replace(0, float('nan')) * 100
        ).round(1)
    if year_filter:
        df_cf_all = df_cf_all[df_cf_all['year'].astype(int).isin(year_filter)]

    df_cf1 = build_cf_df(res1, show_selected_only, year_filter, course_max)

    # ── Main scatter ─────────────────────────────────────────────────────
    if p.get('mode') == "Comparison" and res2 and 'cate' in res2:
        df_cf2 = build_cf_df(res2, show_selected_only, year_filter, course_max)

        # Mark races present for both riders (diamond)
        _merge_keys = ['course', 'year'] + (['stage_num'] if 'stage_num' in df_cf1.columns else [])
        _common = df_cf2[_merge_keys].drop_duplicates().assign(_in2=True)
        df_cf1 = df_cf1.merge(_common, on=_merge_keys, how='left')
        df_cf1['_presence'] = df_cf1['_in2'].map({True: f'also {_label2} ◆', float('nan'): 'only'}).fillna('only')
        df_cf1.drop(columns=['_in2'], inplace=True)
        _common2 = df_cf1[_merge_keys].assign(_in1=True)
        df_cf2 = df_cf2.merge(_common2.drop_duplicates(), on=_merge_keys, how='left')
        df_cf2['_presence'] = df_cf2['_in1'].map({True: f'also {_label1} ◆', float('nan'): 'only'}).fillna('only')
        df_cf2.drop(columns=['_in1'], inplace=True)

        col_l, col_r = st.columns(2)
        with col_l:
            sel1 = st.plotly_chart(
                make_scatter(df_cf1, _label1, x_col, color_col, symbol_col='_presence'),
                use_container_width=True, on_select='rerun', key='scatter_cf1',
            )
        with col_r:
            sel2 = st.plotly_chart(
                make_scatter(df_cf2, _label2, x_col, color_col, symbol_col='_presence'),
                use_container_width=True, on_select='rerun', key='scatter_cf2',
            )
        selected_point = (
            (sel1.get('selection', {}).get('points') or [])
            + (sel2.get('selection', {}).get('points') or [])
        )
        fig_dist = go.Figure()
        for lbl, df_c, color in [(_label1, df_cf1, '#2271B3'), (_label2, df_cf2, '#E8824B')]:
            fig_dist.add_trace(go.Histogram(
                x=df_c['cate'], name=lbl, opacity=0.65,
                marker_color=color, nbinsx=30,
            ))
        fig_dist.update_layout(barmode='overlay', template='plotly_white',
                               xaxis_title='CATE', yaxis_title='Nb races', height=280)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        sel = st.plotly_chart(
            make_scatter(df_cf1, _label1, x_col, color_col),
            use_container_width=True, on_select='rerun', key='scatter_cf1',
        )
        selected_point = sel.get('selection', {}).get('points') or []

    # ── Course card on click (scatter) ───────────────────────────────────────
    if selected_point:
        pt = selected_point[0]
        pt_x, pt_y = pt.get('x'), pt.get('y')
        match = df_cf_all[
            (df_cf_all[x_col].round(3) == round(pt_x, 3))
            & (df_cf_all['cate'].round(3) == round(pt_y, 3))
        ]
        if len(match) == 0:
            match = df_cf_all.iloc[
                [(df_cf_all[x_col] - pt_x).abs().add(
                 (df_cf_all['cate'] - pt_y).abs()).argmin()]
            ]
        _compare_df    = df_cf2 if (p.get('mode') == "Comparison" and 'df_cf2' in dir()) else None
        _compare_label = _label2 if _compare_df is not None else None
        _mrow = match.iloc[0]
        _show_course_card(_mrow, df_ref=df_cf_all, features=res1.get('features'),
                          compare_df=_compare_df, compare_label=_compare_label,
                          cf_model=res1.get('cf_model'), X_train=res1.get('X'),
                          key_suffix='main')
        _race_label = _mrow.get('course_label', _mrow.get('course', ''))
        _race_year  = int(_mrow.get('year', 0))
        _race_course = _mrow.get('course', '')

        if _race_level:
            # ── Race-level: stage selector + separate GC table ───────
            with st.spinner("Loading race results…"):
                _df_all   = load_all_race_stages(_race_course, _race_year)
                _df_gc    = load_gc_classification(_race_course, _race_year)
            st.markdown(f"**Results — {_race_label} {_race_year}**")
            _team_kw = [t.split('|')[0].strip().lower() for t in p.get('teams1', teams1)]

            # ── GC table ─────────────────────────────────────────────
            if _df_gc is not None:
                st.markdown("**General Classification**")
                _team_only_gc = st.toggle("Team only", value=False, key='results_team_only_gc')
                _df_gc_show = _df_gc.copy()
                if _team_only_gc:
                    _df_gc_show = _df_gc_show[_df_gc_show['Team'].str.lower().apply(
                        lambda t: any(kw in t for kw in _team_kw)
                    )]
                st.dataframe(
                    _df_gc_show.style.format({'GC Rank': '{:.0f}', 'UCI pts (GC)': '{:.0f}', 'Team UCI pts (GC)': '{:.0f}'}),
                    use_container_width=True, hide_index=True,
                )
                st.caption("⚠️ Only includes riders present in the model dataset — not the complete GC classification.")
            else:
                st.caption("No GC data available for this race.")

            # ── Stage-by-stage table ──────────────────────────────────
            if _df_all is not None:
                _avail = sorted(_df_all['stage_num'].unique(), key=_stage_sort_key)
                _avail_named = [s for s in _avail if s not in ('', 'gc', 'GC')]
                if _avail_named:
                    st.markdown("**Stage results**")
                    _col_sel, _col_tog = st.columns([3, 1])
                    with _col_sel:
                        _sel_sn = st.selectbox(
                            "Stage", options=_avail_named,
                            format_func=lambda s: f'Stage {s}',
                            key='race_level_stage_sel',
                        )
                    with _col_tog:
                        _team_only_rl = st.toggle("Team only", value=False, key='results_team_only_rl')
                    _df_sn = _df_all[_df_all['stage_num'] == _sel_sn].drop(columns=['stage_num'])
                    _df_sn['Team UCI pts'] = _df_sn.groupby('Team')['UCI pts'].transform('sum')
                    _df_sn = _df_sn.sort_values('Rank').reset_index(drop=True)
                    if _team_only_rl:
                        _eq_kw_rl = _race_equipe_kw(_mrow)
                        _df_sn = _df_sn[
                            _df_sn['Team'].str.lower().str.contains(_eq_kw_rl, regex=False)
                        ]
                    st.dataframe(
                        _df_sn.style.format({'UCI pts': '{:.0f}', 'Team UCI pts': '{:.0f}'}),
                        use_container_width=True, hide_index=False,
                    )
            if _df_gc is None and _df_all is None:
                st.caption("No result data available for this race.")
        else:
            # ── Stage-level: single stage result ─────────────────────
            with st.spinner("Loading results…"):
                _df_results_main = load_race_results(
                    _race_course, _race_year, _mrow.get('stage_num', None)
                )
            if _df_results_main is not None:
                st.markdown(f"**Results — {_race_label}**")
                _df_results_main = _inject_team_uci_pts(_df_results_main, _mrow)
                _team_only_main = st.toggle("Team only", value=False, key='results_team_only_main')
                if _team_only_main:
                    _eq_kw_main = _race_equipe_kw(_mrow)
                    _df_results_main = _df_results_main[
                        _df_results_main['Team'].str.lower().str.contains(_eq_kw_main, regex=False)
                    ]
                st.dataframe(
                    _df_results_main.style.format({'UCI pts': '{:.0f}', 'Team UCI pts': '{:.0f}'}),
                    use_container_width=True, hide_index=False,
                )
            else:
                st.caption("No result data available for this race.")

    # ── CATE vs Actual result (selected races only) ──────────────────────────
    _outcome_col = res1.get('outcome', 'pts_uci_equipe_stage')
    _df_sel_only = df_cf_all[df_cf_all['selected'] == 1].copy()
    if _outcome_col in _df_sel_only.columns and 'cate' in _df_sel_only.columns:
        with st.expander("CATE vs Actual result — selected races"):
            _cluster_colors = {
                '⏱️  TT':              '#9b59b6',
                '🟢  Flat/Sprint':      '#27ae60',
                '⛰️  Medium mountain': '#e67e22',
                '🏔️  High mountain':   '#e74c3c',
            }
            fig_val = go.Figure()
            _groups = (
                _df_sel_only['stage_cluster_label'].fillna('Unknown')
                if 'stage_cluster_label' in _df_sel_only.columns
                else pd.Series(['All'] * len(_df_sel_only))
            )
            for _cl in _groups.unique():
                _mask = (_groups == _cl).values
                _sub  = _df_sel_only[_mask]
                _hover = (
                    _sub['course_label'] if 'course_label' in _sub.columns
                    else _sub['course']
                )
                fig_val.add_trace(go.Scatter(
                    x=_sub['cate'],
                    y=_sub[_outcome_col],
                    mode='markers',
                    name=str(_cl),
                    text=_hover,
                    hovertemplate='<b>%{text}</b><br>CATE: %{x:+.3f}<br>Actual pts: %{y:.0f}<extra></extra>',
                    marker=dict(
                        size=6, opacity=0.65,
                        color=_cluster_colors.get(_cl, '#2271B3'),
                    ),
                ))
            fig_val.add_vline(x=0, line_dash='dash', line_color='red', opacity=0.4)
            _corr = _df_sel_only[['cate', _outcome_col]].dropna().corr().iloc[0, 1]
            fig_val.add_annotation(
                x=0.02, y=0.98, xref='paper', yref='paper',
                text=f"Pearson r = {_corr:.3f}",
                showarrow=False, font=dict(size=13),
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='#ccc', borderwidth=1,
                xanchor='left', yanchor='top',
            )
            fig_val.update_layout(
                xaxis_title='CATE (predicted marginal contribution)',
                yaxis_title=f'Actual {_outcome_col}',
                template='plotly_white',
                height=420,
            )
            _val_sel = st.plotly_chart(
                fig_val, use_container_width=True,
                on_select='rerun', key='val_scatter',
            )
            st.caption("Click a point to see the race profile and team results.")

            # ── Second scatter: CATE vs Y_resid (race-baseline removed) ──
            if 'Y_resid' in _df_sel_only.columns:
                st.markdown("**CATE vs residual outcome** — race baseline removed")
                st.info(
                    "**Y-axis — how to read it:** "
                    "Ỹᵢ = log(1 + actual\_pts) − ĝ(Xᵢ), where ĝ(Xᵢ) is the model's "
                    "baseline prediction for this race context (profile, startlist quality, classification…). "
                    "A positive value means the team scored *more* than expected; negative means *less*. "
                    "Unlike raw UCI points, this is comparable across all race types — a WT stage "
                    "and a 1.Pro race are on the same scale after the baseline is removed.",
                    icon="ℹ️",
                )
                fig_resid = go.Figure()
                for _cl in _groups.unique():
                    _mask = (_groups == _cl).values
                    _sub  = _df_sel_only[_mask]
                    _hover = _sub['course_label'] if 'course_label' in _sub.columns else _sub['course']
                    fig_resid.add_trace(go.Scatter(
                        x=_sub['cate'],
                        y=_sub['Y_resid'],
                        mode='markers',
                        name=str(_cl),
                        text=_hover,
                        hovertemplate='<b>%{text}</b><br>CATE: %{x:+.3f}<br>Ỹ (residual): %{y:+.3f}<extra></extra>',
                        marker=dict(size=6, opacity=0.65, color=_cluster_colors.get(_cl, '#2271B3')),
                    ))
                fig_resid.add_vline(x=0, line_dash='dash', line_color='red', opacity=0.4)
                fig_resid.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.3)
                _corr_r = _df_sel_only[['cate', 'Y_resid']].dropna().corr().iloc[0, 1]
                fig_resid.add_annotation(
                    x=0.02, y=0.98, xref='paper', yref='paper',
                    text=f"Pearson r = {_corr_r:.3f}",
                    showarrow=False, font=dict(size=13),
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#ccc', borderwidth=1,
                    xanchor='left', yanchor='top',
                )
                fig_resid.update_layout(
                    xaxis_title='CATE (predicted marginal contribution, log scale)',
                    yaxis_title='Ỹ = actual − expected (log scale)',
                    template='plotly_white',
                    height=420,
                )
                _resid_sel = st.plotly_chart(
                    fig_resid, use_container_width=True,
                    on_select='rerun', key='val_scatter_resid',
                )
                st.caption("Click a point to see the race profile and team results.")

                # ── Click handler for residual scatter ───────────────────
                _resid_pts = (_resid_sel.get('selection', {}).get('points') or [])
                if _resid_pts:
                    _rpt = _resid_pts[0]
                    _rmatch = _df_sel_only[
                        (_df_sel_only['cate'].round(3) == round(_rpt.get('x', 0), 3))
                        & (_df_sel_only['Y_resid'].round(3) == round(_rpt.get('y', 0), 3))
                    ]
                    if len(_rmatch) == 0:
                        _rmatch = _df_sel_only.iloc[
                            [(_df_sel_only['cate'] - _rpt.get('x', 0)).abs()
                             .add((_df_sel_only['Y_resid'] - _rpt.get('y', 0)).abs())
                             .argmin()]
                        ]
                    _rrow = _rmatch.iloc[0]
                    _show_course_card(
                        _rrow, df_ref=df_cf_all, features=res1.get('features'),
                        cf_model=res1.get('cf_model'), X_train=res1.get('X'),
                        key_suffix='resid',
                    )
                    _rcourse = _rrow.get('course', '')
                    _ryear   = _rrow.get('year', 0)
                    _rstage  = _rrow.get('stage_num', None)
                    with st.spinner("Loading results…"):
                        _df_results_r = load_race_results(_rcourse, _ryear, _rstage)
                    if _df_results_r is not None and len(_df_results_r) > 0:
                        st.markdown(f"**Results — {_rrow.get('course_label', _rcourse)}**")
                        _df_results_r = _inject_team_uci_pts(_df_results_r, _rrow)
                        _team_only_r = st.toggle("Team only", value=False, key='results_team_only_resid')
                        if _team_only_r:
                            _eq_kw_r = _race_equipe_kw(_rrow)
                            _df_results_r = _df_results_r[
                                _df_results_r['Team'].str.lower().str.contains(_eq_kw_r, regex=False)
                            ]
                        st.dataframe(
                            _df_results_r.style.format({'UCI pts': '{:.0f}', 'Team UCI pts': '{:.0f}'}),
                            use_container_width=True, hide_index=False,
                        )

            # ── Click → course card + team results ──────────────────────
            _val_pts = (_val_sel.get('selection', {}).get('points') or [])
            if _val_pts:
                _vpt = _val_pts[0]
                _vmatch = _df_sel_only[
                    (_df_sel_only['cate'].round(3) == round(_vpt.get('x', 0), 3))
                    & (_df_sel_only[_outcome_col].round(1) == round(_vpt.get('y', 0), 1))
                ]
                if len(_vmatch) == 0:
                    _vmatch = _df_sel_only.iloc[
                        [(_df_sel_only['cate'] - _vpt.get('x', 0)).abs()
                         .add((_df_sel_only[_outcome_col] - _vpt.get('y', 0)).abs())
                         .argmin()]
                    ]
                _vrow = _vmatch.iloc[0]
                _show_course_card(
                    _vrow, df_ref=df_cf_all, features=res1.get('features'),
                    cf_model=res1.get('cf_model'), X_train=res1.get('X'),
                    key_suffix='val',
                )
                _vcourse = _vrow.get('course', '')
                _vyear   = _vrow.get('year', 0)
                _vstage  = _vrow.get('stage_num', None)
                with st.spinner("Loading results…"):
                    _df_results = load_race_results(_vcourse, _vyear, _vstage)
                if _df_results is not None and len(_df_results) > 0:
                    st.markdown(f"**Results — {_vrow.get('course_label', _vcourse)}**")
                    _df_results = _inject_team_uci_pts(_df_results, _vrow)
                    _team_only = st.toggle(
                        "Team only", value=False, key='results_team_only',
                    )
                    if _team_only:
                        _eq_kw_v = _race_equipe_kw(_vrow)
                        _df_show = _df_results[
                            _df_results['Team'].str.lower().str.contains(_eq_kw_v, regex=False)
                        ]
                    else:
                        _df_show = _df_results
                    st.dataframe(
                        _df_show.style.format({'UCI pts': '{:.0f}', 'Team UCI pts': '{:.0f}'}),
                        use_container_width=True,
                        hide_index=False,
                    )
                else:
                    st.caption("No result data found for this stage.")

            # Best and worst predicted races
            _col_a, _col_b = st.columns(2)
            _disp_cols = [c for c in ['course_label', 'year', 'stage_cluster_label', 'cate', _outcome_col]
                          if c in _df_sel_only.columns]
            with _col_a:
                st.markdown("**High CATE + high actual pts** *(model correct)*")
                _df_sel_only['_score'] = (
                    (_df_sel_only['cate'] - _df_sel_only['cate'].mean()) / _df_sel_only['cate'].std()
                    + (_df_sel_only[_outcome_col] - _df_sel_only[_outcome_col].mean()) / _df_sel_only[_outcome_col].std()
                )
                st.dataframe(
                    _df_sel_only.nlargest(8, '_score')[_disp_cols].reset_index(drop=True),
                    use_container_width=True,
                )
            with _col_b:
                st.markdown("**High CATE + low actual pts** *(model overestimated)*")
                _df_sel_only['_miss'] = (
                    (_df_sel_only['cate'] - _df_sel_only['cate'].mean()) / _df_sel_only['cate'].std()
                    - (_df_sel_only[_outcome_col] - _df_sel_only[_outcome_col].mean()) / _df_sel_only[_outcome_col].std()
                )
                st.dataframe(
                    _df_sel_only.nlargest(8, '_miss')[_disp_cols].reset_index(drop=True),
                    use_container_width=True,
                )

            # ── Prediction quality by race type ─────────────────────────
            if 'stage_cluster_label' in _df_sel_only.columns:
                st.markdown("**Prediction quality by race type** *(CATE vs actual result)*")
                _q_rows = []
                for _cl, _g in _df_sel_only.dropna(subset=['cate', _outcome_col, 'stage_cluster_label']).groupby('stage_cluster_label'):
                    _corr = _g[['cate', _outcome_col]].corr().iloc[0, 1] if len(_g) > 2 else float('nan')
                    _q_rows.append({
                        'Race type': _cl,
                        'N': len(_g),
                        'Mean CATE': round(float(_g['cate'].mean()), 3),
                        'Mean actual pts': round(float(_g[_outcome_col].mean()), 1),
                        'Correlation': round(float(_corr), 3),
                    })
                _df_q = pd.DataFrame(_q_rows).sort_values('Correlation', ascending=False).reset_index(drop=True)
                st.dataframe(
                    _df_q.style.format({'Mean CATE': '{:.3f}', 'Mean actual pts': '{:.1f}', 'Correlation': '{:.3f}'}),
                    use_container_width=True,
                )
                st.caption("Correlation between predicted CATE and actual team pts — higher = CATE ranking aligns better with real results for this race type.")

    # ── GATE + QINI evaluation ───────────────────────────────────────────────
    if res1.get('dml') and 'T_resid' in res1['dml'] and 'Y_resid' in res1['dml'] and 'cate' in res1:
        with st.expander("CATE calibration — GATE & QINI curve"):
            st.markdown("""
**GATE** (*Grouped Average Treatment Effects*): observations sorted into quintiles by predicted CATE.
For each quintile, the *actual* average treatment effect is estimated from DML residuals
(Ỹ = Y−ĝ(X), T̃ = T−m̂(X)) via the formula θ̂ = Σ(T̃·Ỹ)/Σ(T̃²).
A well-calibrated model produces a **monotone increasing** bar chart.

**QINI curve**: cumulative benefit of targeting riders by descending CATE order vs random selection.
Higher AUTOC = better targeting efficiency.
""")
            _Yr = np.array(res1['dml']['Y_resid'])
            _Tr = np.array(res1['dml']['T_resid'])
            _cate = np.array(res1['cate'])
            _n = len(_cate)

            # ── GATE ────────────────────────────────────────────────────
            _sort_asc = np.argsort(_cate)
            _Yr_s = _Yr[_sort_asc]
            _Tr_s = _Tr[_sort_asc]

            _n_q = 5
            _gate_vals, _gate_se, _gate_labels, _gate_cate_means = [], [], [], []
            _cate_s = _cate[_sort_asc]

            for _qi, _grp in enumerate(np.array_split(np.arange(_n), _n_q)):
                _Yr_g = _Yr_s[_grp]
                _Tr_g = _Tr_s[_grp]
                _denom = float(np.sum(_Tr_g ** 2))
                if _denom < 1e-10:
                    continue
                _th = float(np.dot(_Tr_g, _Yr_g)) / _denom
                # Sandwich SE
                _psi = _Tr_g * _Yr_g - _th * _Tr_g ** 2
                _se = float(np.sqrt(np.sum(_psi ** 2))) / _denom
                _gate_vals.append(_th)
                _gate_se.append(_se)
                _gate_labels.append(f"Q{_qi+1}")
                _gate_cate_means.append(float(_cate_s[_grp].mean()))

            _gate_colors = ['#27ae60' if v > 0 else '#c0392b' for v in _gate_vals]
            fig_gate = go.Figure()
            fig_gate.add_trace(go.Bar(
                x=_gate_labels, y=_gate_vals,
                error_y=dict(type='data', array=[1.96*s for s in _gate_se], visible=True,
                             color='#555', thickness=1.5, width=6),
                marker_color=_gate_colors,
                hovertemplate='%{x}: ATE = %{y:+.4f}<extra></extra>',
            ))
            fig_gate.add_hline(y=0, line_color='#888', line_width=1)
            fig_gate.update_layout(
                title='GATE — actual effect per CATE quintile (log scale)',
                xaxis_title='CATE quintile (Q1=lowest predicted → Q5=highest)',
                yaxis_title='Estimated ATE (log pts)',
                template='plotly_white', height=350,
                annotations=[dict(
                    x=_gate_labels[i], y=_gate_vals[i] + 1.96*_gate_se[i] + 0.005,
                    text=f"τ̂={_gate_cate_means[i]:+.3f}",
                    showarrow=False, font=dict(size=9, color='#555'), yanchor='bottom',
                ) for i in range(len(_gate_labels))],
            )
            st.plotly_chart(fig_gate, use_container_width=True, key='gate_plot')
            st.caption("Error bars = 95% CI. τ̂ above each bar = mean predicted CATE in that quintile. "
                       "Monotone increasing = CATE correctly ranks races by causal impact.")

            # ── QINI curve ──────────────────────────────────────────────
            _sort_desc = np.argsort(-_cate)
            _Yr_d = _Yr[_sort_desc]
            _Tr_d = _Tr[_sort_desc]
            _run_cov = np.cumsum(_Tr_d * _Yr_d)
            _run_var = np.cumsum(_Tr_d ** 2)
            _run_ate = _run_cov / np.maximum(_run_var, 1e-10)
            _fracs = np.arange(1, _n + 1) / _n
            _qini = _run_ate * _fracs

            _overall_ate = float(np.dot(_Tr, _Yr)) / max(float(np.sum(_Tr ** 2)), 1e-10)
            _baseline = _overall_ate * _fracs
            _autoc = float(np.trapz(_qini - _baseline, _fracs))
            _autoc_norm = _autoc / abs(_overall_ate) if abs(_overall_ate) > 1e-10 else float('nan')

            # Normalise so both curves end at 1 (random = diagonal)
            _norm_val = _overall_ate if abs(_overall_ate) > 1e-10 else 1.0
            _qini_n    = _qini    / _norm_val
            _baseline_n = _fracs  # = _baseline / _norm_val = fraction (diagonal)

            # Downsample for plot (max 300 pts)
            _step = max(1, _n // 300)
            _fx = _fracs[::_step]
            _fy = _qini_n[::_step]
            _by = _baseline_n[::_step]

            fig_qini = go.Figure()
            fig_qini.add_trace(go.Scatter(
                x=_fx, y=_fy, mode='lines', name='CATE ordering',
                line=dict(color='#2271B3', width=2),
            ))
            fig_qini.add_trace(go.Scatter(
                x=_fx, y=_by, mode='lines', name='Random ordering',
                line=dict(color='#888', dash='dash', width=1.5),
            ))
            fig_qini.add_annotation(
                x=0.05, y=0.95, xref='paper', yref='paper',
                text=f"AUTOC = {_autoc_norm:.3f}",
                showarrow=False, font=dict(size=13),
                bgcolor='rgba(255,255,255,0.85)', bordercolor='#ccc', borderwidth=1,
                xanchor='left', yanchor='top',
            )
            fig_qini.update_layout(
                title='QINI curve — targeting efficiency',
                xaxis_title='Fraction of observations targeted (by descending CATE)',
                yaxis_title='Normalised cumulative benefit (1 = overall ATE)',
                template='plotly_white', height=360,
                legend=dict(x=0.6, y=0.05),
            )
            st.plotly_chart(fig_qini, use_container_width=True, key='qini_plot')
            st.caption(
                f"**AUTOC = {_autoc_norm:.3f}** — area between QINI curve and random baseline "
                "(diagonal), normalised by overall ATE. "
                "AUTOC > 0 = CATE ordering beats random selection; "
                "AUTOC = 1 = perfect targeting (all benefit concentrated in top observations)."
            )

    # ── CATE by year / month ─────────────────────────────────────────────────
    gran_cf = st.radio(
        "Granularity", ["Year", "Year-Month"],
        horizontal=True, key="gran_cf",
    )
    if gran_cf == "Year-Month" and 'date' in df_cf1.columns:
        df_cf1_g = df_cf1.copy()
        df_cf1_g['_gk'] = pd.to_datetime(df_cf1_g['date'], errors='coerce').dt.to_period('M').astype(str)
        gk_label = 'Month'
    else:
        df_cf1_g = df_cf1.copy()
        df_cf1_g['_gk'] = df_cf1_g['year'].astype(int).astype(str)
        gk_label = 'Year'

    st.subheader(f"Median CATE by {gk_label.lower()}")
    cate_by_year = (
        df_cf1_g.groupby('_gk')['cate']
        .agg(median='median', mean='mean', n='count')
        .reset_index()
        .sort_values('_gk')
    )
    if gran_cf == "Year":
        cate_by_year['leader'] = cate_by_year.apply(
            lambda r: next(
                (fmt_rider(v) for (t, y), v in leaders.items()
                 if int(y) == int(r['_gk']) and t in (p['teams1'] if p else teams1)),
                '—'
            ),
            axis=1,
        )
        hover_extra = {'mean': ':.3f', 'n': True, 'leader': True}
    else:
        hover_extra = {'mean': ':.3f', 'n': True}

    fig_yr = px.bar(
        cate_by_year, x='_gk', y='median',
        color='median',
        color_continuous_scale=['#E8824B', '#BDBDBD', '#2271B3'],
        color_continuous_midpoint=0,
        hover_data=hover_extra,
        labels={'median': 'Median CATE', '_gk': gk_label},
        title=f"Median CATE by {gk_label.lower()}",
        template='plotly_white',
    )
    fig_yr.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.4)
    fig_yr.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_yr, use_container_width=True)

    # ── Summary by tier ────────────────────────────────────────────────────
    if x_col == 'denivele_pos':
        bins, labels_ = [0, 1000, 2500, 4000, 99999], ['<1000m', '1000–2500m', '2500–4000m', '>4000m']
    elif x_col == 'distance_gpx_km':
        bins, labels_ = [0, 100, 180, 250, 9999], ['<100km', '100–180km', '180–250km', '>250km']
    elif x_col == 'startlist_quality':
        bins, labels_ = [0, 300, 600, 900, 99999], ['<300', '300–600', '600–900', '>900']
    else:
        bins, labels_ = None, None

    if bins:
        st.subheader(f"Median CATE by {x_col} tier")
        df_cf1['tier'] = pd.cut(df_cf1[x_col], bins=bins, labels=labels_)
        summary = (df_cf1.groupby('tier', observed=True)['cate']
                   .agg(median='median', mean='mean', n='count').round(3))
        st.dataframe(summary, use_container_width=True)

    # ── Missed opportunities & wasted selections ───────────────────────────
    st.subheader("Selection error analysis")
    with st.expander("ℹ️ How to read these tables?"):
        st.markdown("""
**The CATE measures a marginal effect** — not "how many points would he have scored", but
"how many points MORE compared to what the team does without him".

Examples for Van Aert:
- **Omloop 2023**: Laporte won (547 pts) without Van Aert → low CATE (1.5 marginal pt)
- **Ronde 2022**: team at 190 pts without him → moderate CATE (~1.2 pt)
- **Heist-op-den-Berg**: team at 3 pts without him → high CATE (1.6 pt) because he would change everything

➜ "Missed" classics DO appear in the list, but often not at the top
because the team already gets good results without him on these races (Laporte, Benoot, etc.).

**Common issue — year contamination**: if you include years where the rider was
on another team (e.g. Van Aert 2018 at Vérandas Willems), the model mixes two
very different contexts → incorrect results. **Set the slider to the year he joined the team.**
        """)

    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        top_n = st.slider("Top N races", min_value=5, max_value=30, value=10, step=5)

    all_clusters = sorted(df_cf_all['stage_cluster_label'].dropna().unique().tolist()) if 'stage_cluster_label' in df_cf_all.columns else []
    if all_clusters:
        with ctrl_col2:
            selected_clusters = st.multiselect(
                "Filter by race type",
                options=all_clusters,
                default=all_clusters,
                help="Uncheck a type to exclude it from both tables "
                     "(e.g. remove mountain stages for a classics rider).",
            )
    else:
        selected_clusters = all_clusters

    team_present_filter = st.checkbox(
        "Only races where the team was present (team UCI pts > 0)",
        value=True,
        help="Filters races where the team scored at least 1 pt without this rider — "
             "indicates the team was competing and the rider could have been there.",
    )

    def _filter_pool(df):
        if selected_clusters and 'stage_cluster_label' in df.columns:
            df = df[df['stage_cluster_label'].isin(selected_clusters)]
        if team_present_filter and 'pts_uci_equipe_stage' in df.columns:
            df = df[df['pts_uci_equipe_stage'] > 0]
        return df

    opp_cols = ['course_label', 'year', 'stage_cluster_label', 'cate',
                'cate_pct_max', x_col, 'startlist_quality', 'pts_uci_equipe_stage']
    _seen = set()
    opp_cols = [c for c in opp_cols if c in df_cf_all.columns and c not in _seen and not _seen.add(c)]

    col_opp, col_waste = st.columns(2)
    with col_opp:
        st.markdown("**Missed opportunities** *(not selected, high CATE)*")
        df_missed = _filter_pool(df_cf_all[df_cf_all['selected'] == 0]).nlargest(top_n, 'cate').reset_index(drop=True)
        sel_missed = st.dataframe(
            df_missed[opp_cols], use_container_width=True,
            on_select='rerun', selection_mode='single-row', key='tbl_missed',
        )
        rows_missed = sel_missed.get('selection', {}).get('rows', [])

    with col_waste:
        st.markdown("**Wasted selections** *(selected, low CATE)*")
        df_wasted = _filter_pool(df_cf_all[df_cf_all['selected'] == 1]).nsmallest(top_n, 'cate').reset_index(drop=True)
        sel_wasted = st.dataframe(
            df_wasted[opp_cols], use_container_width=True,
            on_select='rerun', selection_mode='single-row', key='tbl_wasted',
        )
        rows_wasted = sel_wasted.get('selection', {}).get('rows', [])

    # Course cards shown OUTSIDE columns so they use full page width
    if rows_missed:
        _show_course_card(df_missed.iloc[rows_missed[0]], df_ref=df_cf_all, features=res1.get('features'),
                          cf_model=res1.get('cf_model'), X_train=res1.get('X'), key_suffix='missed')
    if rows_wasted:
        _show_course_card(df_wasted.iloc[rows_wasted[0]], df_ref=df_cf_all, features=res1.get('features'),
                          cf_model=res1.get('cf_model'), X_train=res1.get('X'), key_suffix='wasted')

    # ── Variable importance (Causal Forest) ──────────────────────────────
    if 'cf_model' in res1:
        st.subheader("Variable importance — Causal Forest")
        feats = res1['features']
        imps = res1['cf_model'].feature_importances_
        df_imp = (
            pd.DataFrame({'feature': feats, 'importance': imps})
            .sort_values('importance', ascending=True)
        )
        _dml = res1.get('dml') or {}
        _ate = _dml.get('ate_orig', float('nan'))
        _ci_lo = _dml.get('ci_low', float('nan'))
        _ci_hi = _dml.get('ci_high', float('nan'))
        _sig = "✓" if _dml.get('significant') else "✗"
        _mean_cate = float(df_cf_all['cate'].mean()) if 'cate' in df_cf_all.columns else float('nan')
        _n_obs = res1.get('n_obs', '?')
        _n_sel = res1.get('n_selected', '?')
        _imp_title = (
            f"<b>{label1}</b><br>"
            f"ATE = <b>{_ate:+.2f} UCI pts</b>  |  "
            f"95% CI [{_ci_lo:+.2f}, {_ci_hi:+.2f}]  |  "
            f"Significant: {'✓' if _dml.get('significant') else '✗'}  |  "
            f"Mean CATE = <b>{_mean_cate:+.2f} pts</b>  |  "
            f"N = {_n_obs} stages ({_n_sel} selected)<br>"
            f"<i>Which variables best explain CATE heterogeneity?</i>"
        )
        fig_imp = go.Figure(go.Bar(
            x=df_imp['importance'],
            y=df_imp['feature'],
            orientation='h',
            marker_color='#2271B3',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
        ))
        fig_imp.update_layout(
            title=dict(text=_imp_title, x=0, xanchor='left', font=dict(size=14)),
            xaxis_title="Importance (Causal Forest)",
            template='plotly_white',
            height=max(350, len(feats) * 26),
            margin=dict(l=160, t=120, b=50, r=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Importance measures how much each variable is used to create splits "
            "in the Causal Forest — variables at the top influence the heterogeneous "
            "effect of including the rider the most."
        )

        # ── CATE by feature quartile ──────────────────────────────────────────
        _feat_labels = {
            'denivele_pos': 'D+ (m)', 'denivele_neg': 'D− (m)',
            'distance_gpx_km': 'Distance (km)', 'startlist_quality': 'Startlist quality',
            'n_cols_hc': 'HC climbs', 'n_cols_cat1': 'Cat1 climbs',
            'n_cols_cat2': 'Cat2 climbs', 'n_cols_cat3': 'Cat3 climbs',
            'n_cols_cat4': 'Cat4 climbs', 'cobblestones_km': 'Cobblestones (km)',
            'compacted_gravel_km': 'Gravel (km)', 'forme_equipe': 'Team form',
            'n_races_30d': 'Races/30d', 'km_30d': 'Km/30d',
            'is_team_leader': 'Team leader', 'leader_played': 'Leader present',
            'gradient_last_5km': 'Final gradient (5km)',
            'altitude_max': 'Max altitude (m)', 'altitude_min': 'Min altitude (m)',
            'loc_last_col_hc': 'Last HC climb position',
            'loc_last_col_cat1': 'Last Cat1 climb position',
            'deniv_last_5km': 'D+ last 5km',
            'top_score_in_team': 'Top scorer in team',
            'forme_coureur': 'Rider form', 'year': 'Year',
        }
        top_feats_ordered = df_imp.sort_values('importance', ascending=False)['feature'].tolist()
        sel_feat = st.selectbox(
            "Explore CATE heterogeneity for a feature:",
            options=top_feats_ordered,
            format_func=lambda f: _feat_labels.get(f, f),
            key='hetero_feat_sel',
        )
        if sel_feat and sel_feat in df_cf_all.columns:
            _df_q = df_cf_all[['cate', sel_feat]].dropna()
            if len(_df_q) >= 8:
                try:
                    _df_q = _df_q.copy()
                    _feat_lbl = _feat_labels.get(sel_feat, sel_feat)
                    _n_unique = _df_q[sel_feat].nunique()
                    _is_binary = _n_unique <= 5

                    if _is_binary:
                        # Binary / low-cardinality: group by actual values
                        _binary_labels = {1.0: 'Yes', 0.0: 'No', 1: 'Yes', 0: 'No'}
                        _df_q['_q'] = _df_q[sel_feat].map(
                            lambda v: _binary_labels.get(v, str(v))
                        )
                        _q_stats = (
                            _df_q.groupby('_q', observed=True)['cate']
                            .agg(mean='mean', std='std', n='count')
                            .reset_index()
                        )
                        _q_stats['sem95'] = 1.96 * _q_stats['std'] / _q_stats['n'].pow(0.5)
                        _q_labels = _q_stats['_q'].tolist()
                        _chart_title = f'Average CATE by {_feat_lbl}'
                    else:
                        # Continuous: quartile binning
                        _df_q['_q'] = pd.qcut(_df_q[sel_feat], q=4, duplicates='drop')
                        _q_stats = (
                            _df_q.groupby('_q', observed=True)['cate']
                            .agg(mean='mean', std='std', n='count')
                            .reset_index()
                        )
                        _q_stats['sem95'] = 1.96 * _q_stats['std'] / _q_stats['n'].pow(0.5)
                        def _fmt_bound(v):
                            av = abs(v)
                            if av >= 1e9:  return f'{v/1e9:.1f}B'
                            if av >= 1e6:  return f'{v/1e6:.1f}M'
                            if av >= 1e3:  return f'{v/1e3:.1f}k'
                            if av >= 100:  return f'{v:.0f}'
                            if av >= 1:    return f'{v:.1f}'
                            return f'{v:.2f}'
                        _q_labels = [
                            f"Q{i+1}: [{_fmt_bound(iv.left)} – {_fmt_bound(iv.right)}]"
                            for i, iv in enumerate(_q_stats['_q'])
                        ]
                        _chart_title = f'Average CATE by {_feat_lbl} quartile'

                    fig_q = go.Figure(go.Bar(
                        x=_q_labels,
                        y=_q_stats['mean'].round(3),
                        error_y=dict(type='data', array=_q_stats['sem95'].round(3), visible=True),
                        marker_color=[
                            '#2d7a3a' if v > 0 else '#c0392b'
                            for v in _q_stats['mean']
                        ],
                        text=_q_stats['n'].astype(int).astype(str) + ' obs',
                        textposition='outside',
                        hovertemplate='%{x}<br>Avg CATE: %{y:+.3f} pts<extra></extra>',
                    ))
                    fig_q.add_hline(y=0, line_color='#888', line_width=1)
                    fig_q.update_layout(
                        title=_chart_title,
                        yaxis_title='Average CATE (UCI pts)',
                        xaxis_title=_feat_lbl,
                        template='plotly_white',
                        height=350,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_q, use_container_width=True)
                except Exception:
                    st.caption("Not enough distinct values to split into quartiles.")

with tab_cf:
    _render_cf()

# ════════════════════════════════════════════════════════════════════════════════
# TAB — WORLDTOUR RANKINGS
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_ate_results():
    path = Path(cm.BASE_DIR) / 'riders_ate_results.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['ate_orig'] = pd.to_numeric(df['ate_orig'], errors='coerce')
    df['ci_low']   = pd.to_numeric(df['ci_low'],   errors='coerce')
    df['ci_high']  = pd.to_numeric(df['ci_high'],  errors='coerce')
    df['significant'] = df['significant'].astype(int)
    return df.dropna(subset=['ate_orig', 'ci_low', 'ci_high'])


with tab_rank:
    df_rank = _load_ate_results()
    if df_rank is None:
        st.info("No ranking data yet. Run `python precompute_rankings.py` to generate it.")
    else:
        st.subheader("WorldTour Rankings — ATE (Average Treatment Effect)")
        st.caption(
            "ATE = average causal contribution to team UCI points per stage when the rider is selected. "
            "Estimated via Double Machine Learning (DML) on the rider's full career data."
        )

        # ── Filters ──────────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns([3, 1, 1, 1])
        with fc1:
            all_teams = sorted(df_rank['equipe'].dropna().unique())
            sel_teams = st.multiselect("Filter by team", options=all_teams, default=[], key='rank_teams')
        with fc2:
            sig_only = st.toggle("Significant only", value=False, key='rank_sig')
        with fc3:
            min_obs = st.number_input("Min stages", min_value=50, max_value=1000, value=100, step=50, key='rank_minobs')
        with fc4:
            top_n = st.slider("Top N", min_value=10, max_value=100, value=30, step=5, key='rank_topn')

        # ── Filter data ───────────────────────────────────────────────────────
        df_f = df_rank.copy()
        if sel_teams:
            df_f = df_f[df_f['equipe'].isin(sel_teams)]
        if sig_only:
            df_f = df_f[df_f['significant'] == 1]
        df_f = df_f[df_f['n_obs'] >= min_obs]
        # Remove outliers with exploding CIs (keep riders where CI width < 99th percentile)
        ci_width = df_f['ci_high'] - df_f['ci_low']
        ci_cap = ci_width.quantile(0.99)
        df_f = df_f[ci_width <= ci_cap]
        df_f = df_f.sort_values('ate_orig', ascending=False).head(top_n).reset_index(drop=True)

        if len(df_f) == 0:
            st.warning("No riders match the current filters.")
        else:
            # ── Bar chart with CI ─────────────────────────────────────────────
            df_plot = df_f.sort_values('ate_orig', ascending=True)
            colors  = ['#2271B3' if s else '#aac4e0' for s in df_plot['significant']]
            # capitalize() handles accented chars correctly (title() does not)
            rider_labels = df_plot['rider'].apply(
                lambda s: ' '.join(w.capitalize() for w in s.split('_'))
            )

            fig_rank = go.Figure()
            fig_rank.add_trace(go.Bar(
                x=df_plot['ate_orig'],
                y=rider_labels,
                orientation='h',
                marker_color=colors,
                error_x=dict(
                    type='data',
                    arrayminus=df_plot['ate_orig'] - df_plot['ci_low'],
                    array=df_plot['ci_high'] - df_plot['ate_orig'],
                    visible=True,
                    color='#555',
                    thickness=1.2,
                    width=3,
                ),
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    'ATE = %{x:+.3f} UCI pts<br>'
                    'CI: [%{customdata[0]:+.2f}, %{customdata[1]:+.2f}]<br>'
                    'N = %{customdata[2]} stages<extra></extra>'
                ),
                customdata=df_plot[['ci_low', 'ci_high', 'n_obs']].values,
            ))
            fig_rank.add_vline(x=0, line_color='#333', line_width=1)
            fig_rank.update_layout(
                title=dict(
                    text=(
                        f"Top {len(df_plot)} riders by ATE"
                        + (f" — {', '.join(sel_teams)}" if sel_teams else " — All WorldTour teams")
                        + ("<br><sup>Dark blue = significant (95% CI excludes 0). Light blue = non-significant.</sup>" )
                    ),
                    x=0, xanchor='left', font=dict(size=14),
                ),
                xaxis_title="ATE (UCI pts / stage)",
                template='plotly_white',
                height=max(400, len(df_plot) * 22),
                margin=dict(l=200, t=80, b=50, r=20),
            )
            st.plotly_chart(fig_rank, use_container_width=True)

            # ── Table ─────────────────────────────────────────────────────────
            with st.expander("Full table", expanded=False):
                df_table = df_f[['rider', 'equipe', 'ate_orig', 'ci_low', 'ci_high', 'significant', 'r2_t', 'r2_y', 'n_obs', 'n_selected']].copy()
                df_table['rider'] = df_table['rider'].apply(
                    lambda s: ' '.join(w.capitalize() for w in s.split('_'))
                )
                df_table['sig'] = df_table['significant'].map({1: '✓', 0: '✗'})
                df_table = df_table.rename(columns={
                    'rider': 'Rider', 'equipe': 'Team',
                    'ate_orig': 'ATE', 'ci_low': 'CI low', 'ci_high': 'CI high',
                    'sig': 'Sig', 'r2_t': 'R² T', 'r2_y': 'R² Y',
                    'n_obs': 'N stages', 'n_selected': 'N selected',
                }).drop(columns=['significant'])
                st.dataframe(
                    df_table.style.format({'ATE': '{:+.3f}', 'CI low': '{:+.2f}', 'CI high': '{:+.2f}', 'R² T': '{:.3f}', 'R² Y': '{:.3f}'}),
                    use_container_width=True, hide_index=True,
                )

    # ── Team UCI Points Ranking ───────────────────────────────────────────────
    st.divider()
    st.subheader("Team UCI Points Ranking")
    st.caption("Total UCI points accumulated by team, filterable by race classification and year range.")

    @st.cache_data(show_spinner=False)
    def _load_team_pts_all():
        path = Path(cm.BASE_DIR) / 'team_stage_points.csv'
        if not path.exists():
            return None
        df = pd.read_csv(path, low_memory=False)
        df['pts_uci'] = pd.to_numeric(df['pts_uci'], errors='coerce').fillna(0)
        df['year']    = pd.to_numeric(df['year'], errors='coerce')
        return df

    _CLASS_GROUPS = {
        'WorldTour (UWT)': ['1.UWT', '2.UWT'],
        'Pro Series':       ['1.Pro', '2.Pro'],
        'HC races':         ['1.HC', '2.HC'],
        'National Champs':  ['NC'],
        'World Champs':     ['WC'],
        'Cat 1':            ['1.1', '1.2', '1.2U', '1.Ncup'],
        'Cat 2':            ['2.1', '2.2', '2.2U', '2.Ncup', 'CC'],
    }

    df_tpts = _load_team_pts_all()
    if df_tpts is None:
        st.info("No team points data available (team_stage_points.csv not found).")
    else:
        tc1, tc2, tc3 = st.columns([2, 2, 1])
        with tc1:
            class_options = ['All'] + list(_CLASS_GROUPS.keys()) + ['Custom']
            sel_class = st.selectbox("Race classification", class_options, key='rank_class')
        with tc2:
            all_years = sorted(df_tpts['year'].dropna().unique().astype(int))
            yr_range = st.select_slider(
                "Year range", options=all_years,
                value=(min(all_years), max(all_years)), key='rank_yr_team'
            )
        with tc3:
            top_n_team = st.slider("Top N teams", 10, 50, 20, 5, key='rank_topn_team')

        if sel_class == 'Custom':
            all_classes = sorted(df_tpts['classification'].dropna().unique())
            custom_classes = st.multiselect("Select classifications", all_classes, default=['1.UWT'], key='rank_custom_class')
            class_filter = custom_classes
        elif sel_class == 'All':
            class_filter = None
        else:
            class_filter = _CLASS_GROUPS[sel_class]

        df_tpts_f = df_tpts[df_tpts['year'].between(yr_range[0], yr_range[1])]
        if class_filter:
            df_tpts_f = df_tpts_f[df_tpts_f['classification'].isin(class_filter)]

        df_team_rank = (
            df_tpts_f.groupby('Team', as_index=False)['pts_uci']
            .sum()
            .sort_values('pts_uci', ascending=False)
            .head(top_n_team)
            .reset_index(drop=True)
        )
        df_team_rank.index += 1

        if len(df_team_rank) == 0:
            st.warning("No data for the selected filters.")
        else:
            df_plot_t = df_team_rank.sort_values('pts_uci', ascending=True)
            fig_team = go.Figure(go.Bar(
                x=df_plot_t['pts_uci'],
                y=df_plot_t['Team'],
                orientation='h',
                marker_color='#2271B3',
                hovertemplate='<b>%{y}</b><br>%{x:,.0f} UCI pts<extra></extra>',
            ))
            class_label = sel_class if sel_class != 'Custom' else ', '.join(class_filter or [])
            fig_team.update_layout(
                title=dict(
                    text=f"Top {len(df_plot_t)} teams by UCI points — {class_label} ({yr_range[0]}–{yr_range[1]})",
                    x=0, xanchor='left', font=dict(size=14),
                ),
                xaxis_title="Total UCI pts",
                template='plotly_white',
                height=max(400, len(df_plot_t) * 22),
                margin=dict(l=200, t=60, b=50, r=20),
            )
            st.plotly_chart(fig_team, use_container_width=True)

            with st.expander("Full table", expanded=False):
                df_team_rank.insert(0, 'Rank', range(1, len(df_team_rank) + 1))
                st.dataframe(
                    df_team_rank.rename(columns={'pts_uci': 'UCI pts'})
                    .style.format({'UCI pts': '{:,.0f}'}),
                    use_container_width=True, hide_index=True,
                )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 0 — DESCRIPTIVE STATS (model-independent)
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_team_stage_pts(equipe_tuple, years):
    path = Path(cm.BASE_DIR) / 'team_stage_points.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df['Team'].isin(list(equipe_tuple))]
    if years:
        df = df[df['year'].between(years[0], years[1])]
    return df if len(df) > 0 else None


_CLUSTER_ORDER = ['⏱️  TT', '🟢  Flat/Sprint', '⛰️  Medium mountain', '🏔️  High mountain']

@st.cache_data(show_spinner=False)
def cached_leader_per_cluster(equipe_tuple, years):
    """For each (year, cluster), return the rider with the most UCI pts."""
    roster = cached_roster(equipe_tuple)
    if roster is None or len(roster) == 0:
        return None

    rows = []
    for rider in roster['rider'].unique():
        df = cm.load_rider(rider, equipe=list(equipe_tuple), years=years)
        if df is None:
            continue
        df_sel = df[(df['selected'] == 1) & df['stage_cluster_label'].notna()]
        if 'pts_uci' not in df_sel.columns or len(df_sel) == 0:
            continue
        agg = (
            df_sel.groupby(['year', 'stage_cluster_label'])['pts_uci']
            .sum()
            .reset_index()
        )
        agg['rider'] = rider
        rows.append(agg)

    if not rows:
        return None

    all_pts = pd.concat(rows, ignore_index=True)
    # Keep the best per (year, cluster)
    idx     = all_pts.groupby(['year', 'stage_cluster_label'])['pts_uci'].idxmax()
    leaders = all_pts.loc[idx].copy()
    leaders['label'] = (
        leaders['rider']
        + ' ('
        + leaders['pts_uci'].round(0).astype(int).astype(str)
        + ')'
    )
    pivot = (
        leaders
        .pivot(index='year', columns='stage_cluster_label', values='label')
        .reindex(columns=[c for c in _CLUSTER_ORDER if c in leaders['stage_cluster_label'].unique()])
        .sort_index()
    )
    return pivot


def _render_stats():
    tab_eq, tab_rider = st.tabs(["👥 Team", "🚴 Rider"])

    # ══════════════════════════════════════════════════════════════════════════
    with tab_eq:
        if not teams1:
            st.info("Select a team in the sidebar.")
        else:
            # ── Roster by year ──────────────────────────────────────────────
            st.subheader("Roster by year")
            active_years = None  # will be updated by the pills selector
            roster = cached_roster(tuple(sorted(teams1)))
            if roster is not None and len(roster):
                roster_disp = roster.copy()
                roster_disp['rider'] = roster_disp['rider'].apply(fmt_rider)
                pivot = roster_disp.pivot_table(
                    index='rider', columns='year', values='n_sel',
                    aggfunc='sum', fill_value=0
                ).astype(int)
                pivot.columns = [str(c) for c in pivot.columns]
                pivot['Total'] = pivot.sum(axis=1)
                pivot = pivot.sort_values('Total', ascending=False)

                all_years = [str(y) for y in range(2018, 2026)]
                team_years = [c for c in pivot.columns if c != 'Total']

                # Green pills for years with data, grey for the others
                st.markdown("""
                <style>
                div[data-testid="stPills"] button[aria-pressed="true"]  { background:#2d7a3a !important; color:white !important; border-color:#2d7a3a !important; }
                div[data-testid="stPills"] button[aria-pressed="false"] { background:#e8e8e8 !important; color:#888 !important; border-color:#e8e8e8 !important; }
                </style>""", unsafe_allow_html=True)

                sel_years = st.pills(
                    "Years", all_years, selection_mode="multi",
                    default=team_years, key="roster_years",
                    format_func=lambda y: y if y in team_years else f"{y} —",
                )

                active = [y for y in (sel_years or []) if y in team_years]
                active_years = [int(y) for y in active] or None
                if active:
                    df_yr = pivot[active].copy()
                    df_yr = df_yr[df_yr.max(axis=1) > 0]
                    df_yr.columns.name = None
                    df_yr['Total'] = df_yr.sum(axis=1)
                    df_yr = df_yr.sort_values('Total', ascending=False)
                    year_cols_active = [c for c in df_yr.columns if c != 'Total']
                    styled = df_yr.style.apply(
                        lambda col: ['background-color: #f0f0f0; color: #bbb' if v == 0 else '' for v in col],
                        subset=year_cols_active,
                    )
                    st.dataframe(styled, use_container_width=True)
                    st.caption(
                        f"{len(df_yr)} riders with ≥1 selection in {', '.join(active)}."
                    )
                else:
                    st.info("Select at least one year.")

            # ── Team stats from team_stage_points ─────────────────────────
            stats_years = (min(active_years), max(active_years)) if active_years else years
            df_team = _load_team_stage_pts(tuple(sorted(teams1)), stats_years)
            if df_team is not None:
                st.divider()
                st.subheader("Team performance")

                # Global metrics
                total_wins = int(df_team['n_wins'].sum())
                total_top5 = int(df_team['n_top5'].sum())
                total_pts  = df_team['pts_uci'].sum()
                best_yr    = df_team.groupby('year')['pts_uci'].sum().idxmax()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total wins", total_wins)
                c2.metric("Total top 5s", total_top5)
                c3.metric("Total team UCI pts", f"{total_pts:.0f}")
                c4.metric("Best season", int(best_yr))

                # UCI pts + wins by year / month
                gran_team = st.radio(
                    "Granularity", ["Year", "Year-Month"],
                    horizontal=True, key="gran_team",
                )
                if gran_team == "Year-Month":
                    df_team_g = df_team.copy()
                    df_team_g['_gk'] = pd.to_datetime(
                        df_team_g['date'], dayfirst=True, errors='coerce'
                    ).dt.to_period('M').astype(str)
                    x_t = '_gk'
                    x_t_label = 'Month'
                else:
                    df_team_g = df_team.copy()
                    df_team_g['_gk'] = df_team_g['year'].astype(int).astype(str)
                    x_t = '_gk'
                    x_t_label = 'Year'

                yr_team = df_team_g.groupby('_gk').agg(
                    pts_uci=('pts_uci', 'sum'),
                    wins=('n_wins', 'sum'),
                    top5=('n_top5', 'sum'),
                ).reset_index().sort_values('_gk')

                fig_team = go.Figure()
                fig_team.add_trace(go.Bar(
                    x=yr_team['_gk'], y=yr_team['pts_uci'],
                    name='Team UCI pts', marker_color='#E8824B',
                ))
                fig_team.update_layout(
                    barmode='group', template='plotly_white',
                    yaxis_title='UCI pts', xaxis_title=x_t_label, height=320,
                )
                st.plotly_chart(fig_team, use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    fig_v = px.bar(
                        yr_team, x='_gk', y='wins', text='wins',
                        title=f'Wins by {x_t_label.lower()}', template='plotly_white',
                        color_discrete_sequence=['#2271B3'], height=280,
                        labels={'_gk': x_t_label},
                    )
                    fig_v.update_traces(textposition='outside')
                    fig_v.update_layout(showlegend=False)
                    st.plotly_chart(fig_v, use_container_width=True)
                with col_b:
                    fig_t5 = px.bar(
                        yr_team, x='_gk', y='top5', text='top5',
                        title=f'Top 5s by {x_t_label.lower()}', template='plotly_white',
                        color_discrete_sequence=['#43A5A0'], height=280,
                        labels={'_gk': x_t_label},
                    )
                    fig_t5.update_traces(textposition='outside')
                    fig_t5.update_layout(showlegend=False)
                    st.plotly_chart(fig_t5, use_container_width=True)

                # % max pts by race type
                if 'classification' in df_team.columns:
                    st.subheader("Team UCI pts — by race category")
                    cat_agg = df_team.groupby('classification').agg(
                        pts_uci=('pts_uci', 'sum'),
                        courses=('race_name', 'count'),
                        wins=('n_wins', 'sum'),
                    ).reset_index()
                    cat_agg['_rank'] = cat_agg['classification'].map(CLASSIFICATION_RANK).fillna(99)
                    cat_agg = cat_agg.sort_values('_rank').drop(columns='_rank')
                    cat_agg['classification'] = cat_agg['classification'].apply(fmt_classification)
                    st.dataframe(cat_agg.head(15), use_container_width=True)

            # ── Leader by cluster by year ──────────────────────────
            st.divider()
            st.subheader("Leader by race type (rider UCI pts, selected)")
            with st.spinner("Loading rider data…"):
                df_leaders = cached_leader_per_cluster(tuple(sorted(teams1)), years)
            if df_leaders is not None and len(df_leaders) > 0:
                def _fmt_leader_cell(cell):
                    if not isinstance(cell, str) or ' (' not in cell:
                        return cell
                    raw, pts = cell.rsplit(' (', 1)
                    return fmt_rider(raw) + ' (' + pts
                df_leaders_disp = df_leaders.map(_fmt_leader_cell)
                st.dataframe(df_leaders_disp, use_container_width=True)
                st.caption(
                    "For each year and stage type, the rider who accumulated "
                    "the most personal UCI pts on selected races. "
                    "Number in parentheses = the rider's total UCI pts in that category."
                )
            else:
                st.info("Not enough data to compute leaders by cluster.")

    # ══════════════════════════════════════════════════════════════════════════
    with tab_rider:
        if rider1 is None:
            st.info("Select a rider in the sidebar.")
            return

        df_raw = cached_load_raw(
            rider1,
            tuple(sorted(teams1)) if teams1 else tuple(get_rider_teams(rider1)),
            years,
        )
        if df_raw is None:
            st.warning("No data found for this rider.")
            return

        df_sel = df_raw[df_raw['selected'] == 1]
        st.subheader(f"Stats — {fmt_rider(rider1)}")

        # ── Teams by year ─────────────────────────────────────────────
        if 'equipe' in df_raw.columns and 'year' in df_raw.columns:
            teams_by_year = (
                df_raw.groupby('year')['equipe']
                .agg(lambda x: x.mode().iloc[0] if len(x) else '—')
                .reset_index().rename(columns={'equipe': 'Team'}).sort_values('year')
            )
            teams_by_year.columns = ['Year', 'Team']
            st.dataframe(teams_by_year.set_index('Year').T, use_container_width=True)
            st.divider()

        # ── Global metrics ────────────────────────────────────────────────
        total_races = len(df_sel)
        wins = int((df_sel['rang'] == 1).sum()) if 'rang' in df_sel.columns else None
        top5 = int((df_sel['rang'] <= 5).sum()) if 'rang' in df_sel.columns else None
        total_pts_rider = df_sel['pts_uci'].sum() if 'pts_uci' in df_sel.columns else None
        total_pts_team_sel = (
            df_sel['pts_uci_equipe_stage'].sum()
            if 'pts_uci_equipe_stage' in df_sel.columns else None
        )
        best_year = (
            df_sel.groupby('year')['pts_uci'].sum().idxmax()
            if 'pts_uci' in df_sel.columns and len(df_sel) > 0 else None
        )
        pct_mean = df_sel['pts_uci_pct_max'].mean() if 'pts_uci_pct_max' in df_sel.columns else None

        cols = st.columns(6)
        cols[0].metric("Races (selected)", total_races)
        cols[1].metric("Wins", wins if wins is not None else "—")
        cols[2].metric("Top 5", top5 if top5 is not None else "—")
        if total_pts_rider is not None:
            cols[3].metric("Rider UCI pts", f"{total_pts_rider:.0f}")
        if total_pts_team_sel is not None:
            cols[4].metric("Team UCI pts (when selected)", f"{total_pts_team_sel:.0f}",
                           help="Total team UCI points in races where this rider was selected")
        if pct_mean is not None:
            cols[5].metric("Avg % of winner's pts", f"{pct_mean:.1f}%",
                           help="Average of team UCI pts / winner's individual pts for each race")

        # ── Granularity selector (shared by breakdown table + charts) ───────────
        st.subheader("Year-by-year evolution")
        gran_rider = st.radio(
            "Granularity", ["Year", "Year-Month"],
            horizontal=True, key="gran_rider",
        )

        def _group_key_rider(df):
            if gran_rider == "Year-Month" and 'date' in df.columns:
                df = df.copy()
                df['_gk'] = pd.to_datetime(df['date'], errors='coerce').dt.to_period('M').astype(str)
            else:
                df = df.copy()
                df['_gk'] = df['year'].astype(int).astype(str)
            return df

        df_sel_g = _group_key_rider(df_sel)
        df_raw_g = _group_key_rider(df_raw)
        x_label = "Month" if gran_rider == "Year-Month" else "Year"

        # ── UCI pts breakdown by category, per year ───────────────────────────
        # Stage pts from rider_data (df_sel); GC/KOM/Sprint from riders_gc
        if 'pts_uci' in df_sel.columns and len(df_sel) > 0:
            _yr = df_sel['year'].astype(int)
            bd_rows = {
                'Stage (rider)': df_sel.groupby(_yr)['pts_uci'].sum().fillna(0),
            }
            if 'pts_uci_equipe_stage' in df_sel.columns:
                bd_rows['Stage (team)'] = df_sel.groupby(_yr)['pts_uci_equipe_stage'].sum().fillna(0)

            _gc_df = cached_gc_pts(
                rider1,
                tuple(sorted(teams1)) if teams1 else (),
                tuple(int(y) for y in years) if years else (),
            )
            if _gc_df is not None and len(_gc_df) > 0:
                _gc_yr = _gc_df['year'].astype(int)
                for lbl, col in [
                    ('GC (team)',     'pts_uci_equipe_gc'),
                    ('KOM (team)',    'pts_uci_equipe_kom'),
                    ('Sprint (team)', 'pts_uci_equipe_points'),
                ]:
                    if col in _gc_df.columns:
                        bd_rows[lbl] = _gc_df.groupby(_gc_yr)[col].sum().fillna(0)

            bd = pd.DataFrame(bd_rows).T.sort_index(axis=1).astype(float)
            bd_fmt = bd.rename_axis('Year', axis=1).map(lambda v: f'{v:.0f}' if v != 0 else '—')
            st.markdown("**UCI points breakdown (when selected)**")
            st.dataframe(bd_fmt, use_container_width=True)

        yr_agg = {}
        if 'pts_uci' in df_sel_g.columns:
            yr_agg['pts_uci_coureur'] = df_sel_g.groupby('_gk')['pts_uci'].sum()
        if outcome in df_sel_g.columns:
            yr_agg['pts_uci_equipe'] = df_sel_g.groupby('_gk')[outcome].sum()
        if 'pts_uci_pct_max' in df_sel_g.columns:
            yr_agg['pct_max_moyen'] = df_sel_g.groupby('_gk')['pts_uci_pct_max'].mean()
        yr_agg['n_courses'] = df_sel_g.groupby('_gk').size()
        if wins:
            yr_agg['victoires'] = df_sel_g[df_sel_g['rang'] == 1].groupby('_gk').size()

        df_yr = pd.DataFrame(yr_agg).fillna(0).reset_index().rename(columns={'_gk': x_label})
        df_yr = df_yr.sort_values(x_label)

        if 'pts_uci_coureur' in df_yr.columns:
            fig_yr = go.Figure()
            fig_yr.add_trace(go.Bar(
                x=df_yr[x_label], y=df_yr['pts_uci_coureur'],
                name='Rider UCI pts', marker_color='#2271B3',
            ))
            if 'pts_uci_equipe' in df_yr.columns:
                fig_yr.add_trace(go.Bar(
                    x=df_yr[x_label], y=df_yr['pts_uci_equipe'],
                    name=f'Team UCI pts ({outcome})', marker_color='#E8824B', opacity=0.7,
                ))
            fig_yr.update_layout(
                barmode='group', template='plotly_white',
                yaxis_title='UCI pts', xaxis_title=x_label, height=330,
            )
            st.plotly_chart(fig_yr, use_container_width=True)

        # ── % max pts ───────────────────────────────────────────────
        if 'pct_max_moyen' in df_yr.columns:
            st.subheader(f"Avg % max pts by {x_label.lower()} (selected races)")
            fig_pct = go.Figure()
            fig_pct.add_trace(go.Bar(
                x=df_yr[x_label], y=df_yr['pct_max_moyen'].round(1),
                text=df_yr['pct_max_moyen'].round(1).astype(str) + '%',
                textposition='outside',
                marker_color='#43A5A0',
                name='% max pts',
            ))
            fig_pct.update_layout(
                yaxis_title="% of winner's pts per race",
                xaxis_title=x_label, template='plotly_white', height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_pct, use_container_width=True)
            st.caption(
                "For each selected race: team UCI pts ÷ winner's individual UCI pts × 100. "
                "A value of 100% means the team collected as many points as the stage/race winner individually."
            )

        # ── Selection rate ───────────────────────────────────────
        st.subheader(f"Selection rate by {x_label.lower()}")
        sel_yr = df_raw_g.groupby('_gk').agg(
            selected_n=('selected', 'sum'),
            total=('selected', 'count'),
        ).assign(taux=lambda d: d['selected_n'] / d['total']).reset_index()
        sel_yr = sel_yr.sort_values('_gk')
        sel_yr['label'] = (
            sel_yr['selected_n'].astype(int).astype(str) + '/' + sel_yr['total'].astype(str)
        )
        fig_sel = px.bar(
            sel_yr, x='_gk', y='taux', text='label',
            labels={'taux': 'Selection rate', '_gk': x_label},
            template='plotly_white', height=300,
            color='taux', color_continuous_scale=['#BDBDBD', '#2271B3'],
        )
        fig_sel.update_layout(coloraxis_showscale=False)
        fig_sel.update_traces(textposition='outside')
        st.plotly_chart(fig_sel, use_container_width=True)

        # ── Top 15 races ────────────────────────────────────────────────────
        if 'pts_uci' in df_sel.columns:
            st.subheader("Top 15 best races (rider UCI pts)")
            top_cols = [c for c in
                        ['course', 'year', 'stage_num', 'rang', 'pts_uci', outcome]
                        if c in df_sel.columns]
            st.dataframe(
                df_sel.nlargest(15, 'pts_uci')[top_cols].reset_index(drop=True),
                use_container_width=True,
            )

        # ── pts_uci_pct_max distribution by race type ──────────────────
        if 'pts_uci_pct_max' in df_sel.columns and 'stage_cluster_label' in df_sel.columns:
            st.subheader("% of winner's pts — distribution by race type")
            pct_by_cluster = df_sel.groupby('stage_cluster_label').agg(
                mean=('pts_uci_pct_max', 'mean'),
                median=('pts_uci_pct_max', 'median'),
                max=('pts_uci_pct_max', 'max'),
                n=('pts_uci_pct_max', 'count'),
            ).round(1).reset_index()
            st.dataframe(pct_by_cluster, use_container_width=True)

with tab_stats:
    _render_stats()
