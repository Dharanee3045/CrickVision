"""CrickVision Phase 1: player scores from the actual deliveries_clean.csv schema."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / 'deliveries_clean.csv')
required = {'match_id','over','batsman','bowler','batsman_runs','total_runs','extras_type','player_dismissed','dismissal_kind'}
missing = required - set(d.columns)
if missing:
    raise ValueError(f'Missing columns: {sorted(missing)}')
for c in ('batsman','bowler','player_dismissed','dismissal_kind','extras_type'):
    d[c] = d[c].fillna('').astype(str).str.strip()
for c in ('batsman_runs','total_runs','over'):
    d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0)
d['extras_type'] = d['extras_type'].str.lower()
d['dismissal_kind'] = d['dismissal_kind'].str.lower()
# Dataset uses 0..19 overs. For datasets numbered 1..20, death overs are 16..20.
first_over = d.groupby('match_id')['over'].transform('min')
d['_death'] = d['over'] >= np.where(first_over.eq(0),15,16)
d['_legal_ball'] = (~d['extras_type'].isin(['wides','noballs'])).astype(int)
d['_batter_ball'] = (d['extras_type'] != 'wides').astype(int)
# Bowler runs exclude byes and leg-byes; a penalty is also not attributed to the bowler.
d['_bowler_runs'] = np.where(d['extras_type'].isin(['byes','legbyes','penalty']),0,d['total_runs'])
# Only wickets credited to the bowler.
excluded = {'run out','retired hurt','retired out','obstructing the field'}
d['_bowler_wicket'] = ((d['player_dismissed'] != '') & ~d['dismissal_kind'].isin(excluded)).astype(int)
# Dismissals charged to a batter exclude run-outs, retired hurt, and obstructing the field.
non_batter_dismissals = {'run out','retired hurt','obstructing the field'}
d['_batter_dismissal'] = ((d['player_dismissed'] != '') & ~d['dismissal_kind'].isin(non_batter_dismissals)).astype(int)

bat = d.groupby('batsman').agg(runs=('batsman_runs','sum'),balls_faced=('_batter_ball','sum'),fours=('batsman_runs',lambda s:int((s==4).sum())),sixes=('batsman_runs',lambda s:int((s==6).sum())),matches_played=('match_id','nunique')).reset_index().rename(columns={'batsman':'player'})
outs = d[d['_batter_dismissal'].eq(1)].groupby('player_dismissed').size().rename('dismissals')
bat = bat.join(outs,on='player')
bat['dismissals'] = bat['dismissals'].fillna(0).astype(int)
bat['batting_average'] = (bat['runs'] / bat['dismissals'].replace(0,np.nan)).fillna(bat['runs']).round(2)
bat['strike_rate'] = (100*bat['runs']/bat['balls_faced'].replace(0,np.nan)).fillna(0).round(2)
death_bat = d[d['_death']].groupby('batsman').agg(death_over_runs=('batsman_runs','sum'),death_over_balls=('_batter_ball','sum')).reset_index().rename(columns={'batsman':'player'})
bat = bat.merge(death_bat,on='player',how='left')
for c in ('death_over_runs','death_over_balls'):bat[c]=bat[c].fillna(0)
bat['death_over_strike_rate']=(100*bat['death_over_runs']/bat['death_over_balls'].replace(0,np.nan)).fillna(0).round(2)
# Batter-specific death-over scoring rate: runs per six balls faced.
bat['death_over_run_rate']=(6*bat['death_over_runs']/bat['death_over_balls'].replace(0,np.nan)).fillna(0).round(2)
bat['batting_sample_reliable']=bat['balls_faced']>=60

bowl = d.groupby('bowler').agg(balls_bowled=('_legal_ball','sum'),runs_conceded=('_bowler_runs','sum'),wickets=('_bowler_wicket','sum')).reset_index().rename(columns={'bowler':'player'})
bowl['overs_bowled']=bowl['balls_bowled']/6
bowl['economy']=(6*bowl['runs_conceded']/bowl['balls_bowled'].replace(0,np.nan)).fillna(0).round(2)
bowl['bowling_average']=(bowl['runs_conceded']/bowl['wickets'].replace(0,np.nan)).fillna(bowl['runs_conceded']).round(2)
bowl['bowling_sample_reliable']=bowl['balls_bowled']>=60
death_bowl = d[d['_death']].groupby('bowler').agg(death_over_wickets=('_bowler_wicket','sum'),death_over_runs_conceded=('_bowler_runs','sum'),death_over_legal_balls=('_legal_ball','sum')).reset_index().rename(columns={'bowler':'player'})
bowl=bowl.merge(death_bowl,on='player',how='left')
for c in ('death_over_wickets','death_over_runs_conceded','death_over_legal_balls'):bowl[c]=bowl[c].fillna(0)
bowl['death_over_economy']=(6*bowl['death_over_runs_conceded']/bowl['death_over_legal_balls'].replace(0,np.nan)).fillna(0).round(2)

def normalize(s,invert=False):
    s=pd.to_numeric(s,errors='coerce').fillna(0)
    if invert:s=-s
    lo,hi=s.min(),s.max()
    return pd.Series(5.,index=s.index) if hi==lo else 10*(s-lo)/(hi-lo)

bat['batting_impact_score']=(.40*normalize(bat['batting_average'])+.35*normalize(bat['strike_rate'])+.25*normalize(bat['death_over_runs'])).round(2)
bowl['bowling_impact_score']=(.45*normalize(bowl['wickets'])+.30*normalize(bowl['economy'],True)+.25*normalize(bowl['bowling_average'],True)).round(2)
players=bat.merge(bowl,on='player',how='outer')
for c in players.select_dtypes(include='number').columns:players[c]=players[c].fillna(0)
for c in ('batting_sample_reliable','bowling_sample_reliable'):players[c]=players[c].fillna(False).astype(bool)
players['data_confidence']=np.where(players['batting_sample_reliable']|players['bowling_sample_reliable'],'Reliable','Limited Data')
# Role is a heuristic derived from meaningful participation, not just any nonzero score.
# For accurate official roles, merge a curated role dataset separately.
def classify(r):
    regular_bat = r['balls_faced']>=100 and r['runs']>=100
    regular_bowl = (r['balls_bowled']>=120 and (r['balls_bowled'] >= .25*r['balls_faced'] or r['wickets']>=30))
    if regular_bat and regular_bowl:return 'All-Rounder'
    if regular_bowl:return 'Bowler'
    if regular_bat:return 'Batsman'
    if r['balls_bowled']>=24 and r['balls_bowled']>r['balls_faced']:return 'Bowler'
    if r['balls_faced']>0:return 'Batsman'
    if r['balls_bowled']>0:return 'Bowler'
    return 'Unknown'
players['role']=players.apply(classify,axis=1)
# Optional verified roles for ALL players: if player_roles.csv exists,
# its columns should be player,role. It overrides only matching players.
role_file = ROOT / 'player_roles.csv'
if role_file.exists():
    official = pd.read_csv(role_file)
    if {'player', 'role'}.issubset(official.columns):
        role_map = {'batsman':'Batsman','batter':'Batsman','batswoman':'Batsman',
                    'bowler':'Bowler','all-rounder':'All-Rounder',
                    'all rounder':'All-Rounder','allrounder':'All-Rounder'}
        official['role'] = official['role'].astype(str).str.strip().str.lower().map(role_map)
        official = official.dropna(subset=['player','role']).drop_duplicates('player')
        lookup = official.set_index('player')['role']
        players['role'] = players['player'].map(lookup).fillna(players['role'])
        print(f'Applied verified role overrides for {players["player"].isin(lookup.index).sum()} players')

players['overall_impact_score']=np.select([players['role'].eq('All-Rounder'),players['role'].eq('Bowler')],[.55*players['batting_impact_score']+.45*players['bowling_impact_score'],players['bowling_impact_score']],default=players['batting_impact_score']).round(2)
players=players.sort_values('overall_impact_score',ascending=False).reset_index(drop=True)
players.to_csv(ROOT/'player_scores.csv',index=False)
print('Saved player_scores.csv:',len(players),'players')
for name in ('JJ Bumrah','RG Sharma','V Kohli','RA Jadeja','HH Pandya'):
    result=players.loc[players.player.eq(name),['player','role','wickets','death_over_wickets','death_over_runs_conceded','death_over_economy','overall_impact_score']]
    if not result.empty:print(result.to_string(index=False))
