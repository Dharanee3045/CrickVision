#!/usr/bin/env python3
"""Resolve IPL player-name abbreviations using the official Cricsheet register.
Run from the CrickVision project folder with internet access: py complete_player_names.py
Creates a reviewable mapping, updates delivery CSVs and regenerates player_scores.csv.
"""
import csv, io, os, re, sys, urllib.request, zipfile
from collections import defaultdict

BASE='https://cricsheet.org/register/'
FILES={'people.csv':BASE+'people.csv','names.csv':BASE+'names.csv'}

def read_url(url):
    req=urllib.request.Request(url,headers={'User-Agent':'CrickVision player-name normalizer/1.0'})
    with urllib.request.urlopen(req,timeout=45) as r: return r.read().decode('utf-8-sig')

def rows(text): return list(csv.DictReader(io.StringIO(text)))
def norm(s): return re.sub(r'[^a-z0-9]+',' ',str(s).lower()).strip()
def looks_full(s):
    s=str(s).strip()
    # At least two complete words and not a sequence of initials.
    words=s.split()
    return len(words)>=2 and not all(len(w.strip('.'))<=2 for w in words)
def score(s):
    words=s.split()
    return (sum(len(w.strip('.'))>2 for w in words), len(s), -sum(1 for w in words if len(w.strip('.'))==1))

def main():
    root=os.path.dirname(os.path.abspath(__file__))
    try:
        people=rows(read_url(FILES['people.csv']))
        variants=rows(read_url(FILES['names.csv']))
    except Exception as e:
        print('Could not download the official Cricsheet register:',e)
        print('Check your internet connection and try again.')
        sys.exit(2)
    person_by_id={r.get('identifier',''):r for r in people}
    aliases=defaultdict(set)
    for p in people:
        pid=p.get('identifier','');
        for k in ('name','unique_name'):
            if p.get(k): aliases[norm(p[k])].add(pid)
    variants_by_id=defaultdict(set)
    for v in variants:
        pid=v.get('identifier',''); name=(v.get('name') or '').strip()
        if pid and name:
            variants_by_id[pid].add(name); aliases[norm(name)].add(pid)
    # Read current mapping list, built in the previous project package.
    map_path=os.path.join(root,'player_name_map.csv')
    if not os.path.exists(map_path):
        print('player_name_map.csv not found. Extract the project zip and run this script inside its CrickVision folder.')
        sys.exit(2)
    with open(map_path,encoding='utf-8-sig',newline='') as f: old=list(csv.DictReader(f))
    resolved={}; review=[]
    for r in old:
        src=(r.get('source_name') or '').strip()
        if not src: continue
        ids=aliases.get(norm(src),set())
        # Only accept exact alias hits that identify one unique person.
        if len(ids)==1:
            pid=next(iter(ids)); p=person_by_id.get(pid,{})
            choices=set(variants_by_id.get(pid,set()))
            choices.update(x for x in (p.get('name',''),p.get('unique_name','')) if x)
            # Prefer a genuine expanded name rather than initials, but do not invent missing names.
            expanded=[x for x in choices if looks_full(x)]
            if expanded:
                best=max(expanded,key=score)
                resolved[src]=best
                continue
        # preserve previous verified curated mapping where it is already more descriptive
        previous=(r.get('full_name') or src).strip()
        if r.get('status')=='Mapped' and previous!=src:
            resolved[src]=previous
        else:
            review.append((src, sorted(ids)))
    # Map all observed player columns in both delivery files using exact normalized keys.
    def mapped(s): return resolved.get(str(s).strip(), str(s).strip())
    for fn in ('deliveries.csv','deliveries_clean.csv'):
        path=os.path.join(root,fn)
        if not os.path.exists(path): continue
        with open(path,encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); fields=reader.fieldnames; data=list(reader)
        for row in data:
            for col in ('batter','batsman','bowler','non_striker','player_dismissed','fielder'):
                if col in row and row[col]: row[col]=mapped(row[col])
        with open(path,'w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
    # Rewrite audit with official-source results; unresolved entries are explicit, not guessed.
    with open(map_path,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['source_name','full_name','status','source'])
        for r in old:
            src=(r.get('source_name') or '').strip()
            if src in resolved:
                w.writerow([src,resolved[src],'Mapped','Cricsheet register or curated verified mapping'])
            else:
                ids=aliases.get(norm(src),set())
                status='Ambiguous' if len(ids)>1 else 'Needs manual verification'
                w.writerow([src,src,status,'Cricsheet register did not uniquely confirm a full name'])
    # Update the bio table as well so profile labels stay consistent.
    bio_path=os.path.join(root,'player_bio.csv')
    if os.path.exists(bio_path):
        with open(bio_path,encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); fields=reader.fieldnames; data=list(reader)
        for row in data:
            if 'player' in row and row['player']: row['player']=mapped(row['player'])
        with open(bio_path,'w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
    # Rebuild scores with the project's own script, if dependencies are installed.
    analysis=os.path.join(root,'03_player_analysis.py')
    if os.path.exists(analysis):
        import subprocess
        result=subprocess.run([sys.executable,analysis],cwd=root)
        if result.returncode:
            print('Names were updated, but player score regeneration failed. Run 03_player_analysis.py manually.')
    print('Name normalization finished.')
    print('Mapped names:',len(resolved))
    print('Needs review:',len(review))
    print('Review player_name_map.csv for ambiguous/unconfirmed names; no identity was guessed.')

if __name__=='__main__': main()
