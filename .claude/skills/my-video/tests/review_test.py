# What Claude does at the skill's "review plan.json" step, for this Portuguese test clip.
import json, sys
P = sys.argv[1]; p = json.load(open(P)); W = p['words']
def find(word, after=0):
    for w in W:
        if w['t'] >= after and w['w'].lower().strip('.,?').startswith(word): return w['t']
for b, q in zip(p['broll'], ['office meeting clients', 'elderly caregiver assistance']): b['query'] = q; b['path'] = None
fix = {'NSS': 'INSS', 'descolham.': 'desconhecem.', 'escutório': 'escritório', 'gritando.': 'gritante.', 'reducido': 'reduzido',
       'depender.': 'dependentes.', 'IPPC': 'BPC', 'assustencial.': 'assistencial.', 'direita': 'direito', 'décima': 'décimo',
       'terceira': 'terceiro', 'aumento': 'momento', 'atrapilhado': 'repassado', 'vejar': 'planejar', 'tudo': 'estudo', 'testar': 'trata-se'}
for c in p['captions']:
    for w in c['words']:
        w['w'] = fix.get(w['w'], fix.get(w['w'].upper(), w['w'])) if w['w'] not in ('a',) else w['w']
ac = next((c for c in p['cards'] if c['type'] == 'acronym'), None)
t_comp = find('diferença', 17); t_tl = find('valor', 22); t_ck = find('nessa', 44)
cards = [
 {"type": "compare", "t0": round(max(t_comp, ac['t1'] + 0.05 if ac else 0), 2), "t1": round(find('aposentadoria', t_comp) + 1.0, 2),
  "panels": ["O BPC é um benefício assistencial: um salário mínimo para idosos com 65+ ou pessoas com deficiência em vulnerabilidade.",
             "Aposentadoria é uma prestação previdenciária, paga a quem contribuiu para o INSS."], "symbol": "≠"},
 {"type": "timeline", "t0": round(t_tl, 2), "t1": round((find('depend', t_tl) or t_tl + 6.5) + 0.6, 2),
  "points": [{"label": "Salário mínimo", "value": "R$1.621", "at_word": "mínimo"}, {"label": "Teto do INSS", "value": "R$8.475,55", "at_word": "teto"},
             {"label": "13º Salário", "value": "8,33%", "at_word": "décim"}, {"label": "Pensão por morte", "value": "50%", "at_word": "pensão"}]},
 {"type": "checklist", "t0": round(t_ck + 0.8, 2), "t1": round((find('depend', t_ck) or t_ck + 7) + 0.8, 2),
  "items": [{"text": "Salário mínimo", "ok": True, "at_word": "salário"}, {"text": "13º Salário", "ok": False, "at_word": "décim"},
            {"text": "Pensão por morte", "ok": False, "at_word": "falecimento"}]}]
p['cards'] = [c for c in p['cards'] if c['type'] == 'acronym'] + cards
json.dump(p, open(P, 'w'), ensure_ascii=False, indent=1)
print('review applied:', [(c['type'], c['t0'], c['t1']) for c in p['cards']])
