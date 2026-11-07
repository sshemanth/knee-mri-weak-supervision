"""
regex_labeler.py  (labeller 1)
Rule-based report labeller. For each finding, a report sentence counts as
positive when it mentions the structure (anatomy terms) together with an
abnormality term (pathology terms) that is not negated just before it.
Terms cover Spanish, English, German, Dutch, French and Italian.
Output per finding: 1 (present), 0 (stated absent / structure normal) or
None (not mentioned). Writes outputs/weak_labels_v2.csv.
"""
import os, pandas as pd, re, unicodedata, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("KNEE_ROOT", "."))  # folder with data/ and outputs/
DATA = ROOT / "data"
train = pd.read_csv(DATA / "train.csv")

FINDINGS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
            'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

# ---- text normalisation: remove accents and lower-case ----
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def norm(text):
    return strip_accents(str(text)).lower()

# ---- abnormality terms shared by groups of findings ----
LIG_PATHOLOGY = [r'rotura', r'rutura', r'ruptura', r'desgarro', r'lesion', r'esguince',
                  r'\btear', r'ruptur', r'sprain', r'\briss\b', r'scheur', r'dechirure',
                  r'rottura', r'lesione', r'discontinuidad', r'avulsion']

MEN_PATHOLOGY = [r'rotura', r'rutura', r'ruptura', r'desgarro', r'\btear', r'fisura',
                  r'lesion', r'ruptur', r'scheur', r'dechirure', r'rottura',
                  r'meniscopatia', r'degenerat']

OA_PATHOLOGY = [r'artrosis', r'osteoartrosis', r'osteoartritis', r'gonartrosis',
                 r'cambios degenerativos', r'pinzamiento', r'osteofito',
                 r'esclerosis subcondral', r'condropatia', r'condromalacia',
                 r'perdida.{0,15}cartilago', r'adelgazamiento.{0,15}cartilago',
                 r'ulcera.{0,10}condral', r'osteoarthritis', r'degenerative',
                 r'joint space narrowing', r'osteophyte', r'subchondral sclerosis',
                 r'chondropathy', r'chondromalacia', r'cartilage loss',
                 r'cartilage thinning', r'\barthrose\b', r'degenerative veranderungen']

# ---- per finding: anatomy terms (None = finding named directly) and abnormality terms ----
SPEC = {
    'ACL': dict(anatomy=[r'\bacl\b', r'ligamento cruzado anterior', r'\blca\b',
                          r'cruzado anterior', r'anterior cruciate', r'vorderes kreuzband',
                          r'voorste kruisband', r'ligament croise anterieur',
                          r'legamento crociato anteriore'],
                pathology=LIG_PATHOLOGY),
    'MCL': dict(anatomy=[r'\bmcl\b', r'ligamento colateral medial', r'\blcm\b',
                          r'medial collateral', r'mediales seitenband',
                          r'mediale collaterale band', r'ligament collateral medial',
                          r'legamento collaterale mediale'],
                pathology=LIG_PATHOLOGY),
    'Medial Meniscus': dict(anatomy=[r'menisco medial', r'menisco interno',
                                      r'medial meniscus', r'innenmeniskus',
                                      r'mediale meniscus', r'menisque medial',
                                      r'menisque interne', r'menisco mediale'],
                             pathology=MEN_PATHOLOGY),
    'Lateral Meniscus': dict(anatomy=[r'menisco lateral', r'menisco externo',
                                       r'lateral meniscus', r'aussenmeniskus',
                                       r'laterale meniscus', r'menisque lateral',
                                       r'menisque externe', r'menisco laterale'],
                              pathology=MEN_PATHOLOGY),
    'Medial OA': dict(anatomy=[r'\bmedial\b', r'\binterno\b', r'\binterna\b',
                                r'condilo femoral medial', r'platillo tibial medial',
                                r'meseta tibial medial', r'compartimento medial',
                                r'femorotibial medial'],
                       pathology=OA_PATHOLOGY),
    'Lateral OA': dict(anatomy=[r'\blateral\b', r'\bexterno\b', r'\bexterna\b',
                                 r'condilo femoral lateral', r'platillo tibial lateral',
                                 r'meseta tibial lateral', r'compartimento lateral',
                                 r'femorotibial lateral'],
                        pathology=OA_PATHOLOGY),
    'PF OA': dict(anatomy=[r'patelofemoral', r'femoropatelar', r'\brotula\b',
                            r'\bpatela\b', r'patella', r'\btroclea\b', r'trochlea',
                            r'patellofemoral'],
                   pathology=OA_PATHOLOGY),
    'Effusion': dict(anatomy=None,
                      pathology=[r'derrame articular', r'\bderrame\b', r'\beffusion\b',
                                 r'gelenkerguss', r'gewrichtseffusie',
                                 r'epanchement articulaire', r'versamento articolare',
                                 r'derrame intraarticular']),
    'Synovitis': dict(anatomy=None, pathology=[r'sinovitis', r'synovitis', r'synovite']),
    "Baker's": dict(anatomy=None,
                     pathology=[r'quiste de baker', r"baker'?s? cyst", r'quiste popliteo',
                                r'bakerzyste', r'bakercyste', r'kyste poplite',
                                r'cisti di baker', r'cisto poplitea']),
    'Contusion': dict(anatomy=None,
                       pathology=[r'contusion osea', r'edema oseo.{0,20}contusion',
                                  r'bone contusion', r'bone bruise', r'knochenkontusion',
                                  r'botcontusie', r'contusion osseuse']),
    'Fracture': dict(anatomy=None,
                      pathology=[r'\bfractura\b', r'\bfracture\b', r'\bfraktur\b',
                                 r'\bfractuur\b', r'\bfrattura\b']),
}

# ---- negation cues; a match is ignored if one appears up to 45 characters before it ----
NEGATION = [r'\bno\b', r'\bsin\b', r'ausencia de', r'sin evidencia de', r'descarta',
            r'not\b', r'no evidence of', r'without', r'absence of', r'\bkeine?\b',
            r'\bohne\b', r'\bgeen\b', r'\bzonder\b', r'pas de', r'\bsans\b', r'\baucun',
            r'\bsenza\b', r'assenza di', r'\bausente', r'negativo para']
NEG_RE = re.compile('|'.join(NEGATION))

def sentences(text):
    """Split a report into sentences on full stops, semicolons and new lines."""
    return [s for s in re.split(r'[.;\n]+', text) if s.strip()]

def has_any(patterns, s):
    return any(re.search(p, s) for p in patterns)

def negated_before(s, pos, window=45):
    """True if a negation cue occurs in the `window` characters before position `pos`."""
    return bool(NEG_RE.search(s[max(0, pos - window):pos]))

def label_finding(text_norm, finding):
    """Return 1, 0 or None for one finding in one normalised report."""
    spec = SPEC[finding]
    anatomy_pats, path_pats = spec['anatomy'], spec['pathology']
    anatomy_mentioned = False
    for sent in sentences(text_norm):
        relevant = anatomy_pats is None or has_any(anatomy_pats, sent)
        if anatomy_pats is not None and has_any(anatomy_pats, sent):
            anatomy_mentioned = True
        if not relevant:
            continue
        for pat in path_pats:
            m = re.search(pat, sent)
            if m and not negated_before(sent, m.start()):
                return 1          # structure + non-negated abnormality -> present
    # findings without anatomy terms: an explicitly negated mention means absent
    if anatomy_pats is None:
        for sent in sentences(text_norm):
            for pat in path_pats:
                m = re.search(pat, sent)
                if m and negated_before(sent, m.start()):
                    return 0
        return None
    # structure mentioned without abnormality -> absent; never mentioned -> unknown
    return 0 if anatomy_mentioned else None

# ---- label every report ----
train['report_norm'] = train['Report'].fillna('').map(norm)
for f in FINDINGS:
    train[f'pred_{f}'] = train['report_norm'].map(lambda t, f=f: label_finding(t, f))

# ---- score against the gold studies (missing values filled with 0.5 here) ----
gold = train.dropna(subset=FINDINGS, how='all').copy()
print(f"gold studies: {len(gold)}")
print(f"\n{'finding':20s} {'coverage':>9s} {'auc':>6s} {'acc':>6s}")
aucs = []
for f in FINDINGS:
    sub = gold.dropna(subset=[f])
    y_true = sub[f].astype(int)
    pred = sub[f'pred_{f}']
    coverage = pred.notna().mean()
    pred_filled = pred.fillna(0.5)
    try:
        auc = roc_auc_score(y_true, pred_filled)
    except ValueError:
        auc = float('nan')
    acc = ((pred_filled >= 0.5).astype(int) == y_true).mean()
    aucs.append(auc)
    print(f"{f:20s} {coverage:9.2f} {auc:6.3f} {acc:6.3f}")

print(f"\nmacro AUC across findings: {np.nanmean(aucs):.4f}")
out_csv = ROOT / "outputs" / "weak_labels_v2.csv"
train.drop(columns=['report_norm']).to_csv(out_csv, index=False)
print("saved -> outputs/weak_labels_v2.csv")
