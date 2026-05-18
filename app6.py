import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import streamlit as st
import torch
import torch.nn as nn
import pickle
import json
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from transformers import RobertaTokenizer, RobertaModel
from textblob import TextBlob
from better_profanity import profanity
from normalizer import normalize_text
from PIL import Image

# ──────────────────────────────────────────────────────────
# INIT
# ──────────────────────────────────────────────────────────
profanity.load_censor_words()

with open('config.json') as f:
    config = json.load(f)

THRESHOLD = config['best_threshold']

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# ──────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────
class FastClassifier(nn.Module):

    def __init__(self, num_features=6):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(768 + num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding, features):

        x = torch.cat([embedding, features], dim=1)

        return self.classifier(x).squeeze(1)

# ──────────────────────────────────────────────────────────
# LOAD MODELS
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():

    tokenizer = RobertaTokenizer.from_pretrained(
        'roberta-base'
    )

    roberta = RobertaModel.from_pretrained(
        'roberta-base'
    ).to(device)

    roberta.eval()

    model = FastClassifier().to(device)

    model.load_state_dict(
        torch.load(
            'best_model.pt',
            map_location=device
        )
    )

    model.eval()

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return tokenizer, roberta, model, scaler

# ──────────────────────────────────────────────────────────
# TONE DETECTION
# ──────────────────────────────────────────────────────────
MOCK_EMOJIS = {
    '💀', '😬', '💅', '🙄',
    '😒', '😏', '😈', '👺',
    '🤡', '😤'
}

BANTER_EMOJIS = {
    '😂', '🤣', '😭', '😅',
    '🥲', '💯', '🔥', '👀'
}

FRIENDLY_WORDS = {
    'lol', 'lmao', 'haha',
    'hehe', 'jk', 'kidding',
    'joking', 'rofl',
    'bestie', 'bff',
    'ily', 'ilysm'
}

SARCASM_PATTERNS = [

    r'yeah right',
    r'as if',
    r'good for you.*(🙄|😒|💀)',
    r'congratulations.*(🙄|😒)',
    r'oh wow.*(smart|great|amazing)',
]

def detect_tone(text):

    text_lower = text.lower()

    words = set(text_lower.split())

    has_mock_emoji = any(
        e in text for e in MOCK_EMOJIS
    )

    has_banter_emoji = any(
        e in text for e in BANTER_EMOJIS
    )

    has_friendly_word = bool(
        words & FRIENDLY_WORDS
    )

    # sarcasm
    for pattern in SARCASM_PATTERNS:

        if re.search(pattern, text_lower):

            return (
                'sarcastic',
                'sarcasm pattern detected'
            )

    # mocking
    if has_mock_emoji:

        return (
            'mocking',
            'mocking emoji detected'
        )

    # friendly
    if (
        (has_banter_emoji or has_friendly_word)
        and
        not has_mock_emoji
    ):

        return (
            'friendly',
            'friendly/banter indicators detected'
        )

    positive_patterns = [

        'without you',
        'love you',
        'miss you',
        'need you',
        'we love you',
        'you matter',
        'group would be boring without you'
    ]

    for p in positive_patterns:

        if p in text_lower:

            return (
                'friendly',
                'positive relationship phrase detected'
            )

    return 'neutral', ''

# ──────────────────────────────────────────────────────────
# INDIRECT INSULTS
# ──────────────────────────────────────────────────────────
INDIRECT_PATTERNS = [

    r'nobody likes you',
    r'no one asked',
    r'stay mad',
    r'cope harder',
    r'imagine being',
    r'how does it feel to be',
    r'ratio',
    r'took an? [lL]',
]

def has_indirect_insult(text):

    text_lower = text.lower()

    for pattern in INDIRECT_PATTERNS:

        if re.search(pattern, text_lower):

            return True, pattern

    return False, None

# ──────────────────────────────────────────────────────────
# FRIENDLY MITIGATION
# ──────────────────────────────────────────────────────────
def friendly_mitigation(text, tone, context):

    text_lower = text.lower()

    positive_patterns = [

        r'without you',
        r'love you',
        r'ily',
        r'bestie',
        r'bff',
        r'jk',
        r'kidding',
        r'joking',
        r'miss you',
        r'we love you',
        r'you matter',
    ]

    mitigation = 0.0

    # friendly tone
    if tone == 'friendly':

        mitigation -= 0.15

    # positive phrases
    for pattern in positive_patterns:

        if re.search(pattern, text_lower):

            mitigation -= 0.08

    # friend context — only apply if tone is NOT neutral
    # this prevents mitigation from cancelling context boost
    # when there are no friendly signals in the text
    if (
        context == 'Between friends (may contain banter)'
        and tone != 'neutral'
    ):
        mitigation -= 0.10

    # limit mitigation
    mitigation = max(mitigation, -0.30)

    return mitigation

# ──────────────────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────────────────
def extract_text_from_image(image):

    if image.mode != 'RGB':
        image = image.convert('RGB')

    text = pytesseract.image_to_string(
        image,
        lang='eng'
    )

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
# ──────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────
def extract_features(text):

    blob = TextBlob(text)

    return [

        blob.sentiment.polarity,

        sum(
            1 for w in text.split()
            if profanity.contains_profanity(w)
        ),

        len(text.split()),

        sum(
            1 for c in text
            if c.isupper()
        ) / max(len(text), 1),

        text.count('!'),

        text.count('?')
    ]

# ──────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────
def predict(
    text,
    tokenizer,
    roberta,
    model,
    scaler,
    context='Unknown / Public comment'
):

    normalized = normalize_text(text)

    # tone
    tone, tone_reason = detect_tone(text)

    # indirect insults
    is_indirect, matched_pattern = has_indirect_insult(
        normalized
    )

    indirect_boost = 0.0

    if is_indirect:
        indirect_boost = 0.15

    # context boost
    context_boost = 0.0

    if context == 'Between strangers':
        context_boost = 0.10

    elif context == 'Between friends (may contain banter)':
        context_boost = -0.25

    elif context == 'Directed at me personally':
        context_boost = 0.20

    elif context == 'Repeated messages from same person':
        context_boost = 0.25

    # mitigation — only applies when there are actual
    # friendly signals, NOT based on context alone
    mitigation = friendly_mitigation(
        text,
        tone,
        context
    )

    # tokenize
    enc = tokenizer(
        normalized,
        max_length=96,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():

        emb = roberta(
            enc['input_ids'].to(device),
            enc['attention_mask'].to(device)
        ).last_hidden_state[:, 0, :]

        feat = torch.tensor(
            scaler.transform([
                extract_features(normalized)
            ]),
            dtype=torch.float32
        ).to(device)

        prob = model(emb, feat).item()

    # final score
    boosted_prob = (
        prob
        + indirect_boost
        + context_boost
        + mitigation
    )

    # clamp
    boosted_prob = max(
        0.0,
        min(1.0, boosted_prob)
    )

    # final decision
    is_cyber = boosted_prob >= THRESHOLD

    # debug
    print("===================================")
    print("RAW:", prob)
    print("INDIRECT:", indirect_boost)
    print("CONTEXT:", context_boost)
    print("MITIGATION:", mitigation)
    print("FINAL:", boosted_prob)
    print("TONE:", tone)
    print("CONTEXT SELECTED:", context)
    print("===================================")

    return (

        boosted_prob,
        is_cyber,
        tone,
        tone_reason,
        is_indirect,
        matched_pattern,
        prob,
        context_boost,
        mitigation
    )


# ──────────────────────────────────────────────────────────
# WORD-LEVEL DICTIONARIES
# ──────────────────────────────────────────────────────────
THREAT_WORDS = {
    'kill','die','death','dead','murder','shoot','stab',
    'hurt','harm','destroy','attack','beat','punch','cut',
    'bleed','suffer','threaten','exterminate','execute',
}

HATE_WORDS = {
    'ugly','stupid','idiot','dumb','moron','loser',
    'worthless','pathetic','disgusting','gross','freak',
    'weirdo','trash','garbage','waste','fat','pig',
    'rat','filth','dirty','hate','despise','detest',
}

EXCLUSION_WORDS = {
    'nobody','alone','friendless','unwanted','invisible',
    'ignored','outcast','rejected','unlovable','unloved',
    'disappear','nothing',
}

INTENSIFIER_WORDS = {
    'always','never','everyone','worst','literally',
    'absolutely','completely','totally','forever',
}

WORD_CATEGORIES = {
    'threat':      ('#ff4444', '#3a0a0a', '🔴 Threat/Violence'),
    'hate':        ('#ff8c00', '#2d1a00', '🟠 Hate/Dehumanising'),
    'exclusion':   ('#a855f7', '#1e0a2d', '🟣 Exclusion/Social Attack'),
    'profanity':   ('#e74c3c', '#2d0b0b', '🔴 Profanity'),
    'intensifier': ('#facc15', '#2a2000', '🟡 Intensifier'),
    'indirect':    ('#22d3ee', '#001e22', '🔵 Indirect Insult'),
    'caps':        ('#fb923c', '#251000', '🟠 Aggressive Caps'),
}


def analyse_words(text):
    """Return (html_highlighted, flagged_list, category_counts)."""
    import re as _re
    words       = text.split()
    text_lower  = text.lower()
    flagged     = []
    cat_counts  = {k: 0 for k in WORD_CATEGORIES}
    word_flags  = {}   # index -> (category, reason)

    for i, word in enumerate(words):
        clean = _re.sub(r'[^a-z]', '', word.lower())

        if profanity.contains_profanity(word):
            word_flags[i] = ('profanity', 'flagged by profanity filter')
            flagged.append({'word': word, 'category': 'profanity',
                            'reason': 'flagged by profanity filter'})
            cat_counts['profanity'] += 1
        elif clean in THREAT_WORDS:
            word_flags[i] = ('threat', 'threat / violence vocabulary')
            flagged.append({'word': word, 'category': 'threat',
                            'reason': 'threat / violence vocabulary'})
            cat_counts['threat'] += 1
        elif clean in HATE_WORDS:
            word_flags[i] = ('hate', 'hate / dehumanising vocabulary')
            flagged.append({'word': word, 'category': 'hate',
                            'reason': 'hate / dehumanising vocabulary'})
            cat_counts['hate'] += 1
        elif clean in EXCLUSION_WORDS:
            word_flags[i] = ('exclusion', 'exclusion / social attack')
            flagged.append({'word': word, 'category': 'exclusion',
                            'reason': 'exclusion / social attack'})
            cat_counts['exclusion'] += 1
        elif clean in INTENSIFIER_WORDS:
            word_flags[i] = ('intensifier', 'amplifies surrounding toxicity')
            flagged.append({'word': word, 'category': 'intensifier',
                            'reason': 'amplifies surrounding toxicity'})
            cat_counts['intensifier'] += 1
        elif word.isupper() and len(clean) >= 3:
            word_flags[i] = ('caps', 'all-caps aggressive tone')
            flagged.append({'word': word, 'category': 'caps',
                            'reason': 'all-caps aggressive tone marker'})
            cat_counts['caps'] += 1

    # phrase-level: indirect insult patterns
    char_cat = [''] * len(text)
    for pattern in INDIRECT_PATTERNS:
        for m in _re.finditer(pattern, text_lower):
            for idx in range(m.start(), m.end()):
                if idx < len(char_cat):
                    char_cat[idx] = 'indirect'
            flagged.append({'word': m.group(), 'category': 'indirect',
                            'reason': f'indirect insult pattern: `{pattern}`'})
            cat_counts['indirect'] += 1

    # Build highlighted HTML
    parts = []
    pos   = 0
    for i, word in enumerate(words):
        ws = text.find(word, pos)
        we = ws + len(word)
        if ws > pos:
            parts.append(text[pos:ws])

        cat = ''
        if any(char_cat[c] for c in range(ws, we) if c < len(char_cat)):
            cat = 'indirect'
        elif i in word_flags:
            cat = word_flags[i][0]

        if cat and cat in WORD_CATEGORIES:
            fg, bg, _ = WORD_CATEGORIES[cat]
            parts.append(
                f'<span style="background:{bg};color:{fg};'
                f'border:1px solid {fg}55;border-radius:4px;'
                f'padding:1px 6px;font-weight:700;" title="{WORD_CATEGORIES[cat][2]}">'
                f'{word}</span>'
            )
        else:
            parts.append(word)
        pos = we

    if pos < len(text):
        parts.append(text[pos:])

    return ''.join(parts), flagged, cat_counts


# ──────────────────────────────────────────────────────────
# SUGGESTIONS
# ──────────────────────────────────────────────────────────
def get_suggestions(prob):
    if prob >= 0.75:
        return [
            "🚨 High risk cyberbullying detected",
            "📸 Save evidence",
            "🚫 Block/report the user",
            "🗣️ Talk to someone you trust",
        ]
    elif prob >= THRESHOLD:
        return [
            "⚠️ Potential cyberbullying detected",
            "👀 Monitor the situation",
            "📝 Keep records",
        ]
    return [
        "✅ Content appears safe",
        "👀 Stay aware of tone changes",
    ]


def _signal_row(icon, label, detail, color='#f39c12'):
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;gap:10px;'
        f'background:#1a2332;border-radius:7px;border-left:3px solid {color};'
        f'padding:8px 12px;margin:3px 0;">'
        f'<span style="font-size:18px;line-height:1.4">{icon}</span>'
        f'<div><div style="font-weight:600;font-size:13px;color:#dde">{label}</div>'
        f'<div style="font-size:12px;color:#8899aa">{detail}</div></div></div>',
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────────────────
# SHOW RESULT
# ──────────────────────────────────────────────────────────
def show_result(text, tokenizer, roberta, model, scaler,
                context='Unknown / Public comment'):

    if not text.strip():
        st.warning('No text detected.')
        return

    with st.spinner('Analyzing...'):
        (
            boosted_prob, is_cyber, tone, tone_reason,
            is_indirect, matched_pattern, raw_prob,
            context_boost, mitigation
        ) = predict(text, tokenizer, roberta, model, scaler, context)

        html_highlighted, flagged_words, cat_counts = analyse_words(text)

    # ── live features ──────────────────────────────────────
    from textblob import TextBlob
    blob         = TextBlob(text)
    sentiment    = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    word_count   = len(text.split())
    char_count   = len(text)
    caps_ratio   = sum(1 for c in text if c.isupper()) / max(char_count, 1)
    excl_count   = text.count('!')
    ques_count   = text.count('?')
    profane_cnt  = sum(1 for w in text.split() if profanity.contains_profanity(w))
    hashtag_cnt  = len(__import__('re').findall(r'#\w+', text))
    mention_cnt  = len(__import__('re').findall(r'@\w+', text))
    lexical_div  = len(set(text.lower().split())) / max(word_count, 1)
    margin       = boosted_prob - THRESHOLD

    # ── verdict banner ─────────────────────────────────────
    if boosted_prob >= 0.75:
        verdict_color = '#e74c3c'
        st.error('🚨 High Risk Cyberbullying')
    elif boosted_prob >= THRESHOLD:
        verdict_color = '#f39c12'
        st.warning('⚠️ Potential Cyberbullying')
    elif boosted_prob >= 0.35:
        verdict_color = '#3498db'
        st.info('😐 Mixed Tone / Possible Banter')
    else:
        verdict_color = '#2ecc71'
        st.success('✅ Not Cyberbullying')

    st.progress(float(boosted_prob))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Final Score',  f'{boosted_prob*100:.2f}%')
    k2.metric('Raw RoBERTa',  f'{raw_prob*100:.2f}%')
    k3.metric('Threshold',    f'{THRESHOLD*100:.0f}%')
    k4.metric('Margin',       f'{abs(margin)*100:.2f}%',
              delta='above' if margin >= 0 else 'below',
              delta_color='inverse')

    st.caption(f'📌 Context: **{context}**')
    st.divider()

    # ══════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════
    tab_words, tab_score, tab_text, tab_signals, tab_why, tab_explain = st.tabs([
    '🔦 Word Analysis',
    '📊 Score Breakdown',
    '📝 Text Statistics',
    '🔎 Detected Signals',
    '💡 Why This Score?',
    '📖 Detailed Explanation',   # ← new
])

    # ──────────────────────────────────────────────────────
    # TAB: WORD ANALYSIS
    # ──────────────────────────────────────────────────────
    with tab_words:
        st.markdown('#### Which words triggered the score')

        # legend — only show categories that actually fired
        legend_parts = []
        for cat, (fg, bg, label) in WORD_CATEGORIES.items():
            if cat_counts.get(cat, 0) > 0:
                legend_parts.append(
                    f'<span style="background:{bg};color:{fg};'
                    f'border:1px solid {fg}55;border-radius:4px;'
                    f'padding:2px 10px;font-size:12px;font-weight:700;">'
                    f'{label} &nbsp;({cat_counts[cat]})</span>'
                )
        if legend_parts:
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;">'
                + ''.join(legend_parts) + '</div>',
                unsafe_allow_html=True
            )

        # highlighted text
        st.markdown(
            '<div style="background:#0d1b2a;border:1px solid #2a3a4a;'
            'border-radius:10px;padding:16px 20px;font-size:16px;'
            'line-height:2.4;word-wrap:break-word;">'
            + html_highlighted + '</div>',
            unsafe_allow_html=True
        )

        st.markdown('')

        if flagged_words:
            st.markdown(f'**{len(flagged_words)} flagged word(s) / phrase(s):**')

            # table
            rows = ''
            for item in flagged_words:
                fg2, bg2, lbl2 = WORD_CATEGORIES.get(
                    item['category'], ('#fff', '#222', 'Unknown'))
                rows += (
                    '<tr style="border-bottom:1px solid #1a2a3a;">'
                    '<td style="padding:7px 12px;">'
                    f'<span style="background:{bg2};color:{fg2};'
                    f'border:1px solid {fg2}55;border-radius:4px;'
                    f'padding:1px 8px;font-weight:700;">{item["word"]}</span>'
                    '</td>'
                    f'<td style="padding:7px 12px;color:{fg2};font-weight:600;">{lbl2}</td>'
                    f'<td style="padding:7px 12px;color:#8899aa;">{item["reason"]}</td>'
                    '</tr>'
                )
            st.markdown(
                '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
                '<thead><tr style="background:#1a2332;color:#8899aa;text-align:left;">'
                '<th style="padding:8px 12px;border-bottom:1px solid #2a3a4a;">Word / Phrase</th>'
                '<th style="padding:8px 12px;border-bottom:1px solid #2a3a4a;">Category</th>'
                '<th style="padding:8px 12px;border-bottom:1px solid #2a3a4a;">Reason</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True
            )

            # bar chart
            st.markdown('')
            st.markdown('**Category breakdown:**')
            total_f = max(sum(cat_counts.values()), 1)
            for cat, count in cat_counts.items():
                if count == 0:
                    continue
                fg2, bg2, lbl2 = WORD_CATEGORIES[cat]
                pct = count / total_f * 100
                st.markdown(
                    f'<div style="margin:4px 0;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:12px;color:#8899aa;margin-bottom:3px;">'
                    f'<span>{lbl2}</span>'
                    f'<span>{count} word(s) &nbsp;·&nbsp; {pct:.0f}%</span></div>'
                    f'<div style="background:#1a2332;border-radius:4px;height:9px;">'
                    f'<div style="background:{fg2};width:{pct:.1f}%;height:9px;'
                    f'border-radius:4px;transition:width .4s;"></div></div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.success(
                '✅ No individual words matched threat, hate, exclusion, '
                'or profanity vocabularies. The score is driven by the overall '
                'sentence context learned by RoBERTa.'
            )

    # ──────────────────────────────────────────────────────
    # TAB: SCORE BREAKDOWN
    # ──────────────────────────────────────────────────────
    with tab_score:
        st.markdown('#### How the final score was built')

        components = [
            ('RoBERTa model output',  raw_prob,
             '🤖 Base prediction from RoBERTa + classifier'),
            ('Indirect insult boost', 0.15 if is_indirect else 0.0,
             '🎭 Pattern-matched indirect attack language'),
            ('Context adjustment',    context_boost,
             f'📌 Context: "{context}"'),
            ('Friendly mitigation',   mitigation,
             '😊 Friendly signals softened the score'),
        ]

        total = 0.0
        for name, val, note in components:
            total += val
            sign  = '+' if val >= 0 else ''
            color = '#e74c3c' if val > 0.001 else ('#2ecc71' if val < -0.001 else '#8899aa')
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;background:#1a2332;border-radius:7px;'
                f'padding:9px 14px;margin:3px 0;border-left:3px solid {color};">'
                f'<div><div style="font-weight:600;font-size:13px;color:#dde">{name}</div>'
                f'<div style="font-size:11px;color:#8899aa">{note}</div></div>'
                f'<div style="font-size:18px;font-weight:700;color:{color};'
                f'min-width:70px;text-align:right">{sign}{val*100:.2f}%</div></div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;background:#0d1b2a;border-radius:8px;'
            f'padding:11px 14px;margin:10px 0 0;border:2px solid {verdict_color};">'
            f'<div style="font-weight:700;font-size:14px;color:#dde">'
            f'Final Score (clamped 0–100%)</div>'
            f'<div style="font-size:22px;font-weight:700;color:{verdict_color}">'
            f'{boosted_prob*100:.2f}%</div></div>',
            unsafe_allow_html=True
        )

        st.markdown('')
        _score_pct     = round(boosted_prob * 100, 2)
        _threshold_pct = round(THRESHOLD * 100, 2)
        _margin_pct    = round(_score_pct - _threshold_pct, 2)
        if _margin_pct >= 0:
            st.info(
                f'📏 Score is **{_margin_pct:.2f}% above** the {_threshold_pct:.0f}% threshold '
                f'({_score_pct:.2f}% − {_threshold_pct:.0f}% = {_margin_pct:.2f}%). '
                f'A drop of **{_margin_pct:.2f}%** would flip this to **Not Cyberbullying**.'
            )
        else:
            st.success(
                f'📏 Score is **{abs(_margin_pct):.2f}% below** the {_threshold_pct:.0f}% threshold '
                f'({_threshold_pct:.0f}% − {_score_pct:.2f}% = {abs(_margin_pct):.2f}%). '
                f'A rise of **{abs(_margin_pct):.2f}%** would flip this to **Cyberbullying**.'
            )

    # ──────────────────────────────────────────────────────
    # TAB: TEXT STATISTICS
    # ──────────────────────────────────────────────────────
    with tab_text:
        st.markdown('#### Linguistic features fed to the model')

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('Words',        word_count)
            st.metric('Characters',   char_count)
            st.metric('Unique words', len(set(text.lower().split())))
        with c2:
            st.metric('Caps ratio',    f'{caps_ratio*100:.1f}%',
                      help='High caps (>15%) signals aggression')
            st.metric('Exclamations', excl_count)
            st.metric('Questions',    ques_count)
        with c3:
            st.metric('Profane words', profane_cnt)
            st.metric('Hashtags',      hashtag_cnt)
            st.metric('Mentions (@)',  mention_cnt)

        st.divider()
        sc1, sc2 = st.columns(2)
        sc1.metric('Sentiment polarity', f'{sentiment:+.3f}',
                   help='–1 very negative · 0 neutral · +1 very positive')
        sc1.caption('🔴 Negative' if sentiment < -0.1
                    else '🟢 Positive' if sentiment > 0.1 else '⚪ Neutral')
        sc2.metric('Subjectivity', f'{subjectivity:.3f}',
                   help='0 objective · 1 highly emotional')
        sc2.caption('🔵 Highly subjective' if subjectivity > 0.6
                    else '⚪ Mostly objective')
        st.metric('Lexical diversity', f'{lexical_div:.3f}',
                  help='Unique ÷ total words. Low = repetitive/targeted language')

    # ──────────────────────────────────────────────────────
    # TAB: DETECTED SIGNALS
    # ──────────────────────────────────────────────────────
    with tab_signals:
        st.markdown('#### Every signal checked — fired or not')

        tone_map = {
            'sarcastic': ('😏', 'Sarcastic tone',   '#f39c12',
                          'Sarcasm can mask bullying — no mitigation applied'),
            'mocking':   ('😬', 'Mocking tone',      '#e74c3c',
                          'Mocking emoji detected — belittling language'),
            'friendly':  ('😊', 'Friendly/banter',   '#2ecc71',
                          'Friendly indicators — mitigation applied'),
            'neutral':   ('😐', 'Neutral tone',      '#8899aa', 'No strong tone signal'),
        }
        t_icon, t_label, t_color, t_detail = tone_map.get(
            tone, ('❓', tone, '#8899aa', ''))
        _signal_row(t_icon, f'Tone: {t_label}',
                    t_detail + (f' · reason: {tone_reason}' if tone_reason else ''),
                    t_color)

        if is_indirect:
            _signal_row('🎭', 'Indirect insult matched',
                        f'Pattern: `{matched_pattern}` → +15% boost', '#e74c3c')
        else:
            _signal_row('✔️', 'No indirect insult patterns',
                        'Checked 8 patterns (e.g. "nobody likes you", "ratio")', '#2ecc71')

        ctx_color = '#e74c3c' if context_boost > 0 else '#2ecc71' if context_boost < 0 else '#8899aa'
        _signal_row('📌', f'Context: {context}',
                    f'Adjustment: {context_boost:+.0%}', ctx_color)

        if mitigation < 0:
            _signal_row('😊', 'Friendly mitigation applied',
                        f'Offset: {mitigation:+.0%}', '#2ecc71')
        else:
            _signal_row('➖', 'No friendly mitigation', 'No friendly signals found', '#8899aa')

        if profane_cnt > 0:
            _signal_row('🤬', f'{profane_cnt} profane word(s)',
                        'Direct input feature to the classifier', '#e74c3c')
        else:
            _signal_row('✔️', 'No profanity detected', 'Clean language', '#2ecc71')

        if sentiment < -0.2:
            _signal_row('😡', f'Negative sentiment ({sentiment:+.3f})',
                        'Strong negative signal fed to classifier', '#e74c3c')
        elif sentiment > 0.1:
            _signal_row('😊', f'Positive sentiment ({sentiment:+.3f})',
                        'Reduces bullying likelihood estimate', '#2ecc71')
        else:
            _signal_row('😐', f'Neutral sentiment ({sentiment:+.3f})',
                        'No strong polarity', '#8899aa')

        if caps_ratio > 0.15:
            _signal_row('📢', f'High caps ratio ({caps_ratio*100:.1f}%)',
                        'Aggression signal fed to classifier', '#e74c3c')

        with st.expander('🔬 All sarcasm patterns checked'):
            for p in SARCASM_PATTERNS:
                matched = bool(__import__('re').search(p, text.lower()))
                st.markdown(f"{'🔴' if matched else '⚪'} `{p}` — "
                            f"{'**MATCHED**' if matched else 'not matched'}")

        with st.expander('🔬 All indirect-insult patterns checked'):
            for p in INDIRECT_PATTERNS:
                matched = bool(__import__('re').search(p, text.lower()))
                st.markdown(f"{'🔴' if matched else '⚪'} `{p}` — "
                            f"{'**MATCHED**' if matched else 'not matched'}")

    # ──────────────────────────────────────────────────────
    # TAB: WHY THIS SCORE?
    # ──────────────────────────────────────────────────────
    with tab_why:
        st.markdown('#### Plain-English justification')

        reasons_for     = []
        reasons_against = []

        if raw_prob >= 0.5:
            reasons_for.append(
                f'The RoBERTa model gave a base score of **{raw_prob*100:.1f}%**, '
                f'meaning it found language patterns associated with cyberbullying.')
        else:
            reasons_against.append(
                f'The RoBERTa model gave a base score of only **{raw_prob*100:.1f}%**, '
                f'meaning the language did not strongly resemble bullying content.')

        if profane_cnt > 0:
            reasons_for.append(
                f'**{profane_cnt} profane word(s)** were detected — a direct indicator '
                f'of hostile language.')

        if len(flagged_words) > 0:
            threat_f = [w for w in flagged_words if w['category'] == 'threat']
            hate_f   = [w for w in flagged_words if w['category'] == 'hate']
            if threat_f:
                reasons_for.append(
                    f'Threat/violence vocabulary detected: '
                    f'**{", ".join(w["word"] for w in threat_f)}** — '
                    f'these words explicitly reference harm.')
            if hate_f:
                reasons_for.append(
                    f'Hate/dehumanising vocabulary detected: '
                    f'**{", ".join(w["word"] for w in hate_f)}** — '
                    f'language that demeans or degrades the target.')

        if sentiment < -0.2:
            reasons_for.append(
                f'**Strong negative sentiment** (polarity: {sentiment:+.3f}) — '
                f'the text reads as hostile or attacking.')

        if caps_ratio > 0.15:
            reasons_for.append(
                f'**{caps_ratio*100:.1f}% of characters are capitalised** — '
                f'common marker of aggressive/shouting tone.')

        if is_indirect:
            reasons_for.append(
                f'An **indirect insult pattern** was matched (`{matched_pattern}`) — '
                f'added +15% to the score.')

        if context_boost > 0:
            reasons_for.append(
                f'The context **"{context}"** added +{context_boost*100:.0f}% — '
                f'this setting increases the likelihood of genuine harm.')

        if tone == 'mocking':
            reasons_for.append('A **mocking emoji** was detected, signalling belittling intent.')
        if tone == 'sarcastic':
            reasons_for.append('A **sarcasm pattern** was detected — no mitigation applied '
                               'since sarcasm can mask bullying.')

        if mitigation < 0:
            reasons_against.append(
                f'Friendly signals reduced the score by **{abs(mitigation)*100:.0f}%** '
                f'(tone: {tone}).')
        if context_boost < 0:
            reasons_against.append(
                f'The context **"{context}"** reduced the score by '
                f'**{abs(context_boost)*100:.0f}%** (lower-risk setting).')
        if sentiment > 0.1 and profane_cnt == 0:
            reasons_against.append(
                f'Positive sentiment ({sentiment:+.3f}) and no profanity suggest '
                f'the text is not overtly hostile.')

        if reasons_for:
            st.markdown('**Signals that raised the score:**')
            for r in reasons_for:
                st.markdown(f'- {r}')

        if reasons_against:
            st.markdown('**Signals that lowered the score:**')
            for r in reasons_against:
                st.markdown(f'- {r}')

        st.divider()

        if boosted_prob >= 0.75:
            st.error(
                f'**Conclusion:** Multiple strong risk signals combine to give a final '
                f'score of {boosted_prob*100:.2f}%, well above the {THRESHOLD*100:.0f}% '
                f'threshold. High probability of cyberbullying.')
        elif boosted_prob >= THRESHOLD:
            st.warning(
                f'**Conclusion:** Score of {boosted_prob*100:.2f}% crosses the '
                f'{THRESHOLD*100:.0f}% threshold. Content warrants attention.')
        elif boosted_prob >= 0.35:
            st.info(
                f'**Conclusion:** Score of {boosted_prob*100:.2f}% is below the '
                f'{THRESHOLD*100:.0f}% threshold but above 35% — mixed signals. '
                f'Could be banter or borderline.')
        else:
            st.success(
                f'**Conclusion:** Score of {boosted_prob*100:.2f}% is well below the '
                f'{THRESHOLD*100:.0f}% threshold. No strong cyberbullying indicators.')

    st.divider()

    # suggestions always at bottom
    for s in get_suggestions(boosted_prob):
        st.markdown(
            f'<div style="background:#1e2a3a;border-left:4px solid #2ecc71;'
            f'padding:10px;border-radius:6px;margin:4px 0;">{s}</div>',
            unsafe_allow_html=True
        )

    with tab_explain:
     st.markdown('#### 📖 Detailed Explanation')
     st.caption('A full narrative breakdown combining all metrics and signals.')

    # ── OVERALL VERDICT ────────────────────────────────────
    st.markdown('---')
    st.markdown('### 🏁 Overall Verdict')

    if boosted_prob >= 0.75:
        st.error(
            f'This message scores **{boosted_prob*100:.2f}%**, which is well above '
            f'the **{THRESHOLD*100:.0f}% threshold**. This is considered **high-risk '
            f'cyberbullying**. The combination of language patterns, tone, and context '
            f'strongly suggests harmful intent toward the target.'
        )
    elif boosted_prob >= THRESHOLD:
        st.warning(
            f'This message scores **{boosted_prob*100:.2f}%**, crossing the '
            f'**{THRESHOLD*100:.0f}% threshold**. This is flagged as **potential '
            f'cyberbullying**. While not conclusive, the signals present are concerning '
            f'enough to warrant attention and monitoring.'
        )
    elif boosted_prob >= 0.35:
        st.info(
            f'This message scores **{boosted_prob*100:.2f}%**, below the threshold but '
            f'in an ambiguous range. It may be **banter or mixed-tone communication**. '
            f'Context plays a big role here — the same words between friends may be '
            f'harmless but between strangers could signal hostility.'
        )
    else:
        st.success(
            f'This message scores **{boosted_prob*100:.2f}%**, well below the '
            f'**{THRESHOLD*100:.0f}% threshold**. The content does **not appear to be '
            f'cyberbullying**. The language, tone, and context all suggest this is '
            f'normal communication.'
        )

    # ── MODEL ANALYSIS ─────────────────────────────────────
    st.markdown('---')
    st.markdown('### 🤖 What the AI Model Detected')

    if raw_prob >= 0.75:
        model_explanation = (
            f'The RoBERTa model — trained on thousands of real cyberbullying examples — '
            f'gave this text a raw score of **{raw_prob*100:.2f}%**. This is a very high '
            f'base score, meaning the sentence structure, word choices, and phrasing '
            f'closely resemble confirmed bullying content in its training data.'
        )
    elif raw_prob >= 0.5:
        model_explanation = (
            f'The RoBERTa model gave a raw score of **{raw_prob*100:.2f}%**, meaning it '
            f'found moderate-to-strong signals of bullying language. The model detected '
            f'patterns in the text — such as phrasing, tone, or word combinations — '
            f'that are statistically associated with cyberbullying.'
        )
    elif raw_prob >= 0.35:
        model_explanation = (
            f'The RoBERTa model gave a raw score of **{raw_prob*100:.2f}%**. The model '
            f'found some patterns associated with bullying but they are not strong enough '
            f'to be conclusive on their own. The text sits in an ambiguous zone.'
        )
    else:
        model_explanation = (
            f'The RoBERTa model gave a low raw score of **{raw_prob*100:.2f}%**, meaning '
            f'it found very few or no patterns associated with cyberbullying. The language '
            f'structure does not resemble bullying content.'
        )
    st.markdown(model_explanation)

    # ── TONE ANALYSIS ──────────────────────────────────────
    st.markdown('---')
    st.markdown('### 🎭 Tone Analysis')

    if tone == 'friendly':
        st.markdown(
            f'The tone was detected as **friendly/banter** ({tone_reason}). '
            f'This is a mitigating factor — the presence of friendly language markers '
            f'such as casual expressions, banter emojis, or affectionate terms suggests '
            f'the message may not carry genuine harmful intent. '
            f'The score was reduced by **{abs(mitigation)*100:.0f}%** as a result.'
        )
    elif tone == 'mocking':
        st.markdown(
            f'The tone was detected as **mocking** ({tone_reason}). '
            f'Mocking language — even when disguised as humour — is a known vector '
            f'for cyberbullying. It can belittle, embarrass, or demean the target '
            f'while giving the sender plausible deniability. No mitigation was applied.'
        )
    elif tone == 'sarcastic':
        st.markdown(
            f'The tone was detected as **sarcastic** ({tone_reason}). '
            f'Sarcasm is commonly used to mask bullying intent — a genuinely harmful '
            f'message can be framed as a joke. Because of this, sarcasm does **not** '
            f'trigger friendly mitigation. The score was not reduced.'
        )
    else:
        st.markdown(
            f'The tone was detected as **neutral** — no strong friendly or hostile '
            f'tone markers were found. The score is therefore driven primarily by the '
            f'language content itself rather than emotional delivery.'
        )

    # ── LANGUAGE & VOCABULARY ──────────────────────────────
    st.markdown('---')
    st.markdown('### 📝 Language & Vocabulary Breakdown')

    if profane_cnt > 0:
        st.markdown(
            f'**Profanity:** {profane_cnt} profane word(s) were detected. Profanity '
            f'is a direct signal of hostile language and is fed as a feature directly '
            f'into the classifier. The more profane words present, the higher the '
            f'bullying likelihood.'
        )
    else:
        st.markdown(
            f'**Profanity:** No profane words were detected. This is a positive signal '
            f'that the language is not overtly hostile.'
        )

    threat_words_found = [w for w in flagged_words if w['category'] == 'threat']
    hate_words_found   = [w for w in flagged_words if w['category'] == 'hate']
    excl_words_found   = [w for w in flagged_words if w['category'] == 'exclusion']

    if threat_words_found:
        st.markdown(
            f'**Threat/Violence vocabulary:** The word(s) '
            f'**{", ".join(w["word"] for w in threat_words_found)}** were detected. '
            f'These words explicitly reference physical harm or threats and are among '
            f'the strongest indicators of cyberbullying or harassment.'
        )

    if hate_words_found:
        st.markdown(
            f'**Hate/Dehumanising vocabulary:** The word(s) '
            f'**{", ".join(w["word"] for w in hate_words_found)}** were detected. '
            f'This type of language is used to degrade, demean, or dehumanise the '
            f'target — a hallmark of sustained bullying behaviour.'
        )

    if excl_words_found:
        st.markdown(
            f'**Exclusion/Social attack vocabulary:** The word(s) '
            f'**{", ".join(w["word"] for w in excl_words_found)}** were detected. '
            f'Social exclusion language targets a person\'s sense of belonging and '
            f'self-worth, and is a subtle but damaging form of cyberbullying.'
        )

    if not threat_words_found and not hate_words_found and not excl_words_found and profane_cnt == 0:
        st.markdown(
            '**Vocabulary:** No explicitly harmful words were found in the known '
            'dictionaries. The score is driven by the overall sentence-level patterns '
            'learned by RoBERTa, not individual words.'
        )

    # ── SENTIMENT ──────────────────────────────────────────
    st.markdown('---')
    st.markdown('### 😡 Sentiment & Emotional Tone')

    if sentiment < -0.5:
        st.markdown(
            f'**Sentiment polarity: {sentiment:+.3f}** — This is a **very strongly '
            f'negative** sentiment score. The text reads as highly hostile, angry, or '
            f'attacking. Strong negative sentiment combined with other signals is a '
            f'reliable indicator of harmful intent.'
        )
    elif sentiment < -0.2:
        st.markdown(
            f'**Sentiment polarity: {sentiment:+.3f}** — This is a **moderately '
            f'negative** sentiment. The text leans hostile but stops short of being '
            f'extremely aggressive on sentiment alone.'
        )
    elif sentiment > 0.2:
        st.markdown(
            f'**Sentiment polarity: {sentiment:+.3f}** — This is a **positive** '
            f'sentiment score. Positive sentiment generally works against a bullying '
            f'classification, though it can occasionally mask sarcasm or backhanded '
            f'comments.'
        )
    else:
        st.markdown(
            f'**Sentiment polarity: {sentiment:+.3f}** — Sentiment is roughly '
            f'**neutral**. The classification is driven more by specific vocabulary '
            f'and model patterns than emotional tone.'
        )

    st.markdown(
        f'**Subjectivity: {subjectivity:.3f}** — '
        + (
            'The text is **highly subjective and emotional**, which is common in '
            'personal attacks and targeted harassment.'
            if subjectivity > 0.6
            else 'The text is **moderately subjective**.'
            if subjectivity > 0.3
            else 'The text reads as **mostly objective** with little emotional charge.'
        )
    )

    # ── CONTEXT ────────────────────────────────────────────
    st.markdown('---')
    st.markdown('### 📌 Context Explanation')

    context_explanations = {
        'Unknown / Public comment': (
            'No specific context was provided. The model uses a neutral baseline. '
            'Public comments with no known relationship between sender and recipient '
            'carry a moderate default risk level.'
        ),
        'Between strangers': (
            f'Messages between strangers carry an elevated risk — hostile language '
            f'between people with no prior relationship is less likely to be banter '
            f'and more likely to be genuine harassment. A **+{context_boost*100:.0f}% '
            f'boost** was applied to reflect this.'
        ),
        'Between friends (may contain banter)': (
            f'Friends often use language that would seem hostile out of context — '
            f'insults as terms of endearment, dark humour, and aggressive-sounding '
            f'banter are common. A **{context_boost*100:.0f}% adjustment** was applied '
            f'to account for this, but only when friendly tone signals were also present.'
        ),
        'Directed at me personally': (
            f'When a message is directed personally at someone, the impact is amplified. '
            f'Targeted personal attacks are more harmful than general hostile language. '
            f'A **+{context_boost*100:.0f}% boost** was applied.'
        ),
        'Repeated messages from same person': (
            f'Repeated hostile messages from the same person is a defining characteristic '
            f'of sustained cyberbullying — not just a one-off comment. This context '
            f'carries the highest risk adjustment: **+{context_boost*100:.0f}%**.'
        ),
    }
    st.markdown(context_explanations.get(context, 'No context explanation available.'))

    # ── INDIRECT INSULTS ───────────────────────────────────
    st.markdown('---')
    st.markdown('### 🎭 Indirect Insult Detection')

    if is_indirect:
        st.markdown(
            f'An indirect insult pattern was matched: **`{matched_pattern}`**. '
            f'Indirect insults are phrases that attack without using explicit slurs '
            f'or profanity — things like *"nobody likes you"*, *"ratio"*, or '
            f'*"cope harder"*. These are common in modern online bullying because '
            f'they are harder to flag automatically. A **+15% boost** was applied.'
        )
    else:
        st.markdown(
            'No indirect insult patterns were matched. The message does not contain '
            'the common indirect attack phrases checked by the system (e.g. '
            '"nobody likes you", "no one asked", "stay mad", "ratio").'
        )

    # ── WRITING STYLE ──────────────────────────────────────
    st.markdown('---')
    st.markdown('### ✍️ Writing Style Signals')

    style_points = []

    if caps_ratio > 0.15:
        style_points.append(
            f'**All-caps usage ({caps_ratio*100:.1f}%):** A high proportion of '
            f'capitalised characters is associated with shouting or aggressive tone '
            f'in online communication.'
        )
    if excl_count > 2:
        style_points.append(
            f'**Exclamation marks ({excl_count}):** Multiple exclamation marks '
            f'reinforce an aggressive or emotionally charged delivery.'
        )
    if ques_count > 2:
        style_points.append(
            f'**Question marks ({ques_count}):** Repeated questioning can signal '
            f'confrontational or challenging language.'
        )
    if lexical_div < 0.5:
        style_points.append(
            f'**Low lexical diversity ({lexical_div:.2f}):** Repetitive word use '
            f'can indicate targeted, looping attacks on a specific person or trait.'
        )
    if word_count < 10:
        style_points.append(
            f'**Short message ({word_count} words):** Very short messages that are '
            f'still flagged tend to be highly concentrated hostile content.'
        )

    if style_points:
        for point in style_points:
            st.markdown(f'- {point}')
    else:
        st.markdown(
            'No unusual writing style signals were detected. The message is written '
            'in a relatively normal style without aggressive formatting patterns.'
        )

    # ── FINAL RECOMMENDATION ───────────────────────────────
    st.markdown('---')
    st.markdown('### 🛡️ What Should You Do?')

    if boosted_prob >= 0.75:
        st.error(
            '**Immediate action recommended.** This content shows strong signs of '
            'cyberbullying. You should:\n\n'
            '- 📸 **Save evidence** — screenshot the message with timestamps\n'
            '- 🚫 **Block the sender** on the platform\n'
            '- 🚨 **Report the content** to the platform moderators\n'
            '- 🗣️ **Tell a trusted adult, friend, or counsellor** if you are the target\n'
            '- 🏫 **Escalate to school or authorities** if threats are involved'
        )
    elif boosted_prob >= THRESHOLD:
        st.warning(
            '**Monitor and document.** This content is borderline and warrants caution:\n\n'
            '- 📝 **Keep records** of this and any similar messages\n'
            '- 👀 **Watch for patterns** — repeated behaviour from the same person\n'
            '- 🗣️ **Talk to someone** if the messages are making you uncomfortable\n'
            '- 🚫 **Consider blocking** the sender if messages continue'
        )
    elif boosted_prob >= 0.35:
        st.info(
            '**Stay aware.** The content is ambiguous but not clearly harmful:\n\n'
            '- 👀 **Pay attention to tone changes** in the conversation\n'
            '- 🤔 **Consider the context** — is this normal for this relationship?\n'
            '- 📝 **Note if this is part of a pattern** of behaviour'
        )
    else:
        st.success(
            '**No action needed.** The content appears safe:\n\n'
            '- ✅ This message does not show signs of cyberbullying\n'
            '- 👀 Continue to stay aware of how online conversations make you feel\n'
            '- 🗣️ Always feel free to talk to someone if something feels off'
        )

# ──────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────
if 'context_text' not in st.session_state:
    st.session_state.context_text = 'Unknown / Public comment'

if 'context_img' not in st.session_state:
    st.session_state.context_img = 'Unknown / Public comment'

if 'analyze_text' not in st.session_state:
    st.session_state.analyze_text = False

if 'analyze_img' not in st.session_state:
    st.session_state.analyze_img = False

if 'text_input' not in st.session_state:
    st.session_state.text_input = ''

if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = ''

# ──────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title='CyberGuard',
    page_icon='🛡️',
    layout='centered'
)

st.title(
    '🛡️ CyberGuard — Cyberbullying Detector'
)

st.caption(
    f'RoBERTa + Feature Engineering · '
    f'Accuracy: {config["accuracy"]:.2f}%'
)

with st.spinner(
    'Loading models...'
):

    tokenizer, roberta, model, scaler = load_models()

CONTEXT_OPTIONS = [
    'Unknown / Public comment',
    'Between strangers',
    'Between friends (may contain banter)',
    'Directed at me personally',
    'Repeated messages from same person',
]

# tabs
tab1, tab2, tab3 = st.tabs([
    '💬 Text',
    '🖼️ Screenshot',
    '📊 Analysis',     
])

with tab1:
    # just input fields, no show_result() here
    text_input = st.text_area('Enter text:', height=150)
    selected_context_text = st.selectbox('📌 Context', options=CONTEXT_OPTIONS)

    if st.button('Analyze Text', use_container_width=True):
        st.session_state.result_text = text_input
        st.session_state.result_context = selected_context_text

with tab2:
    # just upload + OCR, no show_result() here
    uploaded_file = st.file_uploader('Upload Screenshot', type=['png','jpg','jpeg','webp'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        ocr_text = extract_text_from_image(image)
        if ocr_text:
            st.session_state.result_text = ocr_text
            selected_context_img = st.selectbox('📌 Context', options=CONTEXT_OPTIONS)
            if st.button('Analyze Screenshot', use_container_width=True):
                st.session_state.result_context = selected_context_img

with tab3:
    # 5 analysis tabs appear here
    if 'result_text' in st.session_state and st.session_state.result_text:
        show_result(
            st.session_state.result_text,
            tokenizer, roberta, model, scaler,
            st.session_state.get('result_context', 'Unknown / Public comment')
        )
    else:
        st.info('Analyze some text or a screenshot first.')

        if 'result_text' not in st.session_state:
         st.session_state.result_text = ''

    if 'result_context' not in st.session_state:
        st.session_state.result_context = 'Unknown / Public comment'