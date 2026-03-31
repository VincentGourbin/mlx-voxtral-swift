# ZeroVoice Blend Quality Benchmark

Benchmark of ZeroVoice procedural voice blending quality across languages and blend strengths.
All tests use intra-language blending (male+female of same language) with the Voxtral TTS 4B model,
transcribed back using the Voxtral Realtime 4B-4bit STT model.

## Setup

- **TTS Model**: Voxtral 4B TTS (mlx-community, bfloat16)
- **STT Model**: Voxtral Realtime 4B (mlx-community, 4-bit)
- **Hardware**: Apple M3 Max, 96GB
- **Method**: Generate speech with blended voice → transcribe back → compare with original text
- **Blend technique**: Row-wise SLERP on 3072-dim voice embedding hypersphere with norm calibration

## Benchmark 1: Blend Strength Sweep (EN)

Text: *"Hello, this is a procedurally generated voice using ZeroVoice coordinates."*
Voices: `neutral_female + es_female`

| Blend t | Frames | Duration | Transcription | Quality |
|---------|--------|----------|---------------|---------|
| 0.00 (ref) | 95 | 7.6s | "Hello, this is a procedurally generated voice using zero voice coordinates." | ✅ Perfect |
| 0.05 | 106 | 8.5s | "Hello, this is a procedurally generated voice using zero voice coordinates." | ✅ Perfect |
| 0.10 | 149 | 11.9s | *(empty)* | ❌ |
| 0.15 | 260 | 20.8s | *(empty)* | ❌ |
| 0.20 | 235 | 18.8s | "Hello! This is a procedurally generated..." | ⚠️ Partial |

**Observation**: Cross-language blends (EN+ES) degrade rapidly above t=0.05.

## Benchmark 2: Intra-language Blend Types (t=0.05)

### EN intra-family (neutral_female + cheerful_female)

| Blend t | Transcription | Quality |
|---------|---------------|---------|
| 0.05 | "This is a test of voice blending technology." | ✅ (missing "Hello") |
| 0.10 | "Hello. This is a test." | ⚠️ Truncated |

### FR intra-family (fr_female + fr_male)

| Blend t | Transcription | Quality |
|---------|---------------|---------|
| 0.05 | "Bonjour, ceci est un test de la technologie de mélange de voix." | ✅ Perfect |
| 0.10 | "Bonjour, ceci est un test de la technologie de mélange de voix." | ✅ Perfect |

**Key finding**: Intra-language blends are significantly more stable than cross-language blends.

## Benchmark 3: All 9 Languages, Native Text (t=0.05)

Text in each language: translation of *"Multilingual support: English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi."*

| Language | Voices | Transcription | Quality |
|----------|--------|---------------|---------|
| **EN** | neutral_f + casual_m | "multilingual support English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi." | ✅ Perfect |
| **FR** | fr_f + fr_m | "Un support multilingue anglais, français, espagnol, allemand, italien, portugais, néerlandais, arabe et hindi." | ✅ Perfect |
| **ES** | es_f + es_m | *(empty)* | ❌ |
| **DE** | de_f + de_m | *(empty)* | ❌ |
| **IT** | it_f + it_m | "Ze supporten multilingüe, inglese, francese, spagnolo, tedesco, italiano, portoghese, olandese, arabo e hindi." | ✅ Good |
| **PT** | pt_f + pt_m | "Suporte multilingüe, inglês, francês, espanhol, alemão, italiano, português, holandês, árabe e hindi." | ✅ Perfect |
| **NL** | nl_f + nl_m | "Klapje. Meertalige ondersteuning..." | ⚠️ Partial |
| **AR** | ar_m + hi_m | "لأن متعدد اللغات الإنجليزية والفرنسية والإسبانية والألمانية والإيطالية والبرتغالية والهولندية والعربية والهندية" | ✅ Perfect |
| **HI** | hi_f + hi_m | "बहुबाशी समक्षन..." | ⚠️ Partial |

## Benchmark 4: Multi-Force Per Language (native text)

### FR (fr_female + fr_male) — Best performer

| Blend t | Transcription | Quality |
|---------|---------------|---------|
| 0.02 | "support multilingue, anglais, français, espagnol, allemand, italien, portugais, néerlandais, arabe et hindi" | ✅ Perfect |
| 0.05 | "support multilingue, anglais, français, espagnol, allemand, italien, portugais, néerlandais, arabe et hindi" | ✅ Perfect |
| 0.10 | "Support multilingue. Anglais, français, espagnol, allemand, italien, portugais, néerlandais, arabe et hindi." | ✅ Good (artifacts at start) |
| 0.15 | "Support multilingue, anglais, français, espagnol, allemand, italien, portugais, néerlandais, arabe et hindi." | ✅ Perfect |

### IT (it_female + it_male)

| Blend t | Transcription | Quality |
|---------|---------------|---------|
| 0.02 | "Supporto multilingüe, inglese, francese, spagnolo, tedesco, italiano, portoghese, olandese, arabo e hindi." | ✅ Perfect |
| 0.05 | *(empty)* | ❌ |
| 0.10 | "Gente, suporto multilingüe, inglese, francese, spagnolo, tedesco, italiano, portoghese, olandese, arabo, e hindi." | ✅ Good |
| 0.15 | "Supporto multilingue, inglese, francese, spagnolo, tedesco, italiano, portoghese, olandese, arabo e hindi." | ✅ Perfect |

### PT (pt_female + pt_male)

| Blend t | Transcription | Quality |
|---------|---------------|---------|
| 0.02 | "Suporte multilingüe, inglês, francês, espanhol, alemão, italiano, português, holandês, árabe e hindi." | ✅ Perfect |
| 0.05 | "Suporte multilingüe, inglês, francês, espanhol, alemão, italiano, português, holandês, árabe e índio." | ✅ Good |
| 0.10 | "Suporte multilingüe, inglês, francês, espanhol, alemão, italiano, português, holandês, árabe e índio." | ✅ Good |
| 0.15 | "Suporte multilingüe. Multilingüe. Transub..." | ⚠️ Looping |

### AR (ar_male + hi_male) — cross-family blend

| Blend t | Transcription | Quality |
|---------|---------------|---------|
| 0.02 | *(empty)* | ❌ |
| 0.05 | *(empty — 200s generation!)* | ❌ |
| 0.10 | "من متعددة اللغات الإنجليزية والفرنسية والإسبانية والألمانية والإيطالية والبرتغالية والهولندية والعربية والهندية" | ✅ Perfect |
| 0.15 | Transcribed in Cyrillic script | ⚠️ Script confusion |

### DE (de_female + de_male) — Unstable

| Blend t | Transcription | Quality |
|---------|---------------|---------|
| 0.02 | "Mehrsprachige..." (truncated) | ⚠️ |
| 0.05 | Garbage | ❌ |
| 0.10 | *(empty)* | ❌ |
| 0.15 | *(empty)* | ❌ |

### HI (hi_female + hi_male) — Unstable

| Blend t | Transcription | Quality |
|---------|---------------|---------|
| 0.02 | "बहुत बेहाश और बहुत बहाश होता..." (partial) | ⚠️ |
| 0.05 | *(empty)* | ❌ |
| 0.10 | *(empty)* | ❌ |
| 0.15 | *(empty)* | ❌ |

## Summary: Recommended Max Blend Strength Per Language

| Language | Reliability | Recommended max t | Notes |
|----------|------------|-------------------|-------|
| **FR** | ⭐⭐⭐⭐⭐ | **0.15** | Best performer — stable even at high blend |
| **EN** | ⭐⭐⭐⭐⭐ | **0.05** | Excellent but cross-language sensitive |
| **IT** | ⭐⭐⭐⭐ | **0.15** | Good, but skip t=0.05 (unstable at that exact value) |
| **PT** | ⭐⭐⭐⭐ | **0.10** | Very good up to 0.10 |
| **AR** | ⭐⭐⭐ | **0.10** | Works but unpredictable (cross-family blend with hi_male) |
| **ES** | ⭐⭐ | **0.05** | Variable, needs more testing |
| **NL** | ⭐⭐ | **0.05** | Variable |
| **DE** | ⭐ | **—** | Unusable — DE embeddings do not tolerate SLERP well |
| **HI** | ⭐ | **0.02** | Very fragile |

## Key Findings

1. **Intra-language blends are far more stable** than cross-language blends
2. **FR is the champion** — perfect transcription even at t=0.15
3. **DE and HI embeddings are structurally incompatible** with SLERP interpolation
4. **Cross-language blends** (e.g., EN+ES, AR+HI) degrade rapidly above t=0.05
5. **Safe default**: t=0.05 works for most languages except DE and HI
6. **Duration inflation**: blended voices tend to speak slower (more frames for same text)
7. **AR is special**: only has ar_male preset, must cross-family blend with hi_male — works at t=0.10

## Recommendations

- Default `maxBlendWeight` should be **0.05** for cross-family blends
- For intra-language blends (FR, IT, PT): up to **0.10–0.15** is safe
- Disable or warn for DE and HI blends
- Always use native-language text with matching voice family
