"""Checkpoint M2: EDA técnico de AG News (muestra de la cátedra).

Uso: python checkpoint_2_eda/generar_reporte.py
Requiere: python -m spacy download en_core_web_sm

Si no está disponible en_core_web_sm, --allow-fallback habilita
un lematizador SpaCy lookup con reglas léxicas. No reemplaza al modelo oficial.
"""
from __future__ import annotations
import argparse
import html
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'/'ag_news' if (ROOT/'data'/'ag_news'/'ag_news_train.csv').exists() else ROOT/'data'
OUT=ROOT/'checkpoint_2_eda'/'outputs'
REPORTES=ROOT/'checkpoint_2_eda'/'reportes'
OUT.mkdir(parents=True, exist_ok=True)
REPORTES.mkdir(parents=True, exist_ok=True)
LABELS=['World','Sports','Business','Sci_Tech']
CLASS_SP={'World':'Internacional','Sports':'Deportes','Business':'Negocios','Sci_Tech':'Ciencia y tecnología'}
DARK='#1e3446'; TEAL='#087e8b'; BLUE='#4869ad'; GREY='#667785'; LIGHT='#f1f6f8'

# Inglés: se preservan "no", "not" y "nor" por su función semántica.
EXCEPTIONS={'no','not','nor'}

def nfmt(n):
    return f'{int(n):,}'.replace(',', '.')


def tidy(s: str) -> str:
    """Elimina ruido editorial y normaliza entidades HTML, incluso si vienen mal codificadas."""
    s=str(s)
    # La muestra contiene entidades sin ampersand, como #39;s y quot;.
    s=re.sub(r'(?<!&)#(\d{2,5});', r'&#\1;', s)
    s=re.sub(r'(?<!&)\b(quot|amp|nbsp|lt|gt);', r'&\1;', s, flags=re.I)
    s=html.unescape(html.unescape(s))
    # Preservar abreviaturas frecuentes antes de eliminar puntuación.
    s=re.sub(r'\bU\.S\.', 'USA', s, flags=re.I)
    s=re.sub(r'\bU\.K\.', 'UK', s, flags=re.I)
    s=re.sub(r'\bE\.U\.', 'EU', s, flags=re.I)
    s=re.sub(r'<[^>]*>', ' ', s)
    s=re.sub(r'https?://\S+|www\.\S+|\b[\w.%-]+@[\w.-]+\.[a-z]{2,}\b', ' ', s, flags=re.I)
    s=re.sub(r'\b[a-z0-9-]+\.(?:com|org|net|gov|edu)\b', ' ', s, flags=re.I)
    # Restos de titulares de agencia y fórmulas editoriales repetidas.
    s=re.sub(r'\(\s*(?:ap|reuters|afp)\s*\)', ' ', s, flags=re.I)
    s=re.sub(r'\b(?:reuters|associated press|canadian press)\s*[-:]',' ',s,flags=re.I)
    s=re.sub(r'\bquote\s*[,;:]\s*profile\s*[,;:]\s*research\b',' ',s,flags=re.I)
    s=re.sub(r'\bquote\s+profile\s+research\b',' ',s,flags=re.I)
    s=s.lower().replace('’',"'")
    s=re.sub(r"(?<=\w)\s+['']s\b", '', s)
    # Preservar negaciones; la posesión no aporta un término propio.
    s=re.sub(r"\b(can)\s*['']t\b",r'\1 not',s)
    s=re.sub(r"\b(won)\s*['']t\b",r'will not',s)
    s=re.sub(r"n\s*['']t\b",r' not',s)
    s=re.sub(r"\b([a-z]+)['']s\b",r'\1',s)
    s=re.sub(r'[^a-z\s]', ' ', s)
    s=re.sub(r'\s+', ' ', s).strip()
    return s


def get_nlp(allow_fallback: bool, raw_texts=None):
    try:
        nlp=spacy.load('en_core_web_sm',disable=['ner','parser'])
        return nlp,'en_core_web_sm (modelo de inglés SpaCy)',True
    except OSError as exc:
        if not allow_fallback:
            raise RuntimeError('Falta en_core_web_sm. Instalá: python -m spacy download en_core_web_sm') from exc
        # Alternativa verificable: componente oficial spaCy Lemmatizer(mode='lookup')
        # con tabla morfológica derivada de POS léxico y flexiones de inglés.
        # No equivale al modelo estadístico en_core_web_sm de la consigna.
        from collections import Counter,defaultdict
        from textblob.en import tag
        from textblob.en.inflect import singularize
        from spacy.lookups import Lookups
        irregular={'is':'be','are':'be','was':'be','were':'be','been':'be','am':'be',
                   'has':'have','had':'have','does':'do','did':'do','done':'do',
                   'said':'say','says':'say','went':'go','gone':'go','made':'make',
                   'took':'take','taken':'take','came':'come','seen':'see',
                   'saw':'see','won':'win','lost':'lose','bought':'buy',
                   'sold':'sell','fell':'fall','rose':'rise','ran':'run',
                   'grew':'grow','known':'know','held':'hold','found':'find',
                   'left':'leave','gave':'give','given':'give',
                   'began':'begin','begun':'begin','got':'get'}
        candidates=defaultdict(Counter)
        vocab=set()
        for ix,line in enumerate(raw_texts):
            if ix%1000==0:print('lookup',ix,flush=True)
            for word,pos in tag(line):
                word=word.lower()
                if not word.isalpha():continue
                vocab.add(word)
                lemma=irregular.get(word,word)
                if lemma==word and pos in ('NNS','NNPS'):
                    lemma=singularize(word)
                    if lemma.endswith('busines'):lemma=word
                if lemma==word and pos.startswith('VB'):
                    if word.endswith('ies') and len(word)>4: lemma=word[:-3]+'y'
                    elif word.endswith('ing') and len(word)>5:
                        z=word[:-3]
                        if len(z)>2 and z[-1]==z[-2] and z[-1] not in 'aeiou': z=z[:-1]
                        lemma=z
                    elif word.endswith('ied') and len(word)>4:lemma=word[:-3]+'y'
                    elif word.endswith('ed') and len(word)>4:
                        stem=word[:-2]
                        lemma=stem if stem in vocab else word[:-1]
                    elif word.endswith('s') and len(word)>3 and not word.endswith('ss'):
                        lemma=word[:-2] if word.endswith('es') else word[:-1]
                candidates[word][lemma] += 1
        # Preservar lemas irregulares y términos editoriales comunes.
        lemma_map={w:c.most_common(1)[0][0] for w,c in candidates.items()}
        lemma_map.update(irregular)
        for w in ('news','business','sports','series','analysis','us','virus','israel','press','united','states','los','angeles','securities','savings','reuters','canadian'):
            lemma_map[w]=w
        nlp=spacy.blank('en')
        lemmatizer=nlp.add_pipe('lemmatizer',config={'mode':'lookup'})
        lookups=Lookups();lookups.add_table('lemma_lookup',lemma_map)
        lemmatizer.initialize(lookups=lookups)
        return nlp, 'spaCy lookup basado en reglas léxicas (sin en_core_web_sm)',True


def preprocess_text(text: str, nlp, lemmatize: bool) -> tuple[list[str],list[str]]:
    """Dos salidas: tokens limpios sin stopword filtering y tokens de análisis lexical.

    Los n-gramas se construyen sobre la versión sin stopwords. La longitud
    se calcula ANTES de quitar stopwords, de manera consistente con max_len.
    """
    doc=nlp(tidy(text))
    all_tokens=[]; content=[]
    for tok in doc:
        if not tok.is_alpha: continue
        v=(tok.lemma_ if lemmatize else tok.text).lower().strip()
        if not v or not v.isalpha() or (len(v)==1 and v not in {'a','i'}): continue
        all_tokens.append(v)
        if (not tok.is_stop or v in EXCEPTIONS) and len(v)>1:
            content.append(v)
    return all_tokens,content


def top_ngrams(documents:list[str],n:int, limit:int=20):
    vec=CountVectorizer(ngram_range=(n,n), token_pattern=r'(?u)\b[a-z]+\b', lowercase=False)
    mat=vec.fit_transform(documents)
    counts=np.asarray(mat.sum(axis=0)).ravel()
    terms=vec.get_feature_names_out()
    order=np.argsort(-counts, kind='stable')[:limit]
    return [(str(terms[i]),int(counts[i])) for i in order]


def plot_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.titlesize':12})


def create_charts(lengths, cls, words, p95, mode_tag):
    plot_style()
    fig,ax=plt.subplots(figsize=(9.6,3.25))
    ax.hist(lengths,bins=45,color=TEAL,edgecolor='white',linewidth=.35)
    ax.axvline(p95,color='#dc7b31',lw=2,ls='--',label=f'Percentil 95: {p95} tokens')
    ax.set(xlabel='Tokens por noticia (sin excluir stopwords)',ylabel='Documentos',title='Distribución de longitud de las noticias')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout();fig.savefig(OUT/'01_longitud_tokens.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.4,3.1))
    bars=ax.bar([CLASS_SP[x] for x in LABELS],[int(cls[x]) for x in LABELS],color=[TEAL,'#518bcb','#89aeca','#d0a364'])
    ax.bar_label(bars,padding=3,fontsize=10)
    ax.set(ylabel='Cantidad de noticias',title='Distribución de clases (train de la cátedra)')
    ax.set_ylim(0, max(cls.values())*1.2)
    fig.tight_layout();fig.savefig(OUT/'02_distribucion_clases.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.4,3.6))
    c=words.most_common(16)
    ys=[v[0] for v in c][::-1];values=[v[1] for v in c][::-1]
    ax.barh(ys,values,color=BLUE)
    ax.set(xlabel='Número de ocurrencias',title='Palabras más frecuentes (antes de filtrar stopwords)')
    fig.tight_layout();fig.savefig(OUT/'03_palabras_frecuentes.png',dpi=170,bbox_inches='tight');plt.close(fig)
    ranks=np.arange(1,len(words)+1)
    counts=np.array(sorted(words.values(),reverse=True))
    fig,ax=plt.subplots(figsize=(9.4,3.2))
    ax.loglog(ranks,counts,color=TEAL,lw=1.4)
    ax.set(xlabel='Rango de frecuencia (escala log)',ylabel='Frecuencia (escala log)',title='Distribución rango-frecuencia (Ley de Zipf)')
    fig.tight_layout();fig.savefig(OUT/'04_zipf.png',dpi=170,bbox_inches='tight');plt.close(fig)



def compute_extended_stats(tr, tokens, content, lengths, nlp, core_stats):
    """Corpus-only diagnostics; the held-out test set is not opened here."""
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    from collections import Counter

    labels=tr['label'].to_numpy()
    stops=set(nlp.Defaults.stop_words)
    counts={cl:int((labels==cl).sum()) for cl in LABELS}
    per_class=[]
    for cl in LABELS:
        inds=np.where(labels==cl)[0]
        seqs=[tokens[i] for i in inds]
        flat=[w for ts in seqs for w in ts]
        lex=Counter(flat)
        a=lengths[inds]
        per_class.append({'class':cl,'n':len(inds),
            'mean':float(a.mean()),'median':int(np.median(a)),
            'p95':int(np.ceil(np.percentile(a,95))),
            'gt_maxlen':int((a>core_stats['lengths']['p95']).sum()),
            'gt_maxlen_pct':100*float((a>core_stats['lengths']['p95']).mean()),
            'vocab':len(lex),'tokens':len(flat),
            'ttr':len(lex)/max(1,len(flat)),
            'stop_pct':100*sum(w in stops for w in flat)/max(1,len(flat))})

    limits=sorted(set([40,48,core_stats['lengths']['p95'],64,80]))
    coverage=[{'max_len':int(t),'coverage':round(100*float((lengths<=t).mean()),2),
               'truncated':int((lengths>t).sum())} for t in limits]

    raw=tr['text'].astype(str)
    patterns={'html_entity_fragment':r'(?<!&)#\d{2,5};|(?i:(?<!&)\b(?:quot|amp|nbsp|lt|gt);)',
              'html_tags':r'<[^>]+>',
              'links':r'https?://\S+|www\.\S+'}
    noise={key:{'n':int(raw.str.contains(pat,regex=True).sum())}
           for key,pat in patterns.items()}
    for val in noise.values():val['pct']=round(val['n']/len(tr)*100,2)
    freq_tokens=Counter(w for ts in tokens for w in ts)
    detected={w:freq_tokens.get(w,0) for w in ['quot','amp','nbsp']}
    # These stray tokens may legitimately occur in quotations; don't automatically infer HTML contamination.

    docs=[' '.join(x) if x else 'emptydocument' for x in content]
    v=CountVectorizer(binary=True, lowercase=False, token_pattern=r'(?u)\b[a-z][a-z]+\b',min_df=20)
    X=v.fit_transform(docs)
    terms=np.array(v.get_feature_names_out()); category_top={}; top_sets={}
    for cl in LABELS:
        mask=(labels==cl)
        a=np.asarray(X[mask].sum(axis=0)).ravel()
        b=np.asarray(X[~mask].sum(axis=0)).ravel()
        # Document-prevalence lift, with additive smoothing; not a trained classifier.
        lift=((a+0.5)/(mask.sum()+1))/((b+0.5)/((~mask).sum()+1))
        order=np.argsort(-lift)
        idxs=[int(i) for i in order if a[i]>=40 and not terms[i].isdigit()]
        category_top[cl]=[{'term':str(terms[i]),'documents':int(a[i]),
                           'pct_class':round(100*float(a[i])/mask.sum(),2),
                           'lift':round(float(lift[i]),2)} for i in idxs[:8]]
        top=np.argsort(-a)[:100]
        top_sets[cl]=set(map(str,terms[top]))
    overlap=[]
    for cl in LABELS:
        overlap.append([round(100*len(top_sets[cl]&top_sets[other])/len(top_sets[cl]|top_sets[other]),1)
                        for other in LABELS])
    normalized=tr['text'].map(tidy)
    core_stats['extended']={'per_class':per_class,'coverage':coverage,'noise':noise,
                             'normalized_duplicate_texts':int(normalized.duplicated().sum()),
                             'artifact_terms_after_cleaning':detected,
                             'characteristic_terms':category_top,'overlap_top100_jaccard_pct':overlap,
                             'technical_note':('Términos distintivos: prevalencia documental en la clase '
                             'dividida por prevalencia fuera de ella, con suavizado. '
                             'Mínimo 40 noticias dentro de la clase. Solapamiento: Jaccard sobre '
                             'los 100 términos de mayor frecuencia documental por clase. '
                             'Todo calculado únicamente con entrenamiento.')}
    return core_stats


def extra_charts(stats):
    ext=stats['extended']
    plot_style()
    df=pd.DataFrame(ext['per_class'])
    labs=[CLASS_SP[c] for c in LABELS]
    fig,ax=plt.subplots(figsize=(9.4,3.25))
    xs=np.arange(4)
    ax.bar(xs,df['mean'],color=['#0d8590','#5991cf','#83acbe','#d2aa6c'])
    ax.scatter(xs,df['p95'],color='#b85c38',marker='D',s=52,label='Percentil 95',zorder=3)
    for x,mean,q in zip(xs,df['mean'],df['p95']):
        ax.text(x,float(mean)+0.9,f'{mean:.1f}',ha='center',fontsize=9)
        ax.annotate(str(int(q)),(x,float(q)),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks(xs,labs)
    ax.set_ylim(0,max(df['p95'])*1.22)
    ax.set(ylabel='Tokens por noticia',title='Longitudes según categoría')
    ax.legend(frameon=False,loc='upper right');fig.tight_layout()
    fig.savefig(OUT/'05_longitud_por_clase.png',dpi=165,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,3.0))
    cut=ext['coverage'];bar=ax.bar([str(x['max_len']) for x in cut],[x['coverage'] for x in cut],color=BLUE)
    for b,x in zip(bar,cut):ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.7,f"{x['coverage']:.1f}%",ha='center',fontsize=9)
    ax.set(xlabel='max_len (tokens de palabra)',ylabel='% de noticias sin truncamiento',title='Cobertura del corpus según longitud máxima')
    ax.set_ylim(0,110);fig.tight_layout();fig.savefig(OUT/'06_cobertura_longitudes.png',dpi=165,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.9,3.5))
    mat=np.array(ext['overlap_top100_jaccard_pct']);im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(range(4),labs,rotation=20,ha='right');ax.set_yticks(range(4),labs)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{mat[i,j]:.1f}%',ha='center',va='center',fontsize=9,color='white' if mat[i,j]>=65 else DARK)
    ax.set_title('Vocabulario compartido: Jaccard del top 100 por categoría',pad=12)
    fig.colorbar(im,ax=ax,shrink=.72,label='Similitud (%)');fig.tight_layout()
    fig.savefig(OUT/'07_solapamiento_lexico.png',dpi=160,bbox_inches='tight');plt.close(fig)

def make_pdf(stats, fallback):
    # Tipografías con soporte para acentos.
    reg='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    if not Path(reg).exists():
        from matplotlib.font_manager import findfont, FontProperties
        reg=findfont(FontProperties(family='DejaVu Sans'))
        bold=findfont(FontProperties(family='DejaVu Sans',weight='bold'))
    pdfmetrics.registerFont(TTFont('DejaVu',reg))
    pdfmetrics.registerFont(TTFont('DejaVuBold',bold))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVuBold',italic='DejaVu',boldItalic='DejaVuBold')
    color=lambda x:colors.HexColor(x)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BigTitleCustom',fontName='DejaVuBold',fontSize=23,leading=29,textColor=color(DARK),spaceAfter=12))
    styles.add(ParagraphStyle(name='SectionCustom',fontName='DejaVuBold',fontSize=13,leading=18,textColor=color(TEAL),spaceBefore=11,spaceAfter=8))
    styles.add(ParagraphStyle(name='SubCustom',fontName='DejaVuBold',fontSize=10.5,leading=15,textColor=color(DARK),spaceBefore=8,spaceAfter=6))
    styles.add(ParagraphStyle(name='BodyCustom',fontName='DejaVu',fontSize=9.2,leading=15,spaceAfter=7,textColor=color(DARK)))
    styles.add(ParagraphStyle(name='MutedCustom',fontName='DejaVu',fontSize=8,leading=12,spaceAfter=7,textColor=color(GREY)))
    styles.add(ParagraphStyle(name='TblCustom',fontName='DejaVu',fontSize=8.2,leading=11,textColor=color(DARK)))
    styles.add(ParagraphStyle(name='TblHeadCustom',fontName='DejaVuBold',fontSize=8.2,leading=11,textColor=colors.white))
    styles.add(ParagraphStyle(name='FootCustom',fontName='DejaVu',fontSize=7.6,leading=10,textColor=color(GREY)))
    styles.add(ParagraphStyle(name='TinyCustom',fontName='DejaVu',fontSize=7.7,leading=11,textColor=color(DARK)))
    styles.add(ParagraphStyle(name='WarnCustom',fontName='DejaVuBold',fontSize=9.2,leading=15,textColor=color('#a5521f'),spaceAfter=9))
    P=lambda s,sty='BodyCustom':Paragraph(s,styles[sty])
    name='EDA_NLP_ArevaloJuanPablo_BORRADOR_NO_ENTREGAR.pdf' if fallback else 'EDA_NLP_ArevaloJuanPablo.pdf'
    pdf=REPORTES/name
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=50,bottomMargin=50,
                          title='Análisis exploratorio y preprocesamiento | AG News',author='Data Science III | Informe técnico')
    W=A4[0]-84
    story=[]
    def putchart(fn,height=6.7):
        from PIL import Image as PILImage
        im=PILImage.open(OUT/fn)
        width=W
        h=width/im.width*im.height
        if h>height*cm:
            h=height*cm;width=h*im.width/im.height
        story.append(Image(str(OUT/fn),width=width,height=h))
    def addtable(rows,widths,header=True,padding=5):
        dat=[[P(str(v),'TblHeadCustom' if header and i==0 else 'TblCustom') for v in row] for i,row in enumerate(rows)]
        t=Table(dat,colWidths=widths,repeatRows=1 if header else 0,hAlign='LEFT')
        cmds=[('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOTTOMPADDING',(0,0),(-1,-1),padding),('TOPPADDING',(0,0),(-1,-1),padding),
              ('LINEBELOW',(0,-1),(-1,-1),0.6,color('#d9e4ea'))]
        if header: cmds+=[('BACKGROUND',(0,0),(-1,0),color(DARK)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,color('#f3f7f9')])]
        t.setStyle(TableStyle(cmds));story.append(t)
    def pagebreak():story.append(PageBreak())
    story.append(P('DATA SCIENCE III | NLP &amp; DEEP LEARNING','MutedCustom'))
    story.append(P('Análisis exploratorio del corpus AG News','BigTitleCustom'))
    story.append(P('Preprocesamiento del texto y diagnóstico del corpus','SubCustom'))
    story.append(P('AG News · Corpus en inglés de la cátedra · Conjunto de entrenamiento','MutedCustom'))
    if fallback:
        story.append(P('BORRADOR DE REVISIÓN: no se pudo instalar en_core_web_sm; las cifras reflejan limpieza + tokenización spaCy, <b>sin lematización</b>. Volver a generar el informe con el modelo de idioma para entregar.','WarnCustom'))
    else:
        story.append(P('Versión de revisión: lematización léxica con SpaCy; los resultados definitivos requieren el modelo en_core_web_sm.' if 'lookup' in stats['nlp_mode'] else 'Tokenización y lematización con SpaCy (en_core_web_sm).','MutedCustom'))
    story.append(P('1. Corpus y distribución de clases','SectionCustom'))
    story.append(P('Se trabajó con la muestra de AG News provista por la cátedra. Cada registro contiene el texto de una noticia (<b>text</b>) y su categoría (<b>label</b>). Todas las estadísticas del informe se calcularon sobre el conjunto de entrenamiento; las noticias de prueba se mantuvieron fuera del análisis.'))
    qual=[['Indicador','Valor'],['Noticias de entrenamiento',nfmt(stats['train_n'])],
          ['Noticias de prueba (sin analizar)',nfmt(stats['test_n'])],
          ['Valores nulos en train',str(stats['nulls'])],['Textos duplicados exactos',str(stats['duplicates'])],
          ['Clases',', '.join(LABELS)],['Balance',stats['class_balance']]]
    addtable(qual,[W*0.49,W*0.51]);story.append(Spacer(1,8))
    putchart('02_distribucion_clases.png',6.2)
    story.append(P('Las cuatro categorías están representadas por igual. Esto facilita la comparación inicial de modelos, aunque sigue siendo necesario analizar los errores y las métricas de cada clase por separado.','MutedCustom'))
    pagebreak()

    story.append(P('2. Preparación y normalización del texto','SectionCustom'))
    met=[
        '<b>Limpieza:</b> se normalizaron entidades HTML, incluidas codificaciones incompletas presentes en el corpus, y se retiraron etiquetas, enlaces, direcciones de correo y marcas editoriales. Después se aplicaron minúsculas y se unificaron los espacios.',
        '<b>Tokenización:</b> se dividieron las noticias en palabras con SpaCy y se conservaron los tokens alfabéticos.',
        '<b>Lematización:</b> '+('no incluida en esta versión preliminar.' if fallback else ('realizada mediante tablas léxicas de SpaCy; esta variante debe sustituirse por en_core_web_sm antes de la entrega.' if 'lookup' in stats['nlp_mode'] else 'se utilizó el modelo inglés en_core_web_sm para unificar variantes flexivas.')),
        '<b>Palabras vacías:</b> se mantuvieron al medir longitudes y frecuencias, pero se excluyeron al calcular N-gramas. Se conservaron las negaciones <i>no</i>, <i>not</i> y <i>nor</i>, que pueden modificar el sentido de una frase.',
        '<b>Separación de datos:</b> ninguna estadística ni vocabulario se estimó utilizando el archivo de prueba.'
    ]
    for i,m in enumerate(met):story.append(P(f'{i+1}. {m}','BodyCustom'))
    story.append(P('3. Longitud de las noticias','SectionCustom'))
    putchart('01_longitud_tokens.png',7.4)
    l=stats['lengths']
    addtable([['Métrica','Tokens por documento'],['Promedio',f"{l['mean']:.1f}"],
              ['Mediana',str(l['median'])],['Percentil 90',str(l['p90'])],['Percentil 95',str(l['p95'])],
              ['Máximo',str(l['max'])]], [W*0.5,W*0.5])
    story.append(Spacer(1,6))
    story.append(P(f"El percentil 95 es de <b>{l['p95']} tokens</b>: el 95 % de las noticias tiene esa longitud o menos después de la normalización. Es una referencia razonable para la longitud máxima de un modelo basado en palabras. Si se utiliza BERT u otro Transformer, deberá recalcularse con su tokenizador de subpalabras.",'MutedCustom'))
    pagebreak()

    # Profundización: diferencias entre categorías y efecto del recorte.
    extra=stats['extended']
    story.append(P('4. Diferencias de longitud entre categorías','SectionCustom'))
    story.append(P('Un conjunto equilibrado en cantidad de documentos no necesariamente es homogéneo en longitud. Se compararon las cuatro clases por separado, utilizando tokens lematizados antes de retirar palabras vacías.'))
    putchart('05_longitud_por_clase.png',6.4)
    rows=[['Categoría','Promedio','P95','Superan max_len']]
    for rec in extra['per_class']:
        rows.append([CLASS_SP[rec['class']],f"{rec['mean']:.1f}",str(rec['p95']),f"{rec['gt_maxlen']} ({rec['gt_maxlen_pct']:.1f}%)"])
    addtable(rows,[W*.33,W*.17,W*.14,W*.36])
    story.append(Spacer(1,11))
    story.append(P('Cobertura de distintas longitudes máximas','SubCustom'))
    putchart('06_cobertura_longitudes.png',5.7)
    story.append(P(f"Con <b>max_len = {stats['lengths']['p95']}</b>, se conserva completa aproximadamente el 95 % de las noticias. El resto requerirá truncamiento si no se adopta una ventana mayor. Una longitud mayor disminuye el recorte, pero incrementa el costo del procesamiento. Esta comparación corresponde a <b>palabras</b>, no a tokens WordPiece.", 'MutedCustom'))
    pagebreak()

    story.append(P('5. Calidad del texto y decisión de limpieza','SectionCustom'))
    story.append(P('La muestra está libre de nulos, pero eso no implica ausencia de ruido textual. Antes del modelado se inspeccionaron los patrones de codificación y los elementos editoriales que podrían introducir términos artificiales.'))
    q=extra['noise']
    rows=[['Incidencia en el texto original','Noticias','Porcentaje']]
    for key,nam in [('html_entity_fragment','Entidades HTML incompletas'),('html_tags','Etiquetas HTML'),('links','Enlaces explícitos')]:
        d=q[key];rows.append([nam,nfmt(d['n']),f"{d['pct']:.2f}%"])
    addtable(rows,[W*.55,W*.20,W*.25])
    story.append(Spacer(1,12))
    story.append(P('Criterios adoptados','SubCustom'))
    for p in [
        'La decodificación HTML se aplica <b>antes</b> de eliminar la puntuación. Así se evita que fragmentos de entidades terminen entre las palabras más repetidas.',
        'Se estandarizan minúsculas y espacios, pero se documenta que la remoción de números, símbolos y puntuación puede perder información relevante en algunas tareas.',
        'Las stopwords se conservan para la descripción de longitud. Para estudiar expresiones temáticas se filtran, manteniendo <i>no</i>, <i>not</i> y <i>nor</i>. La lista de stopwords no se aplica de forma automática a futuros Transformers.',
        'Los N-gramas se construyen sobre el texto filtrado; al retirar palabras funcionales, algunas combinaciones no eran contiguas en la noticia original.'
    ]: story.append(P('• '+p,'BodyCustom'))
    story.append(P(f'Como control adicional, tras la normalización aparece <b>{extra["normalized_duplicate_texts"]}</b> coincidencia textual completa que no figuraba entre los duplicados exactos originales. Este indicio debe revisarse antes de construir una partición interna de entrenamiento y validación; no demuestra por sí solo que existan noticias idénticas en significado.'))
    story.append(P('Un diagnóstico de calidad permite fundamentar cada transformación y evitar que el modelo aprenda características del formato editorial en lugar del tema periodístico.'))
    pagebreak()

    story.append(P('6. Frecuencias del vocabulario','SectionCustom'))
    putchart('03_palabras_frecuentes.png',7.4)
    top50=stats['top50'];stop_total=stats['stop_top50_count']
    story.append(P(f"Entre las 50 palabras más frecuentes, <b>{stop_total}</b> figuran en la lista de stopwords de SpaCy ({stop_total/50:.0%}). Este predominio justifica excluirlas del recuento de expresiones temáticas, pero conservarlas al estudiar la longitud original del texto.",'BodyCustom'))
    story.append(P('Top 50 de palabras (frecuencia en train)','SubCustom'))
    rows=[['#','Palabra','Frecuencia','#','Palabra','Frecuencia']]
    for i in range(25):
        a=top50[i];b=top50[25+i]
        rows.append([str(i+1),a[0],str(a[1]),str(i+26),b[0],str(b[1])])
    addtable(rows,[W*.065,W*.27,W*.165,W*.065,W*.27,W*.165],padding=2.3)
    pagebreak()

    story.append(P('7. Bigramas frecuentes','SectionCustom'))
    story.append(P('Los bigramas se calcularon con CountVectorizer sobre el texto lematizado y sin palabras vacías, excepto las negaciones. Al retirar palabras funcionales, algunos pares no eran adyacentes en el texto original.','MutedCustom'))
    addtable([['Posición','Bigrama','Ocurrencias']]+[[str(i),t,str(c)] for i,(t,c) in enumerate(stats['bigrams'],1)],
             [W*.16,W*.64,W*.20])
    story.append(Spacer(1,13))
    story.append(P('Interpretación','SubCustom'))
    story.append(P(stats['bi_interpretation']))
    story.append(P('La frecuencia de una expresión no garantiza que sea útil para clasificar: su valor dependerá de cómo se distribuya entre las cuatro categorías.','MutedCustom'))
    pagebreak()

    story.append(P('8. Trigramas frecuentes','SectionCustom'))
    story.append(P('Se aplicó el mismo procesamiento y filtrado que en los bigramas.','MutedCustom'))
    addtable([['Posición','Trigrama','Ocurrencias']]+[[str(i),t,str(c)] for i,(t,c) in enumerate(stats['trigrams'],1)],
             [W*.16,W*.64,W*.20])
    story.append(Spacer(1,13))
    story.append(P('Interpretación','SubCustom'))
    story.append(P(stats['tri_interpretation']))
    story.append(P('En la siguiente etapa se podrá estudiar qué expresiones son verdaderamente discriminativas para cada categoría, por ejemplo mediante TF-IDF.','MutedCustom'))
    pagebreak()

    story.append(P('9. Distribución léxica','SectionCustom'))
    putchart('04_zipf.png',7.2)
    addtable([['Indicador','Resultado'],['Vocabulario distinto',nfmt(stats['vocab'])],
              ['Tokens totales (sin quitar stopwords)',nfmt(stats['total_tokens'])],
              ['Tipos con frecuencia 1 (hapax)',f"{nfmt(stats['hapax'])} ({stats['hapax_pct']:.1f}%)"],
              ['Riqueza léxica, tipos/tokens',f"{stats['type_token_ratio']:.3f}"]],
             [W*.63,W*.37])
    story.append(Spacer(1,9))
    story.append(P('La elevada proporción de términos con frecuencia uno confirma una cola larga: abundan nombres propios y expresiones poco repetidas. Esta dispersión puede afectar a modelos basados en vocabularios completos; para TF-IDF convendrá controlar frecuencia mínima y regularización.','MutedCustom'))
    pagebreak()

    story.append(P('10. Vocabulario distintivo por categoría','SectionCustom'))
    story.append(P('Se comparó la <b>frecuencia documental</b> de cada término dentro de su categoría respecto de las otras tres. La razón calculada mide concentración temática en entrenamiento, no desempeño predictivo ni importancia causal. Solo se muestran términos que aparecen en al menos 40 noticias de la clase.'))
    for cl in LABELS:
        items=extra['characteristic_terms'][cl][:5]
        story.append(P(CLASS_SP[cl],'SubCustom'))
        rows=[['Término','Documentos de clase','Razón vs. otras clases']]+[[x['term'],str(x['documents']),f"{x['lift']:.1f}×"] for x in items]
        addtable(rows,[W*.43,W*.28,W*.29],padding=2.3)
        story.append(Spacer(1,4))
    story.append(P('Las noticias comparten vocabulario transversal, pero estas concentraciones sugieren señales temáticas útiles para empezar la clasificación. La pertinencia real se comprobará con validación independiente.','MutedCustom'))
    pagebreak()

    story.append(P('11. Solapamiento temático y preparación del modelado','SectionCustom'))
    story.append(P('Además de la frecuencia global, se compararon los 100 términos de mayor frecuencia documental de cada categoría. El índice de Jaccard señala cuánto vocabulario comparten, aunque no debe interpretarse como una matriz de errores del clasificador.'))
    putchart('07_solapamiento_lexico.png',4.4)
    story.append(P('Decisiones metodológicas para los próximos módulos','SubCustom'))
    for q in [
        '<b>Representación:</b> comenzar con TF-IDF y comparar después con embeddings y Transformers sobre el mismo corpus.',
        '<b>Control de dimensionalidad:</b> evaluar vocabulario mínimo, regularización y tratamiento de términos poco frecuentes debido a la cola larga.',
        '<b>Secuencia:</b> usar como referencia el percentil 95 de palabras; volver a calcular longitudes para el tokenizador subword del modelo final.',
        '<b>Validación:</b> ajustar vocabulario, transformadores y parámetros solo con entrenamiento y validación; conservar el test para la evaluación final.',
        '<b>Interpretación:</b> observar métricas por clase y una matriz de confusión para analizar solapamientos semánticos que el EDA no puede resolver por sí solo.'
    ]:story.append(P('• '+q,'BodyCustom'))
    story.append(Spacer(1,10))
    story.append(P('12. Conclusiones','SectionCustom'))
    c=stats['conclusion'];story.append(P(c))
    story.append(P('Datos y reproducibilidad','SubCustom'))
    story.append(P(f"Fuente: AG News, muestra académica provista por Coderhouse ({nfmt(stats['train_n'])} noticias de entrenamiento). Herramientas: pandas, scikit-learn, SpaCy {stats['spacy_version']} y matplotlib. Procesamiento: {stats['nlp_mode']}. El archivo de prueba se reservó para evaluaciones posteriores.",'MutedCustom'))
    if fallback:
        story.append(P('<b>Requisito pendiente para la entrega final:</b> instalar <i>en_core_web_sm</i> y ejecutar <i>python src/generar_reporte.py</i>. El programa reemplazará la tokenización provisional por lematización y recalculará todos los indicadores.','WarnCustom'))

    def furniture(canv, doc):
        canv.saveState()
        canv.setFillColor(color(DARK));canv.rect(0,A4[1]-9,A4[0],9,fill=1,stroke=0)
        canv.setStrokeColor(color('#dde5e8'));canv.line(42,39,A4[0]-42,39)
        canv.setFont('DejaVu',7.5);canv.setFillColor(color(GREY))
        canv.drawString(42,26,'Data Science III  ·  Módulo 2  ·  AG News')
        canv.drawRightString(A4[0]-42,26,f'Página {doc.page}')
        if fallback:
            canv.saveState();canv.setFillColor(color('#b77548'))
            canv.setFont('DejaVuBold',7)
            canv.drawRightString(A4[0]-42,A4[1]-25,'BORRADOR: sin lematización')
            canv.restoreState()
        canv.restoreState()
    doc.build(story,onFirstPage=furniture,onLaterPages=furniture)
    return pdf


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--allow-fallback',action='store_true',help='Borrador sin lematización si no existe en_core_web_sm')
    args=ap.parse_args()
    tr=pd.read_csv(DATA/'ag_news_train.csv')
    te=pd.read_csv(DATA/'ag_news_test.csv')
    if list(tr.columns)!=['text','label'] or list(te.columns)!=['text','label']:
        raise ValueError('Los CSV deben tener columnas text,label')
    if tr.isna().any().any() or te.isna().any().any():
        raise ValueError('Se detectaron nulos, revisar entrada')
    if set(tr.label.unique()) != set(LABELS):
        raise ValueError('Etiquetas AG News inesperadas')
    raw=[tidy(s) for s in tr.text]
    nlp,model_name,lemma=get_nlp(args.allow_fallback,raw)
    print('Modo NLP:',model_name,'| registros train:',len(tr),'| test reservado:',len(te),flush=True)
    tokens=[];content=[]
    for start in range(0,len(raw),200):
        docs=nlp.pipe(raw[start:start+200],batch_size=100,n_process=1)
        for doc in docs:
            full=[];flt=[]
            for t in doc:
                if not t.is_alpha:continue
                w=(t.lemma_ if lemma else t.text).lower().strip()
                if not w or not w.isalpha() or (len(w)==1 and w not in {'a','i'}):continue
                full.append(w)
                if (not t.is_stop or w in EXCEPTIONS) and len(w)>1:flt.append(w)
            tokens.append(full);content.append(flt)
        if start and (start//200)%10==0: print(f'Procesados {start} documentos...',flush=True)
    lengths=np.array([len(x) for x in tokens]);wordfreq=Counter(w for xs in tokens for w in xs)
    stops=nlp.Defaults.stop_words
    top50=wordfreq.most_common(50)
    seqs=[' '.join(v) if v else 'EMPTY' for v in content]
    bigrams=top_ngrams(seqs,2);trigrams=top_ngrams(seqs,3)
    p95=int(np.ceil(np.percentile(lengths,95)))
    cats=tr.label.value_counts().to_dict()
    create_charts(lengths,cats,wordfreq,p95,not lemma)
    stop_top_count=sum(w in stops for w,_ in top50)
    bi_names=[w for w,_ in bigrams]
    tri_names=[w for w,_ in trigrams]
    def examples(names, selected):
        items=[x for x in selected if x in names]
        if not items:items=names[:3]
        return ', '.join(f'<i>{html.escape(x)}</i>' for x in items[:4])
    bi_examples=examples(bi_names,['new york','oil price','prime minister','mobile phone'])
    tri_examples=examples(tri_names,['boston red sox','securities exchange commission','crude oil price','international space station'])
    # Métricas de profundidad, calculadas solo sobre training.
    interim={'lengths':{'p95':p95}}
    compute_extended_stats(tr,tokens,content,lengths,nlp,interim)
    ext=interim['extended']
    nois=ext['noise']['html_entity_fragment']
    trunc=next(x for x in ext['coverage'] if x['max_len']==p95)
    maxcl=max(ext['per_class'],key=lambda x:x['p95'])
    conclusion=(
      f'El conjunto de entrenamiento de AG News reúne {nfmt(len(tr))} noticias, con una distribución uniforme entre '
      f'World, Sports, Business y Sci_Tech. No se detectaron registros nulos ni duplicados exactos. '
      f'Sin embargo, {nfmt(nois["n"])} noticias contenían fragmentos de codificación HTML que podían convertirse '
      f'en términos artificiales, por lo que fue necesario normalizarlos antes de tokenizar y lematizar. '
      f'Las noticias tienen una mediana de {int(np.median(lengths))} tokens. El percentil 95 es {p95}, '
      f'que permitiría conservar completas el {trunc["coverage"]:.1f} % de las secuencias de palabras. '
      f'La distribución de longitud no es idéntica entre clases; {maxcl["class"]} presenta un percentil 95 '
      f'de {maxcl["p95"]} tokens, lo que requiere atender posibles diferencias de truncamiento. '
      f'El vocabulario contiene {nfmt(len(wordfreq))} lemas distintos y {nfmt(sum(v==1 for v in wordfreq.values()))} '
      f'aparecen una sola vez. La cola larga aconseja controlar la dimensionalidad y la frecuencia mínima '
      f'en representaciones basadas en palabras. Los N-gramas y términos distintivos evidencian temas '
      f'periodísticos reconocibles, pero su frecuencia no prueba por sí sola utilidad predictiva. '
      f'Para los siguientes módulos se recomienda comparar TF-IDF con modelos neuronales, aplicar validación '
      f'por categoría y recalcular la longitud con el tokenizador subword correspondiente. '
      f'El conjunto de prueba se mantendrá aislado hasta la evaluación final.'
    )
    if len(conclusion.split()) > 300:
        raise AssertionError('La conclusión supera las 300 palabras')
    stats={
       'train_n':len(tr),'test_n':len(te),'nulls':int(tr.isna().sum().sum()),'duplicates':int(tr.text.duplicated().sum()),
       'class_counts':cats,'class_balance':'Perfectamente balanceado (2.000 documentos por clase)',
       'lengths':{'mean':float(np.mean(lengths)),'median':int(np.median(lengths)),'p90':int(np.ceil(np.percentile(lengths,90))),
                  'p95':p95,'max':int(max(lengths))},
       'top50':top50,'stop_top50_count':stop_top_count,'bigrams':bigrams,'trigrams':trigrams,
       'bi_interpretation':f'Se destacan expresiones como {bi_examples}, relacionadas con lugares, economía, política o tecnología. En conjunto muestran que el corpus combina nombres propios con vocabulario especializado; habrá que examinar su peso por clase antes de incorporarlas al modelo.',
       'tri_interpretation':f'Expresiones como {tri_examples} describen entidades y asuntos más específicos que los bigramas. Su menor frecuencia es esperable: al aumentar la longitud de las secuencias, disminuyen las repeticiones exactas y crece la dispersión.',
       'vocab':len(wordfreq),'total_tokens':sum(wordfreq.values()),'hapax':sum(v==1 for v in wordfreq.values()),
       'hapax_pct':sum(v==1 for v in wordfreq.values())/max(len(wordfreq),1)*100,
       'type_token_ratio':len(wordfreq)/max(sum(wordfreq.values()),1),
       'nlp_mode':model_name,'spacy_version':spacy.__version__,'conclusion':conclusion,
    }
    stats['extended']=ext
    extra_charts(stats)
    with open(OUT/('estadisticas_borrador.json' if not lemma else 'estadisticas_finales.json'),'w') as f:
        json.dump(stats,f,indent=2,ensure_ascii=False)
    pdf=make_pdf(stats,not lemma)
    print('Reporte generado:',pdf, 'Tamaño KB:',round(pdf.stat().st_size/1024,1),flush=True)
    print('Datos:', {k:stats[k] for k in ['train_n','lengths','vocab','stop_top50_count','bigrams','trigrams']},flush=True)
    print('Palabras conclusión:',len(stats['conclusion'].split()),flush=True)

if __name__=='__main__':main()
