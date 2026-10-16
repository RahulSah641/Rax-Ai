#!/usr/bin/env python3
"""Regenerate the presentation. Requires reportlab and the bundled documentation JSON files.

Run from the project root: python docs/build_presentation.py
This script creates only the PDF; it does not run or modify the application.
"""
from pathlib import Path
from html import escape
import json
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
W,H=960,540
NAVY=colors.HexColor('#11263D'); TEAL=colors.HexColor('#008D8A'); INK=colors.HexColor('#263C50')
MUTED=colors.HexColor('#526A7D'); BG=colors.HexColor('#F2F6F8'); LINE=colors.HexColor('#D7E2E9')
ORANGE=colors.HexColor('#E9A849'); WHITE=colors.white
FONT='Helvetica'; BOLD='Helvetica-Bold'
fontdir=Path('/usr/share/fonts/truetype/dejavu')
if (fontdir/'DejaVuSans.ttf').exists():
    pdfmetrics.registerFont(TTFont('DocSans',str(fontdir/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DocSansBold',str(fontdir/'DejaVuSans-Bold.ttf')))
    FONT='DocSans';BOLD='DocSansBold'

out=HERE/'Project_Presentation.pdf'
c=canvas.Canvas(str(out),pagesize=(W,H),pageCompression=1)
c.setTitle('Rax Ai - Python Desktop Edition | Project Presentation')
c.setAuthor('Rax Ai - Project Documentation')
c.setSubject('Purpose, Python stack, setup, architecture, use, validation and future scope')
page=0

def text(s,x,y,w,size=15,color=INK,bold=False,leading=None):
    style=ParagraphStyle('s',fontName=BOLD if bold else FONT,fontSize=size,leading=leading or size*1.42,textColor=color)
    p=Paragraph(s,style)
    _,height=p.wrap(w,1000)
    if y-height<47:
        raise ValueError(f'Page {page}: text reaches footer: {s[:75]!r} at y={y}, h={height}')
    p.drawOn(c,x,y-height)
    return y-height

def header(section,title,subtitle=''):
    global page
    page+=1
    c.setFillColor(BG);c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(TEAL);c.rect(0,H-8,W,8,fill=1,stroke=0)
    text(section.upper(),38,510,884,10,TEAL,True)
    text(title,38,486,884,27,NAVY,True,31)
    if subtitle: text(subtitle,38,443,884,12,MUTED)
    c.setStrokeColor(LINE);c.line(38,38,922,38)
    c.setFillColor(MUTED);c.setFont(FONT,8)
    c.drawString(38,23,'Rax Ai  /  PYTHON DESKTOP EDITION')
    c.drawRightString(922,23,f'{page:02d} / 24')

def finish(source=''):
    if source:
        c.setFillColor(MUTED);c.setFont(FONT,7.5);c.drawString(38,44,source)
    c.showPage()

def card(x,y,w,h,title,body,accent=TEAL):
    c.setFillColor(WHITE);c.roundRect(x,y-h,w,h,9,fill=1,stroke=0)
    c.setFillColor(accent);c.roundRect(x,y-h,4,h,2,fill=1,stroke=0)
    text(title,x+19,y-16,w-37,17,NAVY,True)
    yy=text(body,x+19,y-51,w-37,13.5)
    if yy<y-h+10:raise ValueError(f'Card overflow page {page} {title}')

def bullets(items,x=48,y=399,w=855,size=17,gap=17):
    for item in items:
        c.setFillColor(TEAL);c.circle(x,y-9,3,fill=1,stroke=0)
        y=text(item,x+18,y,w-20,size)-gap
    return y

def table(headers,rows,widths,y=405,size=13):
    st=ParagraphStyle('cell',fontName=FONT,fontSize=size,leading=size*1.3,textColor=INK)
    hb=ParagraphStyle('head',fontName=BOLD,fontSize=size,leading=size*1.3,textColor=WHITE)
    data=[[Paragraph(escape(str(v)),hb) for v in headers]]+[[Paragraph(escape(str(v)),st) for v in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,colors.HexColor('#EAF1F5')]),('LINEBELOW',(0,0),(-1,-1),0.35,LINE)]))
    _,height=t.wrap(W,1000)
    if y-height<60:raise ValueError(f'Table overflow page {page}, height={height}')
    t.drawOn(c,38,y-height)
    return y-height

def code(lines,x=38,y=399,w=884,h=116,size=15):
    c.setFillColor(NAVY);c.roundRect(x,y-h,w,h,8,fill=1,stroke=0)
    for n,line in enumerate(lines):
        c.setFillColor(WHITE);c.setFont('Courier',size);c.drawString(x+20,y-28-n*(size+10),line)


# 01
page+=1
c.setFillColor(NAVY);c.rect(0,0,W,H,fill=1,stroke=0)
c.setFillColor(TEAL);c.rect(0,0,18,H,fill=1,stroke=0)
text('A COMPLETE PYTHON PROJECT',56,482,650,12,ORANGE,True)
text('Rax Ai',56,425,700,52,WHITE,True,60)
text('Python Desktop Edition',59,352,760,29,WHITE,False)
text('Vector search. Your documents. Optional local AI.',59,292,735,20,colors.HexColor('#BFDBE3'))
text('Professional project presentation and practical run guide',59,229,760,15,WHITE)
for x,v,label in [(59,'3','SEARCH ALGORITHMS'),(340,'2','DOCUMENT MODES'),(621,'0','CORE PIP PACKAGES')]:
    text(v,x,152,220,34,WHITE,True);text(label,x,103,240,10,colors.HexColor('#BFDBE3'),True)
text('Source code, setup, architecture, usage, validation and future scope',59,68,840,10,colors.HexColor('#BFDBE3'))
c.showPage()
# 02
header('Purpose','Find useful information. Understand the mechanism.','A practical document assistant and an educational vector-search laboratory.')
card(38,401,280,225,'The problem','Useful knowledge is spread across notes and files. A complex toolchain can make a first AI project difficult to run.')
card(340,401,280,225,'The approach','One desktop app combines transparent algorithms, persistent documents and an offline retrieval path.')
card(642,401,280,225,'The outcome','Run a useful demonstration first. Add local model inference when your computer and project need it.')
text('Designed for learning, personal knowledge exploration and controlled prototypes.',46,128,850,18,NAVY,True)
finish()
# 03
header('Project evolution','The original idea, rebuilt for Python','Based on the supplied Your-OWN-AI-main.zip project.')
table(['Area','Supplied project','Python edition'],[
['Application','C++ engine + browser interface','Python desktop application'],
['Setup','C++ compiler and local AI setup','Python + Tk; optional models later'],
['Vector search','HNSW, KD-Tree, brute force','Readable Python implementations of all three'],
['Documents','Ollama embedding / RAG workflow','Offline extracts + optional Ollama RAG'],
['Persistence / delivery','Original source archive','SQLite, backups, launchers, tests and documentation']],[160,335,389],size=13)
finish('Project-specific comparison: supplied source archive and delivered Python source.')
# 04
header('Delivered capabilities','A complete, understandable workflow','Two kinds of document answers are clearly labelled in the interface.')
card(38,402,431,145,'Explore vectors','Search synthetic 16D vectors; select a metric; inspect PCA; compare algorithms with measured timings.')
card(491,402,431,145,'Manage documents','Add or import notes, preserve chunks in SQLite, delete documents and export or merge JSON backups.')
card(38,237,431,145,'Find source evidence','Offline TF-IDF returns labelled extracts. Find passages and export an answer for later review.')
card(491,237,431,145,'Add local generation','With Ollama installed, create embeddings and generate answers using retrieved context. Models are separate downloads.')
finish()
# 05
header('Language and libraries','Python throughout the application','No C++ compilation. No hosted AI account needed for core functionality.')
table(['Technology','Use','Required?'],[
['Python 3.10+','Application source; 3.12 recommended','Yes'],
['tkinter / ttk','Desktop pages, tables, dialogs and canvas','GUI; Tk support must be installed'],
['sqlite3 + standard modules','Persistence, algorithms, HTTP, files, tests','Included with normal Python setup'],
['pypdf','Selectable-text PDF import','Optional extra'],
['Ollama + two models','Neural embeddings and generated answers','Optional separate installation'],
['ReportLab','Regenerate this presentation PDF','Documentation only']],[215,384,285],size=12.6)
finish('References: Python Tkinter / sqlite3 docs; pypdf; Ollama API and model catalog. Full links on page 24.')
# 06
header('Architecture','One application, three processing paths','The desktop and local API share core services; durable content lives in SQLite.')
# diagram in vector geometry
for x,y,w,h,title,body in [(38,405,262,75,'Desktop UI','Home / documents / questions / vectors'),(330,405,260,75,'Application services','Validation, storage and retrieval'),(660,405,262,75,'SQLite','Documents, chunks, embeddings'),(38,279,262,112,'Synthetic demo','16D vectors -> selected index\n-> ranked neighbors + PCA'),(330,279,262,112,'Offline documents','TF-IDF -> relevant passages\n-> exact source extracts'),(660,279,262,112,'Optional Ollama','Embeddings -> HNSW context\n-> model-generated answer')]:
    c.setFillColor(WHITE);c.roundRect(x,y-h,w,h,8,fill=1,stroke=0)
    text(title,x+14,y-12,w-28,16,NAVY,True)
    text(body.replace('\n','<br/>'),x+14,y-40,w-28,12.1)
c.setStrokeColor(TEAL);c.setLineWidth(2)
for a,b in [(300,330),(590,660)]:c.line(a,366,b,366)
c.line(460,330,460,302);c.line(169,302,791,302)
for x in (169,460,791):c.line(x,302,x,279)
text('The REST API is optional and loopback-only. Its document routes use offline retrieval.',45,113,870,14,MUTED)
finish()
# 07
header('Search algorithms','Three implementations, different trade-offs','The exact baseline makes approximate results assessable.')
table(['Algorithm','How it searches','What to explain'],[
['Brute force','Compute every vector distance and rank results','Exact. O(N x d) distance work, plus result ranking.'],
['KD-Tree','Split by coordinates and prune safe branches','Exact. High dimensions can weaken pruning; no universal logarithmic bound.'],
['HNSW','Navigate a layered neighbor graph','Approximate. Recall depends on data, connectivity and search effort.']],[165,335,384],size=14)
text('The delivered indexes are educational Python implementations. Small datasets may favor a direct scan.',45,119,860,15,NAVY,True)
finish('Concept references: Malkov & Yashunin HNSW paper; SciPy KDTree documentation (not a project dependency).')
# 08
header('Vectors and visualization','Search in full dimensions. Plot a 2D projection.','The demo uses synthetic teaching vectors, not a neural text model.')
card(38,402,280,217,'Cosine','Compares direction.<br/><br/>distance = 1 - cosine similarity<br/><br/>Lower distance is better. Zero vectors require explicit handling.')
card(340,402,280,217,'Euclidean','Straight-line distance.<br/><br/>sqrt(sum of squared coordinate differences)<br/><br/>Feature scale matters.')
card(642,402,280,217,'Manhattan','Coordinate-wise distance.<br/><br/>sum of absolute coordinate differences<br/><br/>A different geometry can change ranking.')
text('PCA keeps two directions of variation. It loses information, so screen distance is not the search result.',44,140,875,17,NAVY,True)
finish()
# 09
header('Interface guide','Five pages, clear actions','Navigation schematic based on the implemented UI; this is not an application screenshot.')
table(['Page','Primary actions','Use it to...'],[
['Home','Open the workspace; load sample documents','Start a demonstration and see status'],
['Documents','Add / Import file / Export backup / Import backup','Manage your saved source material'],
['Ask documents','Ask question / Find passages / Export answer','Find and inspect source evidence'],
['Vector lab','Search vectors / Compare algorithms / Manage vectors','Explore algorithms and distances'],
['Settings','Check connection / Save settings','Configure optional local models']],[165,400,319],size=13)
text('The GUI needs a graphical desktop. Visual launch validation was unavailable in the build environment.',43,111,875,13.5,MUTED)
finish('Use the included desktop checklist on your computer. Core and API validation are reported separately.')
# 10
header('Executed offline example','An answer you can trace to your notes','Actual core execution against bundled sample documents; no model inference was used.')
ex=json.loads((HERE/'offline_example.json').read_text())
card(38,401,431,107,'Question',escape(ex['question']))
card(38,274,431,182,'What the result means','These are existing source sentences. They may include limitations as well as a definition.<br/><br/>Open the complete retrieved passages to understand the surrounding context.')
answer=ex['answer'].replace('—','-')
c.setFillColor(WHITE);c.roundRect(491,92,431,309,9,fill=1,stroke=0)
text('Returned source extracts',508,383,397,17,NAVY,True)
text(escape(answer).replace('\n','<br/>'),508,344,397,13)
finish('The example JSON is included as docs/offline_example.json. Source labels refer to the provided documents.')
# 11
header('Document retrieval','From your files to source-labelled extracts','Paste text, import UTF-8 TXT/Markdown, or optionally import a text-based PDF.')
card(38,402,280,213,'1. Add content','Documents > Add document or Import file.<br/><br/>Chunks contain up to 180 words with 35-word overlap.')
card(340,402,280,213,'2. Rank passages','TF-IDF weights terms and scores lexical overlap with the question.<br/><br/>This mode needs no downloaded AI model.')
card(642,402,280,213,'3. Review evidence','The output contains relevant text from your notes and source labels.<br/><br/>Unrelated questions may have no useful match.')
text('Limits: 10 MB per input file; 200,000 text characters per document. Scanned PDFs require external OCR.',44,141,870,15,NAVY,True)
finish('Concept reference: Stanford Introduction to Information Retrieval. PDF limits: pypdf extraction documentation.')
# 12
header('Optional local AI','Add embeddings and generated answers with Ollama','Both models are separate downloads. Core offline use remains available.')
code(['ollama pull nomic-embed-text','ollama pull llama3.2:1b','ollama list'],y=405,h=118,size=15)
card(38,266,430,175,'Connect','Start Ollama if needed. Open Settings, keep the local base URL, use Check connection, then Save settings.')
card(491,266,431,175,'Ask in AI mode','Choose Ollama (local AI). The app embeds documents, retrieves context and requests a generated answer. Review the sources.')
finish('Official sources: docs.ollama.com/api/embed, /api/generate, /api/tags and model catalog links on page 24.')
# 13
header('Installation','Run the project on your computer','Recommended starting point: Python 3.12 with Tcl/Tk support; minimum Python 3.10.')
card(38,402,431,238,'Windows','1. Install Python and include Tcl/Tk.<br/>2. Extract the entire ZIP.<br/>3. Double-click START_WINDOWS.bat.<br/><br/>Terminal alternative:<br/><b>py -3 app.py</b><br/><br/>Use python app.py if python is your installed command.')
card(491,402,431,238,'macOS / Linux','1. Install Python with matching Tk support.<br/>2. Extract the entire ZIP.<br/>3. Run python3 app.py.<br/><br/>Launcher alternative:<br/><b>bash start.sh</b><br/><br/>On Debian/Ubuntu, missing Tk is commonly resolved with python3-tk.')
text('Verify Tk first: python -m tkinter  |  A GUI session is required for the desktop window.',46,121,870,15,NAVY,True)
finish('Python reference: docs.python.org/3/library/tkinter.html. Interpreter command names vary by platform.')
# 14
header('Run modes','Start small, then add only what you need','Commands run from the extracted project folder.')
table(['Goal','Command'],[
['Desktop application','python app.py'],
['Offline diagnostic','python app.py --self-check'],
['Automated tests','python -m unittest discover -s tests -v'],
['A separate data folder','python app.py --data-dir ./my-data'],
['Optional PDF import','python -m pip install -r requirements-pdf.txt'],
['Optional local REST API','python app.py --api --port 8765']],[268,616],size=14)
finish('Replace python with python3 or py -3 when that is the command for your installed interpreter.')
# 15
header('Storage and source code','Persistent documents, readable modules','Default data folder: .Rax_ai under your home directory.')
card(38,402,431,158,'Durable data','knowledge.sqlite3 stores documents, chunks and model-specific cached embeddings.<br/><br/>settings.json stores preferences.')
card(491,402,431,158,'Backup behavior','Export backup writes document JSON. Import backup merges duplicates and rebuilds chunks.<br/><br/>Model files and embedding caches are not exported.')
table(['Module','Responsibility'],[['gui.py / app.py','Interface and launch modes'],['vectors.py / knowledge.py','Search algorithms; document persistence and retrieval'],['ollama_client.py / api.py','Optional model requests; local JSON endpoints']],[300,584],y=220,size=12.3)
finish('Vector lab edits are session-only. Database relationships and complete module map are in PROJECT_REPORT.md.')
# 16
header('Developer interface','A loopback API for scripts and demonstrations','Desktop use does not require this server.')
table(['Endpoint group','Purpose'],[
['GET /health','Health check without a token'],
['GET /api/stats; GET /api/documents','Statistics and document listing'],
['POST /api/documents; DELETE /api/documents/<id>','Add/delete saved documents'],
['POST /api/search; POST /api/ask','Offline passage search and sourced extracts'],
['/api/vectors; /api/vector-search; /api/benchmark','Session vector CRUD, search and algorithm comparison']],[525,359],size=12.2)
text('The server prints a bearer token. Send Authorization: Bearer &lt;token&gt; on /api/ routes. GUI Ollama mode is separate.',43,114,875,14,NAVY,True)
finish('Exact methods, JSON fields and limits are documented in PROJECT_REPORT.md. The API is not a public hosting service.')
# 17
header('Benchmarking','Measure latency and recall together','No fixed speed claims: actual results depend on the dataset and computer.')
card(38,402,280,213,'Build time','Creating the index costs work before queries begin.<br/><br/>Keep build cost separate from query time when comparing methods.')
card(340,402,280,213,'Query time','A high-resolution timer measures repeated searches.<br/><br/>Tiny datasets can be dominated by Python overhead and background load.')
card(642,402,280,213,'Recall@k','Compare the returned IDs with the exact top-k reference.<br/><br/>4 of 5 recovered neighbors gives recall@5 = 0.8.')
text('Recall measures neighbor agreement. It does not measure whether an AI answer is correct.',44,141,874,17,NAVY,True)
finish()
# 18
header('Validation','What was checked before delivery','The evidence describes this release and its tested environment.')
validation=HERE/'validation_summary.json'
if not validation.exists():raise FileNotFoundError('Final validation_summary.json is required; do not publish provisional test claims.')
v=json.loads(validation.read_text())
bullets(v['bullets'],size=15.3,gap=15,y=398)
text(v['limitation'],46,115,870,13.5,MUTED)
finish('Detailed test commands and evidence are included in the package validation documentation.')
# 19
header('Boundaries and responsible use','Useful locally, with clear limits','This is an educational implementation and personal desktop prototype.')
card(38,402,431,146,'Retrieval limits','TF-IDF may miss synonyms. HNSW can miss neighbors. PDF text order can be imperfect. PCA is a lossy view.')
card(491,402,431,146,'Generation limits','Retrieved context can be incomplete. A language model can still add unsupported statements. Review sources.')
card(38,235,431,146,'Data handling','The database and exports are local and unencrypted. Use trusted documents and protect backups on your computer.')
card(491,235,431,146,'Deployment limits','A local API token is not a multi-user permissions system. Public deployment and production scale need further engineering.')
finish()
# 20
header('Future scope and impact','A practical foundation for the next version','The items below are proposals, not delivered capabilities.')
table(['Next improvement','Why it could matter','How to evaluate'],[
['Hybrid retrieval + reranking','Improve relevant passage ranking','Labelled query set and retrieval recall'],
['OCR / DOCX / page highlighting','Make more document formats useful','Import accuracy and source navigation'],
['Optimized ANN + batch embeddings','Handle larger collections','Memory, build time and query latency'],
['Multilingual evaluation','Support selected language needs','Native-language test questions'],
['Installer + shared-use safeguards','Reduce setup effort; manage access','Install success and permissions tests']],[285,319,280],size=12.4)
finish('Potential impact: faster note lookup and clearer AI education. No business productivity gain is claimed as measured.')
# 21
header('Presentation demo','A five-minute live walkthrough','Use the included notes; optional local models should be installed before presenting.')
table(['Time','Action','What to explain'],[
['0:00-0:40','Show Home','Purpose and offline-first setup'],
['0:40-1:30','Open Vector lab and search','Synthetic vectors; full-space distance; PCA'],
['1:30-2:10','Compare algorithms','Exact reference and approximate recall'],
['2:10-3:20','Ask documents: What is RAG?','Source extracts and persistent knowledge'],
['3:20-4:10','Show Settings / optional Ollama','Embeddings versus generation'],
['4:10-5:00','Close with testing and future scope','Honest limits and next experiment']],[135,344,405],size=12.5)
finish()
# 22
header('Questions and answers','Be ready to explain the design','Short answers for a project review or viva.')
card(38,402,431,145,'Is this a trained AI model?','No. The project implements search and retrieval. Optional pretrained models provide embeddings and generation.')
card(491,402,431,145,'Why Python and SQLite?','Python makes the logic readable. SQLite saves local documents without a database server. No C++ compilation is needed.')
card(38,236,431,145,'Why can brute force be faster?','On small datasets its simple scan can cost less than graph or tree overhead. Measure before choosing.')
card(491,236,431,145,'Do sources guarantee a correct answer?','No. Sources make evidence inspectable. The original text and any generated claim still need review.')
finish()
# 23
header('Key terms','The vocabulary behind the project','Keep representation, retrieval and generation separate.')
table(['Term','Meaning in this project'],[
['Embedding','A numeric vector produced by a model from text'],
['Chunk','A passage used as a retrieval unit'],
['TF-IDF','A term-weighting method for lexical relevance'],
['HNSW / KD-Tree','Graph-based approximate search / tree-based exact spatial search'],
['PCA','A 2D visualization made from higher-dimensional vectors'],
['RAG','Retrieve context, then ask a model to generate a response']],[245,639],size=14)
finish()
# 24
header('Primary references','Technical sources and project documentation','Official documentation and original research; full source list included in REFERENCES.md.')
refs=[
('Python desktop and storage','https://docs.python.org/3/library/tkinter.html','https://docs.python.org/3/library/sqlite3.html'),
('Ollama API and local models','https://docs.ollama.com/api/embed','https://docs.ollama.com/api/generate'),
('Model catalog','https://ollama.com/library/nomic-embed-text','https://ollama.com/library/llama3.2:1b'),
('HNSW research / KD-Tree reference','https://arxiv.org/abs/1603.09320','https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html'),
('TF-IDF / PDF extraction','https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html','https://pypdf.readthedocs.io/en/stable/user/extract-text.html')]
y=403
for title,a,b in refs:
    text(title,40,y,880,12.5,NAVY,True);y-=21
    for url in (a,b):
        text(f'<link href="{url}" color="#008D8A">{escape(url)}</link>',40,y,878,9.4);y-=16
    y-=10
text('Companion files: README.md for running the app; PROJECT_REPORT.md for full design and usage details.',40,81,880,11,NAVY,True)
finish('Sources checked 9 September 2026. External documentation does not certify this implementation.')
assert page==24,page
c.save()
print(out)
