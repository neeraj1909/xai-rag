# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown==3.10.2"]
# ///
"""Read-only structural checks for the interview publication."""
import json
from pathlib import Path
import re
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import markdown

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs/interview-preparation.md'
text = SOURCE.read_text()
parts = re.split(r'^## ', text, flags=re.M)
assert len(parts) == 4, 'Expected exactly three major sections'
assert len(re.findall(r'^### \d{2}\. ', parts[1], re.M)) == 22
assert len(re.findall(r'^### \d{2}\. ', parts[3], re.M)) == 22
questions = re.findall(r'^#### Q(\d{2})\.(\d)\s*—\s*(.+)$', text, re.M)
assert len(questions) == 132, len(questions)
for chapter in range(1, 23):
    ids = [int(q) for c, q, _ in questions if int(c) == chapter]
    assert ids == list(range(1, 7)), (chapter, ids)
answers = re.split(r'^#### Q\d{2}\.\d\s*—[^\n]*\n', parts[3], flags=re.M)[1:]
answer_words = []
for answer in answers:
    answer = re.split(r'^### |^<a id=', answer, flags=re.M)[0]
    count = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", answer))
    assert count >= 140, f'Insufficient descriptive answer: {count} words'
    answer_words.append(count)
spoken = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", parts[2]))
assert 3600 <= spoken <= 4300, spoken

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]; self.images=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag=='a' and 'href' in a: self.links.append(a['href'])
        if tag=='img': self.images.append(a['src'])

parsed=Links(); parsed.feed(markdown.markdown(text,extensions=['tables','fenced_code','toc','sane_lists']))
assert len(parsed.ids)==len(set(parsed.ids)), 'Duplicate anchors'
source_links=0
for link in parsed.links:
    if link.startswith('#'):
        assert unquote(link[1:]) in parsed.ids, link
    prefix='https://github.com/neeraj1909/xai-rag/blob/3fe12f78ff99bc212092099f7d46145f6d6a13f5/'
    if link.startswith(prefix):
        suffix=urlparse(link[len(prefix):])
        path=ROOT/unquote(suffix.path)
        assert path.is_file(), link
        if suffix.fragment:
            nums=re.findall(r'L(\d+)',suffix.fragment)
            assert all(0<int(n)<=len(path.read_text().splitlines()) for n in nums),link
        source_links+=1
for image in parsed.images:
    path=SOURCE.parent/image
    assert path.is_file(),image
    ET.parse(path)
assert len(parsed.images)==6
report={'major_sections':3,'components':22,'questions':len(questions),'answer_word_range':[min(answer_words),max(answer_words)],'spoken_words':spoken,'spoken_minutes_at_140_wpm':round(spoken/140,1),'spoken_minutes_at_120_wpm':round(spoken/120,1),'total_whitespace_words':len(text.split()),'source_links_checked':source_links,'static_figures':len(parsed.images),'internal_links':'pass'}
print(json.dumps(report,indent=2))
