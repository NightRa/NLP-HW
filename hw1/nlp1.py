import requests
from lxml import html
from lxml import etree
from io import StringIO
import re

def show(xml):
    return html.tostring(xml, pretty_print=True, encoding='UTF-8').decode("UTF-8")

def clean(p):
    # Eliminate all newlines & tabs, as they are insignificant in html.
    noNewlines = re.sub(r"\n|\xa0|\t", " ", p)
    # Eliminate repeating spaces, as they are insignificant in html.
    noDuplicateSpaces = re.sub(" +", " ", noNewlines)
    return noDuplicateSpaces.strip()


req = requests.get('http://www.ynet.co.il/articles/0,7340,L-4684564,00.html')
htmlStr = req.text

htmlTree = etree.parse(StringIO(htmlStr), etree.HTMLParser())  # type: etree.ElementTree
htmlStr2 = etree.tostring(htmlTree, pretty_print=True, encoding='UTF-8').decode("UTF-8")

htmlTree = html.document_fromstring(htmlStr2)  # type: etree.ElementTree

titleXML = htmlTree.find_class('art_header_title')[0]  # type: [etree.ElementTree]
title = titleXML.text

subtitleXML = htmlTree.find_class('art_header_sub_title')[0]
subtitle = subtitleXML.text

# go to span - the parent of all the paragraphs, and get the paragraphs.
# the div contained p's that aren't sentences.
paragraphsRaw = htmlTree.find_class('art_body')[0].xpath("//span/p")
cleanedParagraphs = list(map((lambda x: clean(x.text_content())), paragraphsRaw))
paragraphs = list(filter((lambda x: len(x) > 0), cleanedParagraphs))

for p in paragraphs:
    print(p)

