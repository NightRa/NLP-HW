import requests
from lxml import html
from lxml import etree
from io import StringIO
import re

# original article: http://www.ynet.co.il/articles/0,7340,L-4684564,00.html
# Pi article: http://www.ynet.co.il/articles/0,7340,L-4636763,00.html
def getYnetText(url):
    # Get the web page source html, as a string
    req = requests.get(url)
    htmlStr = req.text

    # Parse the text as an XML tree, with error-correcting parsing for html.
    htmlTree = etree.parse(StringIO(htmlStr), etree.HTMLParser())  # type: etree.ElementTree
    # Pretty print it again to a string, now correct xml.
    htmlStr2 = etree.tostring(htmlTree, pretty_print=True, encoding='UTF-8').decode("UTF-8")

    # Parse again, but now with the convenient html api (the parsing from this api isn't error-correcting)
    htmlTree = html.document_fromstring(htmlStr2)  # type: etree.ElementTree

    # Get the title
    title = htmlTree.find_class('art_header_title')[0].text

    # Get the subtitle
    subtitle = htmlTree.find_class('art_header_sub_title')[0].text

    # go to span - the parent of all the paragraphs, and get the paragraphs.
    # the div contained p's that aren't sentences.
    # This returns a list of paragraphs, some containing breaks.
    paragraphsRaw = htmlTree.find_class('art_body')[0].xpath("//span/p")

    # Clean all the paragraphs, see the @clean function
    cleanedParagraphs = list(map((lambda x: clean(x.text_content())), paragraphsRaw))

    # Take only the non-empty paragraphs after cleaning
    paragraphs = list(filter((lambda x: len(x) > 0), cleanedParagraphs))

    return [title, subtitle] + paragraphs


def clean(p):
    # Eliminate all newlines & tabs, as they are insignificant in html.
    noNewlines = re.sub(r"\n|\xa0|\t", " ", p)
    # Eliminate repeating spaces, as they are insignificant in html.
    noDuplicateSpaces = re.sub(" +", " ", noNewlines)
    return noDuplicateSpaces.strip()
