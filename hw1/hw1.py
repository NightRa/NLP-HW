# Authors:
#   Ilan Godik       - 316315332
#   Charlie Mubariky - 316278118

import requests
from lxml import html
from lxml import etree
from io import StringIO
import re
import sys
import os
import codecs

########################################################################################################
########################### Part 1: Get text from Ynet html ############################################
########################################################################################################

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


########################################################################################################
########################### Part 2: Splitting the paragraphs to sentences ##############################
########################################################################################################

# Determine whether the current character indicates the end of a sentence.
# Given the surrounding characters, (that may not exist, if at the start/end of the string)
# Dilemma: Should we split to sentences inside quotes?
# We decided to do so.
# Also, we decided not to split sentences after a dash '-', as usually they indicate elaboration, not new sentences.
def isEndOfSentence(inQuotes, before, current, after):
    # If the next char is terminal, then it's not the end of a sentence:
    #  if the current is not terminal, then it's not the end anyway,
    #  and if the current is terminal, then we have a sequence of terminal chars.
    # We do not need to check whether @next is None, because if it is, it will fail the check, so it includes this case.
    if isTerminalChar(after):
        return False
    # If it's a sentence that ends right before the end of the quote, cut the sentence after the quote.
    # We don't need to check with @isQuotationQuote because we wouldn't break the sentence in either case.
    elif inQuotes and after == '"':
        return False
    # If it's an abbreviation, we won't have a comma after the '"',
    # So no need to check with @isQuotationQuote.
    # Also, we only break to a new line if it's followed by a comma or a terminal char such as a dot.
    # This is so that we won't make idioms go on a new line.
    # Also, we decided not to go to a new line if we have a terminal char just before the end of quotes,
    # such as: 'He said: "Are you sure?" and I disagreed'.
    elif not inQuotes and before == '"' and (current == ',' or isTerminalChar(current)):
        return True
    elif isTerminalChar(current):
        # Don't break sentence on decimal points and on times.
        if (current == '.' or current == ':') and before.isdigit() and after.isdigit():
            return False
        else:
            return True
    else:
        return False


# A function to check whether the current character is a quotation opening/closing, ignoring abbreviations such as ק"מ.
def isQuotationQuote(before, current, after):
    return current == '"' and (not before.isalpha() or not after.isalpha())


def isTerminalChar(c):
    return c == '?' or c == '!' or c == '.' or c == ';' or c == ':'


# Takes a paragraphs and returns a list representing all the sentences in that paragraph.
def splitToSentences(paragraph):
    # Split at '?', '!', and '.' only if not inside a number.
    # inside a number: if the character before it *and* after it is a digit.
    # and also, if we have a couple of consecutive dots, they should behave as one.
    # e.g. unsureness: "I don't know... Are you sure?" should be 2 sentences, not more.
    #   We can't determine without deeper analysis who talks after the "...",
    #   so we decided to favor a new sentence in this case.
    # Also, "??" and "?!" should be considered as one.
    # Because e.g. "How many items did you buy? We bought 3. That's good."
    #   ^ The '.' here does indicate the end of a sentence.
    sentences = []
    prefix = ""
    str = paragraph

    if len(str) == 1 and isTerminalChar(str[0]):
        sentences.append(str[0])
        str = str[1:]
    elif len(str) > 1 and isEndOfSentence(False, "", str[0], str[1]):
        sentences.append(str[0])
        str = str[1:]

    inQuotes = False
    if len(str) > 0:
        inQuotes = str[0] == '"'

    # See pseudo-code below for explanation.
    while len(str) > 2:
        before = str[0]
        current = str[1]
        after = str[2]
        rest = str[3:]
        if isEndOfSentence(inQuotes, before, current, after):
            sentences.append(prefix + before + current)
            prefix = ""
            str = after + rest
        else:
            prefix = prefix + before
            str = current + after + rest

        if isQuotationQuote(before, current, after):
            inQuotes = not inQuotes
    if prefix + str != "":
        sentences.append(prefix + str)
    return sentences

def toSentences(paragraphs):
    res = []
    for p in paragraphs:
        # Each paragraph produces a list of sentences.
        res.extend(splitToSentences(p))
    # Sentences shouldn't start with a space - an artifact of splitting right after a dot.
    return list(map((lambda sentence: sentence.strip()), res))

########################################################################################################
########################### Part 2: Splitting Unit Tests & Pseudo-Code #################################
########################################################################################################

# Pseudo-code for splitToSentences
'''
f (prefix, "") =
    # we got to the end of the sentence.
    if(prefix != ""):
        sentences += prefix
f (prefix, "a") =
    # if it's a terminal char or not, we should anyway return this sentence.
    sentences += (prefix + "a")
f (prefix, "ab") =
    # before = "a", current = "b".
    # whether or not "b" is terminal, we should return the whole sentence anyway.
    sentences += (prefix + "ab")

Overall, add (prefix + str) if it's not empty.

f (prefix, (before, current, after, rest)) =
    if isEndOfSentence(before, current, after):
        sentences += (prefix + before + current)
        recurse on (prefix = [], after + rest)
    else:
        recurse on (prefix = prefix + before, current + after + rest)
'''

'''
Example: "Hello world! How are you doing today?? I'm Ilan, and my birthday is at 06.01.1997. What's your's? Hello.."
Result: ['Hello world!', ' How are you doing today??', " I'm Ilan, and my birthday is at 06.01.1997.", " What's your's?", ' Hello..']
###
Example: "Today is Thursday. Hello world."
Result: ['Today is Thursday.', ' Hello world.']
###
Example: "שלום .3.14 מה שלומך?"
Result: ['שלום .', '3.14 מה שלומך?']
###
Example: "Hi! Hello"
Result: ['Hi!', ' Hello']
###
Example: ""
Result: []
###
Example: ".hi"
Result: ['.', 'hi']
###
Example: "... oh hi there!"
Result: ['...', ' oh hi there!']
###
Example: 'Hello. He said: "Hi there! Hello."'
Result: ['Hello.', ' He said:', ' "Hi there!', ' Hello."']
###
Example: 'What is the time? he said: "The time is 13:30". I answered: "Are you sure?".'
Result: ['What is the time?', ' he said:', ' "The time is 13:30".', ' I answered:', ' "Are you sure?".']
###
Example: 'He said: "What is the time?"'
Result: ['He said:', ' "What is the time?"']
###
Example: 'He said: "Are you sure?" and I disagreed'
Result: ['He said:', ' "Are you sure?" and I disagreed']
###
Example: 'כה הוא אמר: "שלום. מה שלומך?"'
Result: ["כה הוא אמר:", ' "שלום.', ' מה שלומך?"']
###
Example: 'He said: "Pi is special".'
Result: ['He said:', ' "Pi is special".']
###
Example: 'He said: "Pi is tasty", but he did not understand...'
Result: ["He said:", ' "Pi is tasty",', ' but he did not understand...']
###
Example: 'הוא אמר: "ק"מ זה חשוב". "מסכים."'
Result: ['הוא אמר:', ' "ק"מ זה חשוב".', ' "מסכים."']
###
Example: 'הוצגו "מעגלי ההשפעה" של העבודות'
Result: ['הוצגו "מעגלי ההשפעה" של העבודות']
'''


########################################################################################################
############################## Part 3: Tokenization ####################################################
########################################################################################################

# Tokens:
#  words = alpha+, and ['"','-'] if between letters.
# Quotes, by the same rule as before.
# '"', '-' if not between letters.
# [ '!' , ';' ,  ',' , '?' , '(' , ')' , ':', '/', '\', '+', '=']
# '...' should be one token.
# '.' if not a decimal point: between digits, and need to handle sequences.
# We don't take $, %, ₪, as they give the meaning to the digits before them.
# Same for @, _, we shouldn't break up an email.
# '#' isn't used in hebrew.
# '*' should be a separate token, unless it's a phone number: *123 = followed by a number.
# "'" shouldn't be a separate token, it's used for letter alternatives or abbr.:
# ג' ז' וכו'

# Output the first token and the rest of the sentence.
def eatToken(s):
    s = s.strip()  # Throw away spaces in the beginning (after we've eaten a word)
    if len(s) == 0:
        return None

    i = 0
    while 0 <= i < len(s) and not isNewToken(tryCharAt(s, i - 1), s[i], tryCharAt(s, i + 1), tryCharAt(s, i + 2)):
        i += 1

    return s[0:i + 1], s[i + 1:]


def tokenize(s):
    tokens = []
    while len(s) > 0:
        token, rest = eatToken(s)
        if token is not None:
            tokens.append(token)
        s = rest
    return tokens


def tryCharAt(s, i):
    if 0 <= i < len(s):
        return s[i]
    else:
        return ""


# Whether to split after the second char (ab|cd)
# We look at a window of size 4 for the following reason:
# If we had a window of size 3, then we couldn't decide whether to split:
# '"k"m ' : if we look at the second 3, we see: '"m ', and we don't know whether we should split
# before the m or not, we need more info: what's before the ".
def isNewToken(prevPrev, prev, next, nextNext):
    # if the last or next char is always a token, split.
    if alwaysToken(prev) or alwaysToken(next):
        return True
    # Not surrounded by letters
    # 3"7 and k"m
    elif (next == '"' or next == '-') and not (prev.isalnum() and nextNext.isalnum()):
        return True
    # This is why we needed a larger window: 'k"m ' should split, and 'y"*1' should split
    # Current may be whatever, but we must analyse the ", which is in before,
    #   and we need to look 1 character back.
    elif prev == '"' and not (prevPrev.isalnum() and next.isalnum()):
        return True
    elif next == ':' and not (prev.isdigit() and nextNext.isdigit()):
        return True
    elif next == '.' and not prev == '.' and not (prev.isdigit() and nextNext.isdigit()):
        return True
    elif next == '*' and not nextNext.isdigit():
        return True
    else:
        return False


def alwaysToken(c):
    return c in " !?,();/\\+="


def tokenizeAllSentences(sentences):
    # tokenize each sentence and separate tokens by a space
    return list(map((lambda sentence: " ".join(tokenize(sentence))), sentences))

########################################################################################################
############################## Part 3: Tokenization Unit Tests #########################################
########################################################################################################

# tokenize() tests
'''
Example: "Hello! How are you today?  I don't know..., 3.14"
Result: ['Hello', '!', 'How', 'are', 'you', 'today', '?', 'I', "don't", 'know', '...', ',', '3.14']
###
Example: 'הוא אמר: "ק"מ זה חשוב". "מסכים."'
Result: ['הוא', 'אמר', ':', '"', 'ק"מ', 'זה', 'חשוב', '"', '.', '"', 'מסכים', '.', '"']
###
Example: "a"
Result: ['a']
###
Example: '"k"m"'
Result: ['"', 'k"m', '"']
###
Example: 'ה-15'
Result: ['ה-15']
###
Example: 'בית-הספר'
Result: ['בית-הספר']
###
Example: 'עם טוויסט- מועד'
Result: ['עם', 'טוויסט', '-', 'מועד']
###
Example: "13:15:00"
Result: ["13:15:00"]
'''

# isNewToken() tests
# def split(s):
#     return isNewToken(s[0], s[1], s[2], s[3])

'''
Example: 'lo..'
Result: True
###
Example: 'o...'
Result: False
###
Example: '... '
Result: False
###
Example: '.. h'
Result: True
###
Example: ' 3.1'
Result: False
###
Example: '3.14'
Result: False
###
Example: ' 3. '
Result: True
###
Example: ': "k'
Result: True
###
Example: ' "k"'
Result: True
###
Example: ' k"m'
Result: False
###
Example: 'k"m '
Result: False
###
Example: 'lo" '
Result: True
###
Example: 'xy"*'
Result: True
###
Example: 'y"*!'
Result: True
###
Example: ' !he'
Result: True
'''

########################################################################################################
############################# Part Main - input and output #############################################
########################################################################################################

url = sys.argv[1]
outputFolder = sys.argv[2]

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

textParagraphs = getYnetText(url)
sentences = toSentences(textParagraphs)
tokenized = tokenizeAllSentences(sentences)

with codecs.open(os.path.join(outputFolder, 'article.txt'), 'w', 'utf-8') as f:
    for paragraph in textParagraphs:
        f.write(paragraph + '\r\n')

with codecs.open(os.path.join(outputFolder, 'article_sentences.txt'), 'w', 'utf-8') as f:
    for sentence in sentences:
        f.write(sentence + '\r\n')

with codecs.open(os.path.join(outputFolder, 'article_tokenized.txt'), 'w', 'utf-8') as f:
    for line in tokenized:
        f.write(line + '\r\n')
