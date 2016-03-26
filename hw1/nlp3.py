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
    while 0 <= i < len(s) and not isNewToken(tryCharAt(s, i-1), s[i], tryCharAt(s, i + 1), tryCharAt(s, i + 2)):
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
    return map((lambda sentence: " ".join(tokenize(sentence))), sentences)


# Hello ...
# input: string
# outputs: tokens
# def tokenize(sentence):


from hw1.nlp2 import sentences
tokenized = tokenizeAllSentences(sentences)

for line in tokenized:
    print(line)
