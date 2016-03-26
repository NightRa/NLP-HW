# Determine whether the current character indicates the end of a sentence.
# Given the surrounding characters, (that may not exist, if at the start/end of the string)
# Important property: isEndOfSentence is symmetric: can switch @before and @after, and get the same result.
#  This is so that it will work for hebrew too, no matter how hebrew and numbers are sequenced together.
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

# TODO: Move all the tests to here in the end, to show the checker how nice it works.
'''
Example: "Hello world! How are you doing today?? I'm Ilan, and my birthday is at 06.01.1997. What's your's? Hello.."
Result: ['Hello world!', ' How are you doing today??', " I'm Ilan, and my birthday is at 06.01.1997.", " What's your's?", ' Hello..']
'''
'''
Example: "Today is Thursday. Hello world."
Result: ['Today is Thursday.', ' Hello world.']
'''
'''
Example: "שלום .3.14 מה שלומך?"
Result: ['שלום .', '3.14 מה שלומך?']
'''
'''
Example: "Hi! Hello"
Result: ['Hi!', ' Hello']
'''
'''
Example: ""
Result: []
'''


def toSentences(paragraphs):
    res = []
    for p in paragraphs:
        # Each paragraph produces a list of sentences.
        res.extend(splitToSentences(p))
    # Sentences shouldn't start with a space - an artifact of splitting right after a dot.
    return map((lambda sentence: sentence.strip()), res)
