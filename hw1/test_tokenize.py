from unittest import TestCase
from hw1.nlp3 import *


def split(s):
    return isNewToken(s[0], s[1], s[2], s[3])


class TestTokenize(TestCase):
    def test_tokenize1(self):
        self.assertEqual(tokenize("Hello! How are you today?  I don't know..., 3.14"),
                         ['Hello', '!', 'How', 'are', 'you', 'today', '?', 'I', "don't", 'know', '...', ',', '3.14'])

    def test_tokenizeAbbr(self):
        self.assertEqual(tokenize('הוא אמר: "ק"מ זה חשוב". "מסכים."'),
                         ['הוא', 'אמר', ':', '"', 'ק"מ', 'זה', 'חשוב', '"', '.', '"', 'מסכים', '.', '"'])

    def test_tokenizeSingleLetter(self):
        self.assertEqual(tokenize("a"), ['a'])

    def test_tokenizeMixedTerminals(self):
        self.assertEqual(tokenize('"k"m"'), ['"', 'k"m', '"'])

    def test_tokenizeDashInWordOrNum(self):
        self.assertEqual(tokenize('ה-15'), ['ה-15'])

    def test_tokenizeDashInWord(self):
        self.assertEqual(tokenize('בית-הספר'), ['בית-הספר'])

    def test_tokenizeDashNotInWord(self):
        self.assertEqual(tokenize('עם טוויסט- מועד'),
                         ['עם', 'טוויסט', '-', 'מועד'])

    def test_time(self):
        self.assertEqual(tokenize("13:15:00"), ["13:15:00"])

    def test_split01(self):
        self.assertEqual(split('lo..'), True)

    def test_split02(self):
        self.assertEqual(split('o...'), False)

    def test_split03(self):
        self.assertEqual(split('... '), False)

    def test_split04(self):
        self.assertEqual(split('.. h'), True)

    def test_split05(self):
        self.assertEqual(split(' 3.1'), False)

    def test_split06(self):
        self.assertEqual(split('3.14'), False)  # 3.14

    def test_split07(self):
        self.assertEqual(split(' 3. '), True)  # Hi 3.

    def test_split08(self):
        self.assertEqual(split(': "k'), True)

    def test_split09(self):
        self.assertEqual(split(' "k"'), True)

    def test_split10(self):
        self.assertEqual(split(' k"m'), False)

    def test_split11(self):
        self.assertEqual(split('k"m '), False)

    def test_split12(self):
        self.assertEqual(split('lo" '), True)

    def test_split13(self):
        self.assertEqual(split('xy"*'), True)

    def test_split14(self):
        self.assertEqual(split('y"*!'), True)

    def test_split15(self):
        self.assertEqual(split(' !he'), True)
