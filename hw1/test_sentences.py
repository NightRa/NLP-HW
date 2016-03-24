from unittest import TestCase
from hw1.nlp2 import *


class TestSentences(TestCase):
    def test_splitToSentences1(self):
        self.assertEqual(splitToSentences(
            "Hello world! How are you doing today?? I'm Ilan, and my birthday is at 06.01.1997. What's your's? Hello.."),
                         ['Hello world!', ' How are you doing today??', " I'm Ilan, and my birthday is at 06.01.1997.",
                          " What's your's?", ' Hello..'])

    def test_splitToSentences2(self):
        self.assertEqual(splitToSentences("Today is Thursday. Hello world."),
                         ['Today is Thursday.', ' Hello world.'])

    def test_splitToSentencesHebrewNumbers(self):
        self.assertEqual(splitToSentences("שלום .3.14 מה שלומך?"),
                         ['שלום .', '3.14 מה שלומך?'])

    def test_splitToSentencesEndsNotInTerminal(self):
        self.assertEqual(splitToSentences("Hi! Hello"),
                         ['Hi!', ' Hello'])

    def test_splitToSentencesEmpty(self):
        self.assertEqual(splitToSentences(""), [])

    def test_splitToSentencesStartsWithTerminal(self):
        self.assertEqual(splitToSentences(".hi"), ['.', 'hi'])

    def test_splitToSentencesStartsWithTerminals(self):
        self.assertEqual(splitToSentences("... oh hi there!"), ['...', ' oh hi there!'])

    def test_splitToSentencesWithQuotes(self):
        self.assertEqual(splitToSentences('Hello. He said: "Hi there! Hello."'),
                         ['Hello.', ' He said:', ' "Hi there!', ' Hello."'])

    def test_splitToSentencesHoursQuotes(self):
        self.assertEqual(
            splitToSentences('What is the time? he said: "The time is 13:30." I answered: "Are you sure?"'),
            ['What is the time?', ' he said:', ' "The time is 13:30."', ' I answered:', ' "Are you sure?"'])

    def test_splitToSentencesStartWithQuotes(self):
        self.assertEqual(
            splitToSentences('"What is the time?" he said.'),
            ['"What is the time?"', ' he said.'])

    def test_splitToSentencesHebrewQuotes(self):
        self.assertEqual(splitToSentences('כה הוא אמר: "שלום. מה שלומך?"'),
                         ["כה הוא אמר:", ' "שלום.', ' מה שלומך?"'])

    def test_splitToSentencesDotAfterQuote(self):
        self.assertEqual(splitToSentences('He said: "Pi is special".'),
                         ['He said:', ' "Pi is special".'])

    def test_splitToSentencesCommaAfterQuotes(self):
        self.assertEqual(splitToSentences('He said: "Pi is tasty", but he did not understand...'),
                         ["He said:", ' "Pi is tasty",', ' but he did not understand...'])
