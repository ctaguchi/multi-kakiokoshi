import unittest
import segment


class TestNormalizeText(unittest.TestCase):
    def test_normalize_test(self):
        test_cases = [
            ("Hello, world!", {"norm_text": "Hello world!", "std_text": "hello world"}),
            ("Well, <hesitation> I don't know.", {"norm_text": "Well I don't know", "std_text": "well i don't know."}),
            ("[laugh] Wow this is funny [laugh]", {"norm_text": "Wow this is funny", "std_text": "wow this is funny"}),
        ]
        
        for input_text, expected_output in test_cases:
            with self.subTest(input_text=input_text):
                self.assertEqual(segment.normalize_text(input_text), expected_output)


if __name__ == "__main__":
    unittest.main()