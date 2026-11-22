"""Unit test cho module tiền xử lý văn bản và chống rò rỉ nhãn."""

import unittest

from src.preprocessing.clean import clean_text
from src.preprocessing.leakage import remove_label_leakage
from src.preprocessing.tokenizer import preprocess_text, tokenize_vietnamese


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_removes_html_url_and_lowercases(self) -> None:
        raw = "<p>Xin CHÀO https://example.com/truyen !!!</p>"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "xin chào")

    def test_clean_text_handles_none_and_empty(self) -> None:
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")

    def test_remove_label_leakage_strips_genre_names(self) -> None:
        text = "đây là bộ truyện tiên hiệp pha chút ngôn tình rất hay"
        result = remove_label_leakage(text)
        self.assertNotIn("tiên hiệp", result)
        self.assertNotIn("ngôn tình", result)
        self.assertIn("rất hay", result)

    def test_remove_label_leakage_strips_header_genre_block(self) -> None:
        text = "thể loại ngôn tình cận đại làm ruộng văn án vừa xuyên qua cô đã gặp nam chính"
        result = remove_label_leakage(text)
        self.assertNotIn("thể loại", result)
        self.assertNotIn("văn án", result)
        self.assertIn("vừa xuyên qua", result)

    def test_tokenize_filters_stopwords(self) -> None:
        tokens = tokenize_vietnamese("thiếu niên bước vào hành trình", stopwords={"vào"})
        self.assertNotIn("vào", tokens.split())
        self.assertTrue(len(tokens) > 0)

    def test_preprocess_text_end_to_end(self) -> None:
        cleaned, tokens = preprocess_text(
            "<b>Truyện kinh dị</b> tại https://abc.vn ngôi nhà cổ",
            stopwords={"tại"},
        )
        self.assertNotIn("kinh dị", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertIn("ngôi", tokens)

    def test_clean_text_splits_case_boundaries(self) -> None:
        self.assertEqual(clean_text("Góc trên bên tráiĐại Khư"), "góc trên bên trái đại khư")
        self.assertEqual(clean_text("HEEditor: abc HEĐộ dài"), "he editor abc he độ dài")

    def test_clean_text_normalizes_nfd_before_splitting(self) -> None:
        self.assertEqual(clean_text("Thê\u0301Có"), "thế có")

    def test_clean_text_preserves_camelcase_whitelist(self) -> None:
        self.assertEqual(
            clean_text("Dùng WeChat và iPhone xem YouTube"),
            "dùng wechat và iphone xem youtube",
        )

    def test_preprocess_text_strips_joined_header_and_leakage(self) -> None:
        cleaned, tokens = preprocess_text(
            "Tác giả: Nhĩ CănThể loại: Tiên HiệpGiới thiệu:Thiên địa là vạn vật",
            stopwords=set(),
        )
        self.assertNotIn("thể loại", cleaned)
        self.assertNotIn("tiên hiệp", cleaned)
        self.assertIn("thiên địa", cleaned)
        self.assertIn("vạn_vật", tokens)


if __name__ == "__main__":
    unittest.main()
