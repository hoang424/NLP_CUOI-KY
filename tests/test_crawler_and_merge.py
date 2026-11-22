"""Unit test cho bộ cào WebTruyen và bước khử trùng lặp ở tiền xử lý."""

import unittest

import lxml.html
import pandas as pd

from src.crawl.crawler import (
    _extract_webtruyen_category_story_urls,
    _extract_webtruyen_max_page,
    parse_webtruyen_story_html,
)
from src.crawl.merge_data import deduplicate_stories, normalize_title


class TestCrawlerAndMerge(unittest.TestCase):
    def test_normalize_title_and_deduplicate_stories(self) -> None:
        self.assertEqual(normalize_title("  Ngôn   Tình  Cổ  Đại "), "ngôn tình cổ đại")

        df = pd.DataFrame(
            [
                {
                    "title": "Truyện A",
                    "summary": "Văn án ngắn hơn của truyện A",
                    "genres_raw": "Ngôn Tình|Cổ Đại",
                },
                {
                    "title": "Truyện B",
                    "summary": "Văn án riêng của truyện B đủ dài để kiểm thử",
                    "genres_raw": "Tiên Hiệp",
                },
                {
                    "title": "  truyện   a ",
                    "summary": "Văn án đầy đủ và chi tiết hơn rất nhiều cho truyện A",
                    "genres_raw": "ngôn tình|Sủng|Xuyên Không",
                },
                {
                    "title": "Truyện C bản sao",
                    "summary": "Văn án riêng của truyện B đủ dài để kiểm thử",
                    "genres_raw": "Huyền Huyễn",
                },
            ]
        )

        dedup = deduplicate_stories(df)
        self.assertEqual(len(dedup), 2)
        self.assertEqual(dedup.iloc[0]["title"], "truyện   a")
        self.assertIn("chi tiết hơn rất nhiều", dedup.iloc[0]["summary"])
        self.assertEqual(dedup.iloc[0]["genres_raw"], "Ngôn Tình|Cổ Đại|Sủng|Xuyên Không")
        self.assertEqual(dedup.iloc[1]["genres_raw"], "Tiên Hiệp|Huyền Huyễn")

    def test_extract_webtruyen_category_pagination_and_main_urls(self) -> None:
        html = """
        <html><body>
          <div class="col-truyen-main">
            <h3><a href="https://wetruyen.com/truyen-mot.101/">Truyện Một</a></h3>
            <h3><a href="/truyen-hai.202/">Truyện Hai</a></h3>
            <h3><a href="https://wetruyen.com/truyen-mot.101/">Truyện Một Trùng</a></h3>
            <ul class="pagination" id="pagination">
              <li><a href="https://wetruyen.com/truyen-tien-hiep/full/trang-2/">2</a></li>
              <li><a href="https://wetruyen.com/truyen-tien-hiep/full/trang-15/">Last</a></li>
            </ul>
          </div>
          <div class="col-truyen-side">
            <h3><a href="https://wetruyen.com/truyen-sidebar-hot.999/">Truyện Hot Sidebar</a></h3>
          </div>
        </body></html>
        """
        doc = lxml.html.fromstring(html)
        self.assertEqual(_extract_webtruyen_max_page(doc), 15)
        self.assertEqual(
            _extract_webtruyen_category_story_urls(doc),
            [
                "https://wetruyen.com/truyen-mot.101/",
                "https://wetruyen.com/truyen-hai.202/",
            ],
        )

    def test_parse_webtruyen_story_html(self) -> None:
        html = """
        <html><body>
          <h1>Kiếm Đạo Độc Tôn</h1>
          <div class="info-holder">
            <div class="info">
              <div><h3>Tác giả:</h3><a itemprop="author">Tác Giả Test</a></div>
              <div><h3>Thể loại:</h3>
                <a itemprop="genre">Tiên Hiệp</a>
                <a itemprop="genre">Kiếm Hiệp</a>
              </div>
              <div><h3>Trạng thái:</h3><span>Full</span></div>
              <div><h3>Đánh giá:</h3><span itemprop="ratingValue">8.9</span></div>
            </div>
          </div>
          <div itemprop="description">
            Thiếu niên bước ra từ tiểu trấn bắt đầu hành trình tu luyện kiếm đạo giữa thiên địa rộng lớn.
          </div>
        </body></html>
        """
        rec = parse_webtruyen_story_html(
            html,
            url="https://wetruyen.com/kiem-dao-doc-ton.888/",
            extra_genres=["Huyền Huyễn"],
            category_crawled="Tiên Hiệp",
            min_summary_words=10,
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["title"], "Kiếm Đạo Độc Tôn")
        self.assertEqual(rec["author"], "Tác Giả Test")
        self.assertEqual(rec["genres_raw"], "Tiên Hiệp|Kiếm Hiệp|Huyền Huyễn")
        self.assertEqual(rec["status"], "Full")
        self.assertEqual(rec["rating"], "8.9")
        self.assertEqual(rec["source"], "webtruyen.vn")
        self.assertEqual(rec["category_crawled"], "Tiên Hiệp")

    def test_parse_webtruyen_story_html_preserves_line_break_spaces(self) -> None:
        html = """
        <html><body>
          <h1>Mục Thần Ký</h1>
          <div class="info-holder">
            <div class="info">
              <div><a itemprop="genre">Huyền Huyễn</a></div>
            </div>
          </div>
          <div itemprop="description">
            Tác giả: Nhĩ Căn<br>Thể loại: Tiên Hiệp<p>Đại Khư tổ huấn đã nói rõ trời tối đừng ra ngoài</p>
          </div>
        </body></html>
        """
        rec = parse_webtruyen_story_html(
            html,
            url="https://wetruyen.com/muc-than-ky.123/",
            min_summary_words=5,
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertNotIn("CănThể", rec["summary"])
        self.assertNotIn("HiệpĐại", rec["summary"])
        self.assertIn("Căn Thể", rec["summary"])
        self.assertIn("Hiệp Đại", rec["summary"])


if __name__ == "__main__":
    unittest.main()
