"""Unit test cho bộ cào WebTruyen và bước khử trùng lặp ở tiền xử lý."""

import unittest

import lxml.html
import pandas as pd

from src.crawl.crawler import (
    _extract_metruyenchu_max_page,
    _extract_truyenfull_category_story_urls,
    _extract_truyenfull_max_page,
    _extract_webtruyen_category_story_urls,
    _extract_webtruyen_max_page,
    _is_valid_webnovel_url,
    parse_metruyenchu_story_html,
    parse_truyenfull_story_html,
    parse_webnovel_story_html,
    parse_webtruyen_story_html,
)
from src.crawl.merge_data import deduplicate_stories, normalize_genres, normalize_title


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

    def test_parse_webnovel_story_html_and_url_filter(self) -> None:
        self.assertTrue(_is_valid_webnovel_url("https://webnovel.vn/kiem-khieu-linh-tieu/"))
        self.assertFalse(_is_valid_webnovel_url("https://webnovel.vn/all/"))
        self.assertFalse(_is_valid_webnovel_url("https://webnovel.vn/all/2/"))
        self.assertFalse(_is_valid_webnovel_url("https://webnovel.vn/pub/abc/"))
        self.assertFalse(_is_valid_webnovel_url("https://webnovel.vn/kiem-khieu-linh-tieu/chuong-1/"))
        self.assertFalse(_is_valid_webnovel_url("https://other.vn/kiem-khieu-linh-tieu/"))

        html = """
        <html><body>
          <h1 class="book-detail__title">Kiếm Khiếu Linh Tiêu</h1>
          <div class="book-detail__author">Tác Giả Demo</div>
          <div class="book-detail__genres">
            <a href="/tien-hiep/">Tiên Hiệp</a>
            <a href="/kiem-hiep/">Kiếm Hiệp</a>
          </div>
          <span class="book-detail__status">Đã Hoàn</span>
          <span class="book-detail__stat-text">4.6</span>
          <span class="book-detail__stat-text">379 chương</span>
          <div class="book-detail__summary">
            <p>Đoạn mở đầu giới thiệu</p><p>Hành trình tu tiên gian nan</p>
          </div>
        </body></html>
        """
        rec = parse_webnovel_story_html(html, "https://webnovel.vn/kiem-khieu-linh-tieu/")
        self.assertEqual(rec["title"], "Kiếm Khiếu Linh Tiêu")
        self.assertEqual(rec["author"], "Tác Giả Demo")
        self.assertEqual(rec["genres_raw"], "Tiên Hiệp|Kiếm Hiệp")
        self.assertEqual(rec["status"], "Đã Hoàn")
        self.assertEqual(rec["chapters"], "379 chương")
        self.assertEqual(rec["rating"], "4.6")
        self.assertEqual(rec["source"], "webnovel.vn")
        self.assertIn("giới thiệu Hành trình", rec["summary"])

    def test_parse_metruyenchu_story_html_and_pagination(self) -> None:
        cat_html = """
        <html><body>
          <ul class="pagination">
            <li><a href="/the-loai/tien-hiep?page=2">2</a></li>
            <li><a href="/the-loai/tien-hiep?page=14">14</a></li>
          </ul>
        </body></html>
        """
        cat_doc = lxml.html.fromstring(cat_html)
        self.assertEqual(_extract_metruyenchu_max_page(cat_doc), 14)

        seo_only_html = """
        <html><body>
          <h1>Truyện Không Văn Án</h1>
          <li class="genres"><a href="/the-loai/tien-hiep">Tiên Hiệp</a></li>
          <div class="scrolltext">
            MeTruyenChu vừa cập nhật bộ truyện mới Truyện Không Văn Án của tác giả Vô Danh. Đây là bộ truyện được đăng tải mới này.
          </div>
        </body></html>
        """
        self.assertIsNone(
            parse_metruyenchu_story_html(seo_only_html, "https://metruyenchu.com.vn/truyen-khong-van-an")
        )

        story_html = """
        <html><body>
          <h1>Từ Tam Quốc Bắt Đầu Tu Tiên</h1>
          <a itemprop="author">Phàm Nhân Nhậm</a>
          <ul><li class="genres"><a href="/the-loai/tien-hiep">Tiên Hiệp</a><a href="/the-loai/lich-su">Lịch Sử</a></li></ul>
          <div class="scrolltext">
            MeTruyenChu vừa cập nhật bộ truyện mới Từ Tam Quốc Bắt Đầu Tu Tiên. Đây là bộ truyện được đăng tải mới này.
            Giới thiệu nội dung: <p>Thiếu niên xuyên về cuối thời Đông Hán</p><p>giữa thời loạn thế bắt đầu con đường tu tiên trường sinh.</p>
          </div>
        </body></html>
        """
        rec = parse_metruyenchu_story_html(
            story_html,
            "https://metruyenchu.com.vn/tu-tam-quoc-bat-dau-tu-tien",
            min_summary_words=10,
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["title"], "Từ Tam Quốc Bắt Đầu Tu Tiên")
        self.assertEqual(rec["author"], "Phàm Nhân Nhậm")
        self.assertEqual(rec["genres_raw"], "Tiên Hiệp|Lịch Sử")
        self.assertEqual(rec["source"], "metruyenchu.com.vn")
        self.assertNotIn("MeTruyenChu vừa cập nhật", rec["summary"])
        self.assertIn("Đông Hán giữa thời loạn thế", rec["summary"])

    def test_parse_truyenfull_story_html_and_pagination(self) -> None:
        cat_html = """
        <html><body>
          <div class="col-truyen-main">
            <h3><a href="https://truyenfull.live/truyen-tham-hiem-1/">Truyện 1</a></h3>
            <h3><a href="/truyen-tham-hiem-2/">Truyện 2</a></h3>
            <ul class="pagination">
              <li><a href="https://truyenfull.live/the-loai/tham-hiem/trang-2/">2</a></li>
              <li><a href="https://truyenfull.live/the-loai/tham-hiem/trang-9/">9</a></li>
            </ul>
          </div>
          <div class="col-truyen-side">
            <h3><a href="https://truyenfull.live/truyen-sidebar/">Sidebar</a></h3>
          </div>
        </body></html>
        """
        cat_doc = lxml.html.fromstring(cat_html)
        self.assertEqual(_extract_truyenfull_max_page(cat_doc), 9)
        self.assertEqual(
            _extract_truyenfull_category_story_urls(cat_doc),
            [
                "https://truyenfull.live/truyen-tham-hiem-1/",
                "https://truyenfull.live/truyen-tham-hiem-2/",
            ],
        )

        story_html = """
        <html><body>
          <h3 class="title">Bí Ẩn Lăng Mộ Cổ</h3>
          <div class="info">
            <div><a href="/tac-gia/nam-phai/">Nam Phái</a></div>
            <div>
              <a href="/the-loai/linh-di/">Linh Dị</a>
              <a href="/the-loai/tham-hiem/">Thám Hiểm</a>
            </div>
            <span>Full</span>
          </div>
          <div class="desc-text">
            Chuyến hành trình khám phá cổ mộ kỳ bí<br>dưới lòng đất đầy rẫy cạm bẫy sinh tử.
          </div>
        </body></html>
        """
        rec = parse_truyenfull_story_html(
            story_html,
            url="https://truyenfull.live/bi-an-lang-mo-co/",
            extra_genres=["Trinh Thám"],
            category_crawled="Thám Hiểm",
            min_summary_words=10,
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["title"], "Bí Ẩn Lăng Mộ Cổ")
        self.assertEqual(rec["author"], "Nam Phái")
        self.assertEqual(rec["genres_raw"], "Linh Dị|Thám Hiểm|Trinh Thám")
        self.assertEqual(rec["status"], "Full")
        self.assertEqual(rec["source"], "truyenfull.live")
        self.assertIn("kỳ bí dưới lòng đất", rec["summary"])

    def test_normalize_genres_35_label_taxonomy(self) -> None:
        raw = (
            "Đô Thị|Hiện Đại|Xuyên Nhanh|Dã Sử|Truyện Teen|Dị Năng|Mạt Thế|"
            "Khoa Huyễn|Thám Hiểm|Nữ Phụ|Quan Trường|Quân Sự|Đoản Văn|"
            "Tiểu Thuyết|Khác|Việt Nam|Light Novel|Đông Phương|Phương Tây|"
            "Phát triển bản thân"
        )
        labels = normalize_genres(raw)
        expected = [
            "hien_dai",
            "xuyen_khong",
            "lich_su",
            "hoc_duong",
            "di_nang",
            "mat_the",
            "khoa_huyen",
            "tham_hiem",
            "nu_phu",
            "quan_truong",
            "quan_su",
            "doan_van",
            "tieu_thuyet",
            "khac",
        ]
        self.assertEqual(labels, expected)


if __name__ == "__main__":
    unittest.main()
