import unittest
from unittest.mock import patch

import pandas as pd

from app import (
    SORT_BY_RECENT,
    add_sentiment,
    discover_topics,
    fetch_youtube_comments,
    prepare_comments,
    prepare_pasted_comments,
    validate_youtube_url,
)


class CommentAnalysisTests(unittest.TestCase):
    def test_prepares_text_and_optional_author_columns(self) -> None:
        frame = pd.DataFrame(
            {"Text": ["  Great video! ", None, ""], "Author": ["A", "B", "C"]}
        )
        result = prepare_comments(frame, "Text")
        self.assertEqual(result["comment"].tolist(), ["Great video!"])
        self.assertEqual(result["author"].tolist(), ["A"])

    def test_prepares_pasted_comments_and_skips_blank_lines(self) -> None:
        result = prepare_pasted_comments(" Great video!\n\nCould you explain more? \n")
        self.assertEqual(
            result["comment"].tolist(),
            ["Great video!", "Could you explain more?"],
        )

    def test_adds_sentiment_labels(self) -> None:
        frame = pd.DataFrame(
            {"comment": ["This is wonderful!", "This is awful.", "The video is here."]}
        )
        result = add_sentiment(frame)
        self.assertEqual(
            result["sentiment"].tolist(), ["Positive", "Negative", "Neutral"]
        )

    def test_topic_discovery_handles_insufficient_text(self) -> None:
        self.assertEqual(discover_topics(["hello", "world"]), [])

    def test_topic_discovery_groups_comments_by_recurring_terms(self) -> None:
        topics = discover_topics(
            [
                "Cats purr softly",
                "Cats are playful",
                "Dogs bark loudly",
                "Dogs enjoy walks",
                "Baking bread smells good",
                "Bread rises slowly",
            ]
        )
        self.assertEqual(sum(topic["comments"] for topic in topics), 6)
        self.assertGreaterEqual(len(topics), 2)

    def test_validates_supported_youtube_url_formats(self) -> None:
        for url in (
            "https://www.youtube.com/watch?v=abcdefghijk",
            "https://youtu.be/abcdefghijk",
            "https://youtube.com/shorts/abcdefghijk",
        ):
            with self.subTest(url=url):
                self.assertEqual(validate_youtube_url(url), url)

    def test_rejects_non_youtube_and_malformed_video_urls(self) -> None:
        for url in (
            "https://example.com/watch?v=abcdefghijk",
            "https://www.youtube.com/watch?v=short",
            "javascript://youtube.com/watch?v=abcdefghijk",
            "https://youtube.com:8080/watch?v=abcdefghijk",
            "https://user@youtube.com/watch?v=abcdefghijk",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_youtube_url(url)

    @patch("app.YoutubeCommentDownloader")
    def test_fetches_all_comments_and_prepares_authors(self, downloader_class) -> None:
        downloader = downloader_class.return_value
        downloader.get_comments_from_url.return_value = iter(
            {"text": f"Comment {index}", "author": f"Author {index}"}
            for index in range(600)
        )

        comments = fetch_youtube_comments("https://youtu.be/abcdefghijk")

        self.assertEqual(len(comments), 600)
        self.assertEqual(comments["comment"].iloc[-1], "Comment 599")
        self.assertEqual(comments["author"].iloc[-1], "Author 599")
        downloader.get_comments_from_url.assert_called_once_with(
            "https://youtu.be/abcdefghijk", sort_by=SORT_BY_RECENT
        )

    @patch("app.YoutubeCommentDownloader")
    def test_fetch_returns_empty_comments_when_video_has_no_comments(
        self, downloader_class
    ) -> None:
        downloader_class.return_value.get_comments_from_url.return_value = iter(())

        comments = fetch_youtube_comments("https://youtu.be/abcdefghijk")

        self.assertEqual(list(comments.columns), ["comment", "author"])
        self.assertTrue(comments.empty)


if __name__ == "__main__":
    unittest.main()
