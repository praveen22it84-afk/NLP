import math
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import streamlit as st
from youtube_comment_downloader import SORT_BY_RECENT, YoutubeCommentDownloader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


TOKEN_PATTERN = re.compile(r"[a-z][a-z']{1,}")
STOP_WORDS = {
    "about", "after", "again", "against", "also", "and", "any", "are", "because",
    "been", "before", "being", "but", "can", "could", "did", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "her", "here", "hers", "him", "his", "how", "into", "its",
    "itself", "just", "more", "most", "not", "off", "once", "only", "other",
    "our", "out", "over", "own", "same", "she", "should", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "too", "under", "until", "very", "was", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "youtube", "video",
}


def validate_youtube_url(url: str) -> str:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Enter a valid YouTube video URL.") from error
    if parsed.scheme not in {"http", "https"} or hostname not in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    } or parsed.username or parsed.password or port not in {
        None,
        80 if parsed.scheme == "http" else 443,
    }:
        raise ValueError("Enter a valid YouTube video URL.")

    video_id = ""
    if hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) == 2 and path_parts[0] in {"embed", "shorts", "live", "v"}:
            video_id = path_parts[1]

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Enter a valid YouTube video URL.")
    return url.strip()


def fetch_youtube_comments(url: str) -> pd.DataFrame:
    validated_url = validate_youtube_url(url)
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(
        validated_url, sort_by=SORT_BY_RECENT
    )
    return prepare_comments(
        pd.DataFrame(list(comments), columns=["text", "author"]), "text"
    )


def prepare_comments(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    if text_column not in frame.columns:
        raise ValueError(f"Column '{text_column}' was not found.")

    comments = pd.DataFrame({"comment": frame[text_column].fillna("").astype(str)})
    comments["comment"] = comments["comment"].str.strip()
    comments = comments[comments["comment"] != ""].copy()

    author_column = next(
        (column for column in frame.columns if column.lower() in {"author", "username"}),
        None,
    )
    comments["author"] = (
        frame.loc[comments.index, author_column].fillna("").astype(str)
        if author_column
        else ""
    )
    return comments.reset_index(drop=True)


def prepare_pasted_comments(text: str) -> pd.DataFrame:
    comments = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    return pd.DataFrame({"comment": comments, "author": [""] * len(comments)})


def add_sentiment(comments: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    result = comments.copy()
    scores = result["comment"].map(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    result["sentiment_score"] = scores
    result["sentiment"] = scores.map(
        lambda score: "Positive"
        if score >= 0.05
        else "Negative"
        if score <= -0.05
        else "Neutral"
    )
    return result


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOP_WORDS
    ]


def get_common_terms(texts: list[str], limit: int = 15) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize(text)
        counts.update(tokens)
        counts.update(
            f"{left} {right}"
            for left, right in zip(tokens, tokens[1:])
        )
    return pd.DataFrame(
        [{"term": term, "mentions": count} for term, count in counts.most_common(limit)],
        columns=["term", "mentions"],
    )


def discover_topics(
    texts: list[str], max_topics: int = 5
) -> list[dict[str, Any]]:
    documents = [Counter(tokenize(text)) for text in texts]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())

    document_count = len(documents)
    if document_count < 3:
        return []

    term_scores: Counter[str] = Counter()
    for document in documents:
        for term, frequency in document.items():
            if document_frequency[term] > 1:
                inverse_frequency = math.log(
                    (document_count + 1) / (document_frequency[term] + 1)
                )
                term_scores[term] += frequency * inverse_frequency

    if not term_scores:
        return []

    theme_terms = [
        term for term, _ in term_scores.most_common(max_topics * 3)
    ]
    groups: dict[str, list[Counter[str]]] = {}
    for document in documents:
        candidates = [
            term for term in theme_terms if term in document
        ]
        if not candidates:
            continue
        main_term = max(
            candidates,
            key=lambda term: document[term]
            * math.log((document_count + 1) / (document_frequency[term] + 1)),
        )
        groups.setdefault(main_term, []).append(document)

    topics: list[dict[str, Any]] = []
    for main_term, group in groups.items():
        if len(group) < 2:
            continue
        group_terms: Counter[str] = Counter()
        for document in group:
            group_terms.update(document)
        keywords = [main_term] + [
            term for term, _ in group_terms.most_common(5) if term != main_term
        ]
        topics.append(
            {
                "topic": f"Theme: {main_term}",
                "keywords": ", ".join(keywords[:5]),
                "comments": len(group),
            }
        )
    return sorted(topics, key=lambda topic: topic["comments"], reverse=True)


def analyze_comments(
    comments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    analyzed = add_sentiment(comments)
    terms = get_common_terms(analyzed["comment"].tolist())
    topics = discover_topics(analyzed["comment"].tolist())
    return analyzed, terms, topics


def main() -> None:
    st.set_page_config(
        page_title="YouTube Comment Analyzer", page_icon="📊", layout="wide"
    )
    st.title("YouTube Comment Analyzer")
    st.write("Explore comment sentiment, frequently used terms, and discussion themes.")

    source = st.sidebar.radio(
        "Comment source", ["YouTube URL", "Paste comments", "CSV upload"]
    )
    comments: pd.DataFrame | None = None

    if source == "YouTube URL":
        with st.form("youtube_url_form"):
            video_url = st.text_input(
                "YouTube video URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            submitted = st.form_submit_button("Analyze video comments")

        source_key = video_url.strip()
        if submitted and video_url.strip():
            try:
                with st.spinner("Fetching all available comments..."):
                    comments = fetch_youtube_comments(video_url)
            except ValueError as error:
                st.error(str(error))
            except RuntimeError as error:
                st.error(f"Could not fetch YouTube comments: {error}")
            else:
                st.session_state["youtube_comments"] = (source_key, comments)
        stored_comments = st.session_state.get("youtube_comments")
        if comments is None and stored_comments and stored_comments[0] == source_key:
            comments = stored_comments[1]
    elif source == "Paste comments":
        pasted_text = st.text_area(
            "Paste comments (one comment per line)",
            height=240,
            placeholder="This video was helpful!\nCould you explain this in more detail?",
        )
        if pasted_text:
            comments = prepare_pasted_comments(pasted_text)
    else:
        upload = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if upload:
            try:
                frame = pd.read_csv(upload)
                if frame.empty or len(frame.columns) == 0:
                    st.error("The uploaded CSV contains no rows or columns.")
                else:
                    text_column = st.sidebar.selectbox(
                        "Comment text column",
                        options=frame.columns,
                        index=next(
                            (
                                index
                                for index, column in enumerate(frame.columns)
                                if column.lower()
                                in {"comment", "comments", "text", "content"}
                            ),
                            0,
                        ),
                    )
                    comments = prepare_comments(frame, text_column)
            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                st.error(f"Could not read the CSV file: {error}")

    if comments is None:
        st.info(
            "Enter a YouTube video URL, paste comments, or upload a CSV file."
        )
        return
    if comments.empty:
        st.warning("There are no non-empty comments to analyze.")
        return

    analyzed, terms, topics = analyze_comments(comments)
    positive = int((analyzed["sentiment"] == "Positive").sum())
    negative = int((analyzed["sentiment"] == "Negative").sum())
    neutral = int((analyzed["sentiment"] == "Neutral").sum())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Comments analyzed", len(analyzed))
    metric_columns[1].metric("Positive", positive)
    metric_columns[2].metric("Neutral", neutral)
    metric_columns[3].metric("Negative", negative)

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment")
        st.bar_chart(
            analyzed["sentiment"]
            .value_counts()
            .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        )
    with right:
        st.subheader("Common terms")
        if terms.empty:
            st.info("No meaningful terms found.")
        else:
            st.bar_chart(terms.set_index("term"))

    st.subheader("Discussion themes")
    if topics:
        st.dataframe(pd.DataFrame(topics), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough varied comment text to discover themes.")

    with st.expander("View analyzed comments"):
        st.dataframe(analyzed, use_container_width=True, hide_index=True)

    st.download_button(
        "Download analyzed comments as CSV",
        data=analyzed.to_csv(index=False).encode("utf-8"),
        file_name="youtube_comment_analysis.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        import sys

        from streamlit.web import cli

        sys.argv = ["streamlit", "run", __file__]
        raise SystemExit(cli.main())
    main()
