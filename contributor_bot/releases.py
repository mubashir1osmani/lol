"""Read only dated release articles and their New Contributors sections."""
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit

from .schedule import parse_time

RELEASE_INDEX = "https://docs.litellm.ai/release_notes"


def release_url(href: str) -> str | None:
    url = urlsplit(urljoin(RELEASE_INDEX, href))
    if (url.scheme != "https" or url.netloc != "docs.litellm.ai"
            or url.query or url.fragment
            or not re.fullmatch(r"/release_notes(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*){2,}/?", url.path)):
        return None
    return url.geturl().rstrip("/")


class IndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.entries = {}
        self.articles = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.articles.append({"date": None, "urls": set()})
        if not self.articles:
            return
        if tag == "time" and attrs.get("datetime"):
            self.articles[-1]["date"] = parse_time(attrs["datetime"])
        if tag == "a" and (url := release_url(attrs.get("href", ""))):
            self.articles[-1]["urls"].add(url)

    def handle_endtag(self, tag):
        if tag == "article" and self.articles:
            article = self.articles.pop()
            if article["date"]:
                for url in article["urls"]:
                    self.entries[url] = article["date"]


def parse_release_index(html: str) -> dict:
    parser = IndexParser()
    parser.feed(html)
    if not parser.entries:
        raise RuntimeError("Release index contains no dated articles; refusing a partial roundup")
    return parser.entries


class ContributorsParser(HTMLParser):
    def __init__(self, repository):
        super().__init__()
        self.repository = repository.lower()
        self.section = False
        self.level = 0
        self.heading = None
        self.heading_text = []
        self.numbers = set()
        self.saw_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if re.fullmatch(r"h[1-6]", tag):
            level = int(tag[1])
            if self.section and level <= self.level:
                self.section = False
            self.heading = level
            self.heading_text = []
            if attrs.get("id") == "new-contributors":
                self.section, self.level = True, level
            if level == 1:
                self.saw_title = True
        if tag == "a" and self.section:
            url = urlsplit(attrs.get("href", ""))
            match = re.fullmatch(r"/([^/]+/[^/]+)/pull/([1-9][0-9]*)/?", url.path)
            if (url.scheme == "https" and url.netloc == "github.com" and match
                    and match[1].lower() == self.repository):
                self.numbers.add(int(match[2]))

    def handle_data(self, data):
        if self.heading:
            self.heading_text.append(data)

    def handle_endtag(self, tag):
        if self.heading and tag == f"h{self.heading}":
            title = "".join(self.heading_text).replace("\u200b", "").strip().lower()
            if title == "new contributors":
                self.section, self.level = True, self.heading
            self.heading = None


def parse_new_contributors(html: str, repository: str) -> list[int]:
    parser = ContributorsParser(repository)
    parser.feed(html)
    if not parser.saw_title:
        raise RuntimeError("Release note is not a valid document")
    # Patch releases legitimately have no New Contributors section.
    return sorted(parser.numbers)
