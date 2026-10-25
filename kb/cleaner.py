"""
HTML parser and section extractor for MedDriftGuard.

Extracts structured sections (heading + text) from health documentation,
stripping web clutter such as navigation, footers, headers, scripts, and ads.
"""

from dataclasses import dataclass
from typing import List
from bs4 import BeautifulSoup, Tag


@dataclass
class SectionContent:
    """Represents a clean section extracted from a medical document."""
    heading: str
    text: str


class DocumentCleaner:
    """Cleans and extracts sections from raw HTML health articles."""

    UNWANTED_TAGS = [
        "script", "style", "nav", "header", "footer", "aside", "form",
        "noscript", "iframe", "svg", "button", "input", "select", "option",
    ]

    UNWANTED_CLASSES_OR_IDS = [
        "ad", "advertisement", "cookie", "banner", "sidebar", "nav", "navigation",
        "social-share", "share-links", "comments", "footer", "disclaimer-footer",
        "modal", "popup", "newsletter", "breadcrumb",
    ]

    @classmethod
    def clean_html(cls, html_content: str, default_heading: str = "Overview") -> List[SectionContent]:
        """
        Parses raw HTML, removes non-clinical clutter, and returns clean sections
        divided by HTML headings (h1, h2, h3, h4).
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Decompose completely unwanted tags
        for tag_name in cls.UNWANTED_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 2. Decompose elements matching ad/navigation class or id keywords
        for el in soup.find_all(attrs={"class": True}):
            classes = " ".join(el.get("class", [])).lower()
            if any(keyword in classes for keyword in cls.UNWANTED_CLASSES_OR_IDS):
                el.decompose()

        for el in soup.find_all(attrs={"id": True}):
            el_id = str(el.get("id", "")).lower()
            if any(keyword in el_id for keyword in cls.UNWANTED_CLASSES_OR_IDS):
                el.decompose()

        # 3. Extract sections based on heading tags
        sections: List[SectionContent] = []
        current_heading = default_heading
        current_paragraphs: List[str] = []

        # Find main content container if available, otherwise search body
        content_container = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", attrs={"role": "main"})
            or soup.find("body")
            or soup
        )

        for elem in content_container.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            tag_name = elem.name.lower()

            if tag_name in ["h1", "h2", "h3", "h4"]:
                # Save previous section if it has content
                clean_text = " ".join(" ".join(current_paragraphs).split())
                if clean_text:
                    sections.append(SectionContent(heading=current_heading, text=clean_text))
                current_heading = elem.get_text(separator=" ", strip=True) or default_heading
                current_paragraphs = []
            else:
                text = elem.get_text(separator=" ", strip=True)
                if text and len(text) > 10:
                    current_paragraphs.append(text)

        # Append final section
        clean_text = " ".join(" ".join(current_paragraphs).split())
        if clean_text:
            sections.append(SectionContent(heading=current_heading, text=clean_text))

        return sections
