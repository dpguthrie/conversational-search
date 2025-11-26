"""Citation quality scorer using deterministic metrics."""
import re
from typing import Dict, Any, List


class CitationQualityScorer:
    """Evaluate citation quality with multiple dimensions."""

    def extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from text."""
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]

    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple heuristic)."""
        # Remove source list if present
        text = re.split(r'\n\s*Sources?:', text)[0]
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def has_factual_content(self, sentence: str) -> bool:
        """Heuristic to detect if sentence contains factual claims."""
        # Skip questions, greetings, meta-statements
        if not sentence:
            return False
        if sentence.strip().endswith('?'):
            return False
        if any(sentence.lower().startswith(p) for p in ['however', 'therefore', 'thus', 'in conclusion']):
            return False
        # Must have reasonable length
        if len(sentence.split()) < 5:
            return False
        return True

    def score_coverage(self, answer: str) -> float:
        """Score citation coverage: % of factual sentences with citations.

        Args:
            answer: Answer text

        Returns:
            Coverage score 0-1
        """
        sentences = self.extract_sentences(answer)

        if not sentences:
            return 0.0

        # Count sentences with factual content
        factual_sentences = [s for s in sentences if self.has_factual_content(s)]

        if not factual_sentences:
            return 1.0  # No factual claims = no citations needed

        # Count how many have citations
        cited_sentences = [s for s in factual_sentences if re.search(r'\[\d+\]', s)]

        coverage = len(cited_sentences) / len(factual_sentences)
        return coverage

    def score_precision(self, answer: str, sources: List[Dict]) -> float:
        """Score citation precision: are citations valid and reasonable.

        Args:
            answer: Answer text
            sources: List of sources

        Returns:
            Precision score 0-1
        """
        citations = self.extract_citations(answer)

        if not citations:
            return 1.0  # No citations = no precision errors

        num_sources = len(sources)

        # Check all citations are in valid range
        invalid_citations = [c for c in citations if c < 1 or c > num_sources]

        if invalid_citations:
            return 0.0

        # All citations valid
        return 1.0

    def score_source_quality(self, sources: List[Dict]) -> float:
        """Score quality of sources (heuristic).

        Args:
            sources: List of sources with url, title, snippet

        Returns:
            Source quality score 0-1
        """
        if not sources:
            return 0.0

        quality_scores = []

        for source in sources:
            url = source.get("url", "").lower()
            score = 0.5  # Default

            # Higher quality domains (heuristic)
            high_quality_domains = [
                'edu', 'gov', 'wikipedia.org', 'nature.com',
                'sciencedirect.com', 'arxiv.org', 'ieee.org'
            ]
            if any(domain in url for domain in high_quality_domains):
                score = 0.9

            # Lower quality domains
            low_quality_domains = ['blog', 'forum', 'reddit.com']
            if any(domain in url for domain in low_quality_domains):
                score = 0.3

            quality_scores.append(score)

        return sum(quality_scores) / len(quality_scores)

    def score(self, output: Dict[str, Any], expected: Dict[str, Any] = None) -> Dict[str, Any]:
        """Score citation quality across dimensions.

        Args:
            output: Agent output with answer and sources
            expected: Expected output (not used)

        Returns:
            Score dict with composite citation quality score
        """
        answer = output.get("answer", "")
        sources = output.get("sources", [])

        coverage = self.score_coverage(answer)
        precision = self.score_precision(answer, sources)
        source_quality = self.score_source_quality(sources)

        # Weighted composite score
        composite_score = (
            0.4 * coverage +
            0.4 * precision +
            0.2 * source_quality
        )

        return {
            "name": "citation_quality",
            "score": composite_score,
            "metadata": {
                "coverage": coverage,
                "precision": precision,
                "source_quality": source_quality,
                "num_citations": len(self.extract_citations(answer)),
                "num_sources": len(sources)
            }
        }


# Braintrust-compatible scorer function
def citation_quality_scorer(output, expected=None):
    """Braintrust-compatible wrapper for citation quality scorer."""
    scorer = CitationQualityScorer()
    return scorer.score(output, expected)
