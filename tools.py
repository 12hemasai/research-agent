import requests
from bs4 import BeautifulSoup


def web_search(query):
    """
    Search the web using DuckDuckGo HTML search.

    Returns up to 5 results containing:
    title, URL, and snippet.
    """

    url = "https://html.duckduckgo.com/html/"

    try:
        response = requests.post(
            url,
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        results = []

        for result in soup.select(".result")[:5]:

            title = result.select_one(".result__title")
            link = result.select_one(".result__a")
            snippet = result.select_one(".result__snippet")

            if title and link:

                results.append({
                    "title": title.get_text(" ", strip=True),
                    "url": link.get("href"),
                    "snippet": (
                        snippet.get_text(" ", strip=True)
                        if snippet
                        else "No snippet available."
                    )
                })

        return results

    except Exception as e:

        print(f"Search error: {e}")

        return []


def fetch_page(url):
    """
    Fetch and extract readable text from a webpage.
    """

    try:

        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary elements
        for element in soup(
            ["script", "style", "nav", "footer", "header"]
        ):
            element.decompose()

        text = soup.get_text(
            " ",
            strip=True
        )

        return text[:10000]

    except Exception as e:

        return f"Unable to fetch page: {e}"