import os
import re
import json
import html
import time
import heapq
import urllib.parse
import urllib.request
from collections import defaultdict

import config


# ============================================================
# WIKIPEDIA GRAPH COLLECTOR v3
#
# RU + EN
# Graph traversal
# Persistent state
# Priority frontier
# Language balancing
# Redirect handling
# API continuation
# Legacy corpus compatibility
#
# This file is self-contained.
# No additional functions need to be added manually.
# ============================================================


# ============================================================
# BASIC CONFIGURATION
# ============================================================

WIKIPEDIA_APIS = {
    "ru": "https://ru.wikipedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}

WIKIPEDIA_BASE_URLS = {
    "ru": "https://ru.wikipedia.org/wiki/",
    "en": "https://en.wikipedia.org/wiki/",
}

LANGUAGES = ("ru", "en")


# ------------------------------------------------------------
# Number of NEW articles collected per run.
# Comes from config.py.
# ------------------------------------------------------------

ARTICLES_PER_COLLECTION = int(
    getattr(
        config,
        "WIKIPEDIA_ARTICLES_PER_COLLECTION",
        5,
    )
)


# ------------------------------------------------------------
# Language balance.
#
# 5 articles:
#     approximately 3 RU + 2 EN
#
# 10 articles:
#     approximately 6 RU + 4 EN
# ------------------------------------------------------------

LANGUAGE_BALANCE = {
    "ru": 0.60,
    "en": 0.40,
}


# ------------------------------------------------------------
# Network
# ------------------------------------------------------------

REQUEST_TIMEOUT = 30

USER_AGENT = (
    "PersonalResearchLLM/3.0 "
    "(educational local research project)"
)


# ------------------------------------------------------------
# Graph parameters
# ------------------------------------------------------------

MAX_LINKS_PER_ARTICLE = 1000

MAX_TITLE_LENGTH = 300

# Maximum depth is intentionally generous.
#
# 0 = seed
# 1 = direct link from seed
# 2 = link from level 1
# etc.
#
# None means unlimited.
MAX_GRAPH_DEPTH = None


# ------------------------------------------------------------
# Priority parameters
# ------------------------------------------------------------

# New links receive a base priority.

BASE_PRIORITY = 1.0

# Repeated incoming references increase priority.

INBOUND_LINK_WEIGHT = 2.0

# Shallower nodes are slightly preferred.

DEPTH_WEIGHT = 0.25

# Prefer titles that look more like normal knowledge articles
# and less like list/index pages.

TITLE_QUALITY_WEIGHT = 0.5


# ============================================================
# PATHS
# ============================================================

WIKI_DIR = config.WIKI_DIR

RU_DIR = os.path.join(
    WIKI_DIR,
    "ru",
)

EN_DIR = os.path.join(
    WIKI_DIR,
    "en",
)

STATE_PATH = os.path.join(
    WIKI_DIR,
    "_knowledge_graph_v3.json",
)


os.makedirs(
    WIKI_DIR,
    exist_ok=True,
)

os.makedirs(
    RU_DIR,
    exist_ok=True,
)

os.makedirs(
    EN_DIR,
    exist_ok=True,
)


# ============================================================
# STARTING SEEDS
# ============================================================

# IMPORTANT:
#
# RU uses real Russian Wikipedia titles.
#
# We do NOT use:
#
#     "Physics"
#
# as a Russian seed.
#
# We use:
#
#     "Физика"
#

DEFAULT_RU_SEEDS = [
    "Физика",
    "Биология",
    "Химия",
    "Математика",
    "Информатика",
    "История",
    "Философия",
    "Психология",
    "Искусственный интеллект",
    "Нейронаука",
]


# English seeds come from config.py if present.

DEFAULT_EN_SEEDS = list(
    getattr(
        config,
        "WIKIPEDIA_TOPICS",
        [
            "Physics",
            "Biology",
            "Chemistry",
            "Mathematics",
            "Computer science",
            "History",
            "Philosophy",
            "Psychology",
            "Artificial intelligence",
            "Neuroscience",
        ],
    )
)


# ============================================================
# ARTICLE SECTIONS THAT SHOULD NOT ENTER THE CORPUS
# ============================================================

STOP_SECTIONS = {
    "references",
    "external links",
    "see also",
    "notes",
    "citations",
    "sources",
    "further reading",
    "bibliography",

    "примечания",
    "ссылки",
    "см. также",
    "литература",
    "источники",
}


# ============================================================
# STATE CREATION
# ============================================================

def create_empty_state():
    return {
        "version": 3,

        "created_at": time.time(),

        "updated_at": time.time(),

        "initialized": False,

        # ----------------------------------------------------
        # Frontier
        #
        # List of dictionaries:
        #
        # {
        #     "title": "...",
        #     "depth": 0,
        #     "priority": 10.5
        # }
        # ----------------------------------------------------

        "frontier": {
            "ru": [],
            "en": [],
        },

        # ----------------------------------------------------
        # Titles that were ever put into the queue.
        # ----------------------------------------------------

        "queued": {
            "ru": [],
            "en": [],
        },

        # ----------------------------------------------------
        # Articles/pages that have already been processed.
        # ----------------------------------------------------

        "visited": {
            "ru": [],
            "en": [],
        },

        # ----------------------------------------------------
        # Articles actually saved to corpus.
        # ----------------------------------------------------

        "downloaded": {
            "ru": [],
            "en": [],
        },

        # ----------------------------------------------------
        # Depth in graph.
        # ----------------------------------------------------

        "depth": {
            "ru": {},
            "en": {},
        },

        # ----------------------------------------------------
        # Number of incoming graph references.
        #
        # Example:
        #
        # Physics -> Mechanics
        # Thermodynamics -> Mechanics
        #
        # Mechanics gets inbound_count = 2.
        # ----------------------------------------------------

        "inbound_count": {
            "ru": {},
            "en": {},
        },

        # ----------------------------------------------------
        # Actual graph.
        #
        # Source title:
        #
        # {
        #     "Physics": [
        #         "Mechanics",
        #         "Thermodynamics"
        #     ]
        # }
        # ----------------------------------------------------

        "edges": {
            "ru": {},
            "en": {},
        },

        # ----------------------------------------------------
        # Statistics.
        # ----------------------------------------------------

        "stats": {
            "collections": 0,

            "articles_processed": {
                "ru": 0,
                "en": 0,
            },

            "articles_downloaded": {
                "ru": 0,
                "en": 0,
            },

            "api_errors": 0,

            "missing_articles": 0,

            "redirects": 0,

            "links_discovered": {
                "ru": 0,
                "en": 0,
            },

            "frontier_added": {
                "ru": 0,
                "en": 0,
            },

            "last_collection_time": None,
        },
    }


# ============================================================
# STATE LOADING
# ============================================================

def load_state():
    if not os.path.exists(
        STATE_PATH
    ):
        return create_empty_state()

    try:
        with open(
            STATE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            state = json.load(f)

    except Exception as e:

        print(
            "Could not load graph state:",
            e,
        )

        print(
            "Creating new state.",
        )

        return create_empty_state()


    # --------------------------------------------------------
    # Defensive migration.
    #
    # If a field was not present in an older state,
    # create it.
    # --------------------------------------------------------

    default = create_empty_state()

    for key, value in default.items():

        if key not in state:
            state[key] = value


    for language in LANGUAGES:

        state["frontier"].setdefault(
            language,
            [],
        )

        state["queued"].setdefault(
            language,
            [],
        )

        state["visited"].setdefault(
            language,
            [],
        )

        state["downloaded"].setdefault(
            language,
            [],
        )

        state["depth"].setdefault(
            language,
            {},
        )

        state["inbound_count"].setdefault(
            language,
            {},
        )

        state["edges"].setdefault(
            language,
            {},
        )


    # --------------------------------------------------------
    # Ensure statistics fields exist.
    # --------------------------------------------------------

    stats = state["stats"]

    stats.setdefault(
        "collections",
        0,
    )

    stats.setdefault(
        "articles_processed",
        {},
    )

    stats.setdefault(
        "articles_downloaded",
        {},
    )

    stats.setdefault(
        "api_errors",
        0,
    )

    stats.setdefault(
        "missing_articles",
        0,
    )

    stats.setdefault(
        "redirects",
        0,
    )

    stats.setdefault(
        "links_discovered",
        {},
    )

    stats.setdefault(
        "frontier_added",
        {},
    )

    for language in LANGUAGES:

        stats["articles_processed"].setdefault(
            language,
            0,
        )

        stats["articles_downloaded"].setdefault(
            language,
            0,
        )

        stats["links_discovered"].setdefault(
            language,
            0,
        )

        stats["frontier_added"].setdefault(
            language,
            0,
        )


    return state


# ============================================================
# STATE SAVING
# ============================================================

def save_state(state):
    state["updated_at"] = time.time()

    temporary_path = (
        STATE_PATH
        + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temporary_path,
        STATE_PATH,
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    # Decode HTML entities:
    #
    # &nbsp;
    # &amp;
    # &#39;
    # etc.
    text = html.unescape(
        text
    )

    # Remove numeric citation markers:
    #
    # [1]
    # [23]
    # [123]
    #
    text = re.sub(
        r"\[\s*\d+\s*\]",
        "",
        text,
    )

    # --------------------------------------------------------
    # Remove sections after:
    #
    # References
    # External links
    # See also
    # etc.
    # --------------------------------------------------------

    heading_pattern = re.compile(
        r"(?im)^==+\s*(.+?)\s*==+\s*$"
    )

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        match = heading_pattern.match(
            line.strip()
        )

        if match:

            heading = (
                match.group(1)
                .strip()
                .lower()
            )

            if heading in STOP_SECTIONS:
                break

        cleaned_lines.append(
            line
        )


    text = "\n".join(
        cleaned_lines
    )


    # Remove remaining HTML.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )


    # Normalize spaces.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )


    # Normalize newlines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )


    return text.strip()


# ============================================================
# TITLE NORMALIZATION
# ============================================================

def normalize_title(title):
    if not title:
        return ""

    title = html.unescape(
        title
    )

    title = title.replace(
        "_",
        " ",
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


# ============================================================
# TITLE FILTERING
# ============================================================

def valid_title(title):
    if not title:
        return False

    title = title.strip()

    if not title:
        return False

    if len(title) > MAX_TITLE_LENGTH:
        return False

    # Namespace pages should never enter
    # the knowledge graph.
    forbidden_prefixes = (
        "File:",
        "Category:",
        "Template:",
        "Wikipedia:",
        "Help:",
        "Portal:",
        "Special:",
        "Talk:",
        "Module:",

        "Файл:",
        "Категория:",
        "Шаблон:",
        "Википедия:",
        "Справка:",
        "Портал:",
        "Служебная:",
        "Обсуждение:",
        "Модуль:",
    )

    lowered = title.lower()

    for prefix in forbidden_prefixes:

        if lowered.startswith(
            prefix.lower()
        ):
            return False

    return True


# ============================================================
# TITLE QUALITY
# ============================================================

def title_quality(title):
    """
    Gives a small heuristic score.

    We don't want to reject pages aggressively.
    This is only a priority signal.

    Normal knowledge pages:
        "Quantum mechanics"

    Less useful pages:
        "List of..."
        "Outline of..."
        "Index of..."
    """

    score = 1.0

    lowered = title.lower()

    bad_patterns = [
        "list of ",
        "outline of ",
        "index of ",
        "glossary of ",
        "timeline of ",
        "lists of ",
        "список ",
        "списки ",
        "указатель ",
        "хронология ",
    ]

    for pattern in bad_patterns:

        if lowered.startswith(
            pattern
        ):
            score -= 0.6


    # Very short titles can be ambiguous.
    if len(title) <= 2:
        score -= 0.3


    # Long titles are not automatically bad,
    # but extremely long ones are less useful.
    if len(title) > 150:
        score -= 0.2


    return max(
        0.1,
        score,
    )


# ============================================================
# FILE NAME
# ============================================================

def safe_filename(title):
    title = title.strip()

    title = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        title,
    )

    title = re.sub(
        r"\s+",
        "_",
        title,
    )

    title = title.strip(
        " ._"
    )

    if not title:
        title = "article"

    # Windows path protection.
    return title[:180]


# ============================================================
# ARTICLE PATH
# ============================================================

def article_path(
    language,
    title,
):
    directory = (
        RU_DIR
        if language == "ru"
        else EN_DIR
    )

    return os.path.join(
        directory,
        safe_filename(title)
        + ".txt",
    )


# ============================================================
# LEGACY ENGLISH PATH
# ============================================================

def legacy_english_article_path(
    title,
):
    return os.path.join(
        WIKI_DIR,
        safe_filename(title)
        + ".txt",
    )


# ============================================================
# ARTICLE EXISTENCE
# ============================================================

def article_already_exists(
    language,
    title,
):
    # New v3 location.
    if os.path.exists(
        article_path(
            language,
            title,
        )
    ):
        return True

    # Old collector location.
    if language == "en":

        if os.path.exists(
            legacy_english_article_path(
                title,
            )
        ):
            return True

    return False


# ============================================================
# HTTP REQUEST
# ============================================================

def api_request(
    language,
    params,
):
    api_url = WIKIPEDIA_APIS[
        language
    ]

    query = dict(
        params
    )

    query["format"] = "json"

    query["formatversion"] = "2"

    query["redirects"] = "1"

    url = (
        api_url
        + "?"
        + urllib.parse.urlencode(
            query
        )
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT,
    ) as response:

        raw = response.read()


    return json.loads(
        raw.decode(
            "utf-8"
        )
    )


# ============================================================
# FETCH ARTICLE
# ============================================================

def fetch_article(
    language,
    title,
):
    """
    Fetch one Wikipedia article.

    Returns:

        {
            title,
            text,
            links,
            url
        }

    Handles API continuation so that pages with
    more than the first batch of links are not
    silently truncated.
    """

    all_links = []

    continuation = None

    real_title = title

    article_text = ""

    redirect_seen = False


    while True:

        params = {
            "action": "query",

            "prop": "extracts|links",

            "explaintext": "1",

            "titles": title,

            "plnamespace": "0",

            "pllimit": "max",

        }


        if continuation:

            params.update(
                continuation
            )


        data = api_request(
            language,
            params,
        )


        pages = (
            data
            .get(
                "query",
                {}
            )
            .get(
                "pages",
                []
            )
        )


        if not pages:
            return None


        page = pages[0]


        if page.get(
            "missing"
        ):
            return None


        page_title = page.get(
            "title",
            title,
        )


        if page_title != title:
            redirect_seen = True


        real_title = page_title


        # Text is normally present on the
        # first response.
        if page.get(
            "extract"
        ):

            article_text = page.get(
                "extract",
                "",
            )


        # ----------------------------------------------------
        # Links
        # ----------------------------------------------------

        for link in page.get(
            "links",
            []
        ):

            if link.get(
                "ns",
                0,
            ) != 0:
                continue


            linked_title = normalize_title(
                link.get(
                    "title",
                    "",
                )
            )


            if not valid_title(
                linked_title
            ):
                continue


            if (
                linked_title
                == real_title
            ):
                continue


            all_links.append(
                linked_title
            )


        # ----------------------------------------------------
        # API continuation.
        # ----------------------------------------------------

        continuation_data = data.get(
            "continue"
        )


        if not continuation_data:
            break


        # We need to preserve all continuation
        # parameters except the generic "continue".
        continuation = {
            key: value
            for key, value
            in continuation_data.items()
            if key != "continue"
        }


        # Safety guard.
        if not continuation:
            break


    # Deduplicate.
    all_links = list(
        dict.fromkeys(
            all_links
        )
    )


    if len(all_links) > MAX_LINKS_PER_ARTICLE:

        all_links = all_links[
            :MAX_LINKS_PER_ARTICLE
        ]


    return {
        "language": language,

        "title": real_title,

        "text": clean_text(
            article_text
        ),

        "links": all_links,

        "redirected": redirect_seen,

        "url": (
            WIKIPEDIA_BASE_URLS[
                language
            ]
            + urllib.parse.quote(
                real_title.replace(
                    " ",
                    "_",
                )
            )
        ),
    }


# ============================================================
# SAVE ARTICLE
# ============================================================

def save_article(
    article,
):
    language = article[
        "language"
    ]

    title = article[
        "title"
    ]

    text = article[
        "text"
    ]


    if not text:
        return False


    if article_already_exists(
        language,
        title,
    ):
        return False


    path = article_path(
        language,
        title,
    )


    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            text
        )

        f.write(
            "\n\n"
        )

        f.write(
            "Wikipedia: "
        )

        f.write(
            article["url"]
        )

    return True


# ============================================================
# NODE KEY
# ============================================================

def node_key(
    language,
    title,
):
    return (
        language
        + "::"
        + title
    )


# ============================================================
# INBOUND COUNT
# ============================================================

def get_inbound_count(
    state,
    language,
    title,
):
    return int(
        state[
            "inbound_count"
        ][language].get(
            title,
            0,
        )
    )


# ============================================================
# PRIORITY CALCULATION
# ============================================================

def calculate_priority(
    state,
    language,
    title,
    depth,
):
    """
    Higher usefulness => higher priority.

    Factors:

    1. Base priority.
    2. Number of incoming references.
    3. Shallower depth.
    4. Title quality.

    This gives us something more intelligent than:
        "take links in random order".
    """

    inbound = get_inbound_count(
        state,
        language,
        title,
    )


    quality = title_quality(
        title
    )


    priority = (
        BASE_PRIORITY
        + inbound * INBOUND_LINK_WEIGHT
        + quality * TITLE_QUALITY_WEIGHT
        - depth * DEPTH_WEIGHT
    )


    return priority


# ============================================================
# FRONTIER LOOKUP
# ============================================================

def frontier_contains(
    state,
    language,
    title,
):
    for item in state[
        "frontier"
    ][language]:

        if item.get(
            "title"
        ) == title:

            return True

    return False


# ============================================================
# ADD NODE TO FRONTIER
# ============================================================

def add_to_frontier(
    state,
    language,
    title,
    depth,
):
    if language not in LANGUAGES:
        return False


    title = normalize_title(
        title
    )


    if not valid_title(
        title
    ):
        return False


    if (
        MAX_GRAPH_DEPTH is not None
        and depth > MAX_GRAPH_DEPTH
    ):
        return False


    visited = set(
        state[
            "visited"
        ][language]
    )


    if title in visited:
        return False


    # --------------------------------------------------------
    # Increment inbound reference count.
    #
    # Even if the node is already queued,
    # another article linking to it increases
    # its importance.
    # --------------------------------------------------------

    inbound = (
        state[
            "inbound_count"
        ][language].get(
            title,
            0,
        )
    )

    inbound += 1

    state[
        "inbound_count"
    ][language][title] = inbound


    # --------------------------------------------------------
    # Preserve shortest known depth.
    # --------------------------------------------------------

    previous_depth = state[
        "depth"
    ][language].get(
        title
    )


    if (
        previous_depth is None
        or depth < previous_depth
    ):

        state[
            "depth"
        ][language][title] = depth

    else:

        depth = previous_depth


    # --------------------------------------------------------
    # Already queued?
    #
    # We don't create duplicate frontier nodes.
    # But we DO update its priority.
    # --------------------------------------------------------

    if frontier_contains(
        state,
        language,
        title,
    ):

        for item in state[
            "frontier"
        ][language]:

            if item.get(
                "title"
            ) == title:

                item[
                    "priority"
                ] = calculate_priority(
                    state,
                    language,
                    title,
                    depth,
                )

        return False


    # --------------------------------------------------------
    # First time queued.
    # --------------------------------------------------------

    if title not in state[
        "queued"
    ][language]:

        state[
            "queued"
        ][language].append(
            title
        )


    priority = calculate_priority(
        state,
        language,
        title,
        depth,
    )


    state[
        "frontier"
    ][language].append(
        {
            "title": title,
            "depth": depth,
            "priority": priority,
        }
    )


    state[
        "stats"
    ][
        "frontier_added"
    ][language] += 1


    return True


# ============================================================
# FRONTIER SORT
# ============================================================

def sort_frontier(
    state,
    language,
):
    """
    Python's list is enough here.

    We sort descending by priority.
    """

    state[
        "frontier"
    ][language].sort(
        key=lambda item: (
            -float(
                item.get(
                    "priority",
                    0,
                )
            ),

            int(
                item.get(
                    "depth",
                    0,
                )
            ),

            item.get(
                "title",
                "",
            ).lower(),
        )
    )


# ============================================================
# POP NEXT NODE
# ============================================================

def pop_next(
    state,
    language,
):
    frontier = state[
        "frontier"
    ][language]


    if not frontier:
        return None


    sort_frontier(
        state,
        language,
    )


    return frontier.pop(
        0
    )


# ============================================================
# INITIALIZE GRAPH
# ============================================================

def initialize_graph(
    state,
):
    if state.get(
        "initialized"
    ):
        return


    print()
    print(
        "Initializing RU graph..."
    )


    for title in DEFAULT_RU_SEEDS:

        add_to_frontier(
            state=state,
            language="ru",
            title=title,
            depth=0,
        )


    print(
        "Initializing EN graph..."
    )


    for title in DEFAULT_EN_SEEDS:

        add_to_frontier(
            state=state,
            language="en",
            title=title,
            depth=0,
        )


    state[
        "initialized"
    ] = True


    save_state(
        state
    )


# ============================================================
# LANGUAGE FRONTIER SIZE
# ============================================================

def frontier_size(
    state,
    language,
):
    return len(
        state[
            "frontier"
        ][language]
    )


# ============================================================
# CHOOSE LANGUAGE
# ============================================================

def choose_language(
    state,
    collected,
):
    """
    Maintains approximate 60/40 balance.

    Example with 5 articles:

        RU RU RU EN EN

    If one language's frontier is empty,
    the other language is allowed to continue.
    """

    ru_available = (
        frontier_size(
            state,
            "ru",
        )
        > 0
    )

    en_available = (
        frontier_size(
            state,
            "en",
        )
        > 0
    )


    if not ru_available and not en_available:
        return None


    if not ru_available:
        return "en"


    if not en_available:
        return "ru"


    ru_count = collected[
        "ru"
    ]

    en_count = collected[
        "en"
    ]


    total = (
        ru_count
        + en_count
    )


    if total == 0:
        return "ru"


    current_ru_ratio = (
        ru_count
        / total
    )


    current_en_ratio = (
        en_count
        / total
    )


    target_ru = LANGUAGE_BALANCE[
        "ru"
    ]

    target_en = LANGUAGE_BALANCE[
        "en"
    ]


    # Language furthest below its target
    # gets selected.
    ru_deficit = (
        target_ru
        - current_ru_ratio
    )

    en_deficit = (
        target_en
        - current_en_ratio
    )


    if ru_deficit > en_deficit:
        return "ru"


    if en_deficit > ru_deficit:
        return "en"


    # Equal:
    # choose language with smaller frontier
    # pressure.
    if ru_count <= en_count:
        return "ru"


    return "en"


# ============================================================
# PROCESS ONE ARTICLE
# ============================================================

def process_article(
    state,
    language,
    title,
    depth,
):
    print()
    print(
        "-" * 70
    )

    print(
        f"[{language.upper()}]"
    )

    print(
        "Depth:",
        depth,
    )

    print(
        "Article:",
        title,
    )


    # --------------------------------------------------------
    # Fetch
    # --------------------------------------------------------

    try:

        article = fetch_article(
            language,
            title,
        )

    except Exception as e:

        print(
            "API ERROR:",
            e,
        )

        state[
            "stats"
        ][
            "api_errors"
        ] += 1


        # Mark as visited so a broken page
        # doesn't remain permanently at the
        # front of the queue.
        if title not in state[
            "visited"
        ][language]:

            state[
                "visited"
            ][language].append(
                title
            )


        return {
            "processed": False,
            "downloaded": False,
            "links": 0,
        }


    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    if article is None:

        print(
            "Article not found."
        )

        state[
            "stats"
        ][
            "missing_articles"
        ] += 1


        if title not in state[
            "visited"
        ][language]:

            state[
                "visited"
            ][language].append(
                title
            )


        return {
            "processed": False,
            "downloaded": False,
            "links": 0,
        }


    real_title = article[
        "title"
    ]


    # --------------------------------------------------------
    # Redirect
    # --------------------------------------------------------

    if article.get(
        "redirected"
    ):

        state[
            "stats"
        ][
            "redirects"
        ] += 1

        print(
            "Redirected to:",
            real_title,
        )


    # --------------------------------------------------------
    # Mark visited.
    # --------------------------------------------------------

    if title not in state[
        "visited"
    ][language]:

        state[
            "visited"
        ][language].append(
            title
        )


    if real_title not in state[
        "visited"
    ][language]:

        state[
            "visited"
        ][language].append(
            real_title
        )


    state[
        "stats"
    ][
        "articles_processed"
    ][language] += 1


    # --------------------------------------------------------
    # Save article.
    # --------------------------------------------------------

    downloaded = save_article(
        article
    )


    if downloaded:

        if real_title not in state[
            "downloaded"
        ][language]:

            state[
                "downloaded"
            ][language].append(
                real_title
            )


        state[
            "stats"
        ][
            "articles_downloaded"
        ][language] += 1


        print(
            "SAVED:",
            real_title,
        )

    else:

        print(
            "Already exists or empty:",
            real_title,
        )


    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    links = article[
        "links"
    ]


    state[
        "stats"
    ][
        "links_discovered"
    ][language] += len(
        links
    )


    source_title = real_title


    # Save the edge list.
    state[
        "edges"
    ][language][
        source_title
    ] = []


    next_depth = (
        depth
        + 1
    )


    new_nodes = 0


    for linked_title in links:

        if linked_title == source_title:
            continue


        # Save edge.
        state[
            "edges"
        ][language][
            source_title
        ].append(
            linked_title
        )


        # Add new node to frontier.
        added = add_to_frontier(
            state=state,
            language=language,
            title=linked_title,
            depth=next_depth,
        )


        if added:
            new_nodes += 1


    # Remove duplicate edges.
    state[
        "edges"
    ][language][
        source_title
    ] = list(
        dict.fromkeys(
            state[
                "edges"
            ][language][
                source_title
            ]
        )
    )


    print(
        "Links found:",
        len(links),
    )

    print(
        "New frontier nodes:",
        new_nodes,
    )


    print(
        "Current frontier:",
        frontier_size(
            state,
            language,
        ),
    )


    return {
        "processed": True,
        "downloaded": downloaded,
        "links": len(links),
    }


# ============================================================
# STATUS
# ============================================================

def print_status(
    state,
):
    print()
    print(
        "=" * 70
    )

    print(
        "WIKIPEDIA GRAPH STATUS v3"
    )

    print(
        "=" * 70
    )


    print(
        "State:",
        STATE_PATH,
    )

    print()


    for language in LANGUAGES:

        print(
            language.upper()
            + ":"
        )


        print(
            "  Frontier:",
            len(
                state[
                    "frontier"
                ][language]
            ),
        )


        print(
            "  Queued:",
            len(
                state[
                    "queued"
                ][language]
            ),
        )


        print(
            "  Visited:",
            len(
                state[
                    "visited"
                ][language]
            ),
        )


        print(
            "  Downloaded:",
            len(
                state[
                    "downloaded"
                ][language]
            ),
        )


        print(
            "  Graph nodes:",
            len(
                state[
                    "edges"
                ][language]
            ),
        )


        print(
            "  Graph links:",
            sum(
                len(
                    links
                )
                for links
                in state[
                    "edges"
                ][language].values()
            ),
        )


        print(
            "  Processed:",
            state[
                "stats"
            ][
                "articles_processed"
            ][language],
        )


        print(
            "  New frontier nodes:",
            state[
                "stats"
            ][
                "frontier_added"
            ][language],
        )


        print()


    print(
        "Total API errors:",
        state[
            "stats"
        ][
            "api_errors"
        ],
    )


    print(
        "Total missing articles:",
        state[
            "stats"
        ][
            "missing_articles"
        ],
    )


    print(
        "Total redirects:",
        state[
            "stats"
        ][
            "redirects"
        ],
    )


# ============================================================
# FRONTIER PREVIEW
# ============================================================

def print_frontier_preview(
    state,
    language,
    limit=10,
):
    print()
    print(
        f"TOP {language.upper()} FRONTIER"
    )

    print(
        "-" * 70
    )


    items = list(
        state[
            "frontier"
        ][language]
    )


    items.sort(
        key=lambda item: (
            -float(
                item.get(
                    "priority",
                    0,
                )
            ),
            int(
                item.get(
                    "depth",
                    0,
                )
            ),
        )
    )


    for index, item in enumerate(
        items[:limit],
        start=1,
    ):

        print(
            f"{index:02d}. "
            f"P={item.get('priority', 0):.2f} "
            f"D={item.get('depth', 0)} "
            f"{item.get('title', '')}"
        )


# ============================================================
# COLLECTION
# ============================================================

def collect():
    print()
    print(
        "=" * 70
    )

    print(
        "WIKIPEDIA GRAPH COLLECTOR v3"
    )

    print(
        "RU + EN"
    )

    print(
        "Graph + Priority + Persistent State"
    )

    print(
        "Language balance: RU 60% / EN 40%"
    )

    print(
        "=" * 70
    )


    print(
        "Articles per collection:",
        ARTICLES_PER_COLLECTION,
    )


    # --------------------------------------------------------
    # Load state.
    # --------------------------------------------------------

    state = load_state()


    # --------------------------------------------------------
    # Initialize seeds only once.
    # --------------------------------------------------------

    initialize_graph(
        state
    )


    print_status(
        state
    )


    print_frontier_preview(
        state,
        "ru",
    )


    print_frontier_preview(
        state,
        "en",
    )


    # --------------------------------------------------------
    # Per-run statistics.
    # --------------------------------------------------------

    collected = {
        "ru": 0,
        "en": 0,
    }


    processed_this_run = 0

    failed_this_run = 0


    # --------------------------------------------------------
    # Main loop.
    # --------------------------------------------------------

    while (
        collected["ru"]
        + collected["en"]
        < ARTICLES_PER_COLLECTION
    ):


        language = choose_language(
            state,
            collected,
        )


        if language is None:

            print()
            print(
                "Both frontiers are empty."
            )

            print(
                "The graph has no more queued nodes."
            )

            break


        node = pop_next(
            state,
            language,
        )


        if node is None:
            continue


        title = node.get(
            "title",
            "",
        )


        depth = int(
            node.get(
                "depth",
                0,
            )
        )


        # ----------------------------------------------------
        # Process.
        # ----------------------------------------------------

        result = process_article(
            state=state,
            language=language,
            title=title,
            depth=depth,
        )


        processed_this_run += 1


        if result[
            "downloaded"
        ]:

            collected[
                language
            ] += 1

        elif not result[
            "processed"
        ]:

            failed_this_run += 1


        # ----------------------------------------------------
        # Recalculate priorities after new
        # inbound links were discovered.
        # ----------------------------------------------------

        sort_frontier(
            state,
            language,
        )


        # ----------------------------------------------------
        # SAVE IMMEDIATELY.
        #
        # This is deliberate.
        #
        # If:
        #
        # - Python crashes
        # - Internet disappears
        # - laptop sleeps
        # - user presses Ctrl+C
        #
        # graph progress is preserved.
        # ----------------------------------------------------

        save_state(
            state
        )


        # ----------------------------------------------------
        # Safety guard.
        #
        # If Wikipedia has lots of unavailable pages,
        # don't loop forever trying to obtain N new files.
        # ----------------------------------------------------

        max_attempts = max(
            50,
            ARTICLES_PER_COLLECTION * 25,
        )


        if (
            processed_this_run
            >= max_attempts
            and (
                collected["ru"]
                + collected["en"]
                < ARTICLES_PER_COLLECTION
            )
        ):

            print()
            print(
                "Safety stop."
            )

            print(
                "Too many nodes processed "
                "without enough new articles."
            )

            break


    # ========================================================
    # FINISH
    # ========================================================

    state[
        "stats"
    ][
        "collections"
    ] += 1


    state[
        "stats"
    ][
        "last_collection_time"
    ] = time.time()


    save_state(
        state
    )


    print()
    print(
        "=" * 70
    )

    print(
        "COLLECTION COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        "New articles:"
    )

    print(
        "  RU:",
        collected["ru"],
    )

    print(
        "  EN:",
        collected["en"],
    )

    print(
        "  TOTAL:",
        collected["ru"]
        + collected["en"],
    )


    print()
    print(
        "Processed nodes:",
        processed_this_run,
    )


    print(
        "Failed/missing:",
        failed_this_run,
    )


    print()


    print_status(
        state
    )


    print_frontier_preview(
        state,
        "ru",
    )


    print_frontier_preview(
        state,
        "en",
    )


    print()
    print(
        "Graph state saved:"
    )

    print(
        STATE_PATH
    )


    return collected


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    collect()