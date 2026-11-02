"""Checks of the wiki pages in wiki/ and of the links to them.

The folder wiki/ holds the pages of the GitHub wiki (https://github.com/Tag-E/h4lat/wiki), which
.github/workflows/wiki.yml publishes at every push to main. Nothing else reads these Markdown files before they are
published, so the tests below catch the usual ways in which documentation goes stale: links to pages or sections
that do not exist (any more), pages missing from the sidebar, names of h4lat that were renamed or removed, and math
delimiters left open (GitHub then shows the raw LaTeX instead of the formula).

Conventions checked here (see CONTRIBUTING.md, "Documentation"): wiki pages link to each other with full URLs,
https://github.com/Tag-E/h4lat/wiki/Page-Name#section, because GitHub documents only those for Markdown wikis;
inline math is written $`...`$ and display math in ```math blocks.
"""

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WIKI = ROOT / "wiki"
WIKI_URL = "https://github.com/Tag-E/h4lat/wiki"
REPO_URL = "https://github.com/Tag-E/h4lat"

# The wiki pages, by page name (the file name without .md). _Sidebar.md and _Footer.md are not pages: GitHub shows
# them next to and below every page.
PAGES = {path.stem: path for path in sorted(WIKI.glob("*.md")) if not path.name.startswith("_")}
# Every Markdown file whose links and h4lat names are checked: the wiki files (sidebar and footer included) and the
# README, which links to the wiki.
DOCUMENTS = sorted(WIKI.glob("*.md")) + [ROOT / "README.md"]

# A link to the wiki: the bare URL is the Home page; the page name and the #anchor are optional.
WIKI_LINK = re.compile(re.escape(WIKI_URL) + r"(?:/([\w\-]+))?(?:#([\w\-]+))?")
# A link to a section of the README, as shown on the GitHub page of the repository
README_LINK = re.compile(re.escape(REPO_URL) + r"#([\w\-]+)")
# A dotted name of the package, e.g. h4lat.cg_calculator.alpha_list. The look-behind skips URLs such as
# github.com/Tag-E/h4lat.git, where h4lat follows a "/".
H4LAT_NAME = re.compile(r"(?<![\w/.\-])h4lat((?:\.[A-Za-z_]\w*)+)")

pytestmark = pytest.mark.skipif(not WIKI.is_dir(), reason="the wiki/ folder is not available")


def _ids(paths):
    """Readable test ids: the file names (README.md, Home.md, …) instead of the full paths."""
    return [path.name for path in paths]


def _code_blocks(markdown: str) -> list[tuple[str, str]]:
    """The fenced code blocks of *markdown*, as (language, content) pairs, e.g. ("python", "import h4lat\\n")."""
    return re.findall(r"^[ \t]*```(\w*)[ \t]*\n(.*?)^[ \t]*```", markdown, re.DOTALL | re.MULTILINE)


def _prose(markdown: str) -> str:
    """*markdown* without its fenced code blocks, whose lines (e.g. Python comments "# …") are not headings."""
    return re.sub(r"^[ \t]*```\w*[ \t]*\n.*?^[ \t]*```", "", markdown, flags=re.DOTALL | re.MULTILINE)


def _anchors(markdown: str) -> set[str]:
    """The anchors that GitHub gives to the headings of a Markdown page.

    GitHub lowercases the text of a heading, drops every character that is not a letter, a digit, a space, "-" or
    "_", and replaces the spaces with "-": "## Tensor products and the Clebsch-Gordan series" becomes
    "tensor-products-and-the-clebsch-gordan-series". A repeated heading gets "-1", "-2", … appended.
    """
    anchors, seen = set(), {}
    for heading in re.findall(r"^#{1,6}[ \t]+(.+?)[ \t]*$", _prose(markdown), re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.replace("`", "").lower()).replace(" ", "-")
        anchors.add(slug if slug not in seen else f"{slug}-{seen[slug]}")
        seen[slug] = seen.get(slug, 0) + 1
    return anchors


def _resolves(dotted: str) -> bool:
    """Whether a dotted name such as "h4lat.cg_calculator.alpha_list" exists: attributes, or modules to import."""
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for i, part in enumerate(parts[1:], start=1):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        # A submodule that the package does not import itself, e.g. h4lat.cli
        try:
            obj = importlib.import_module(".".join(parts[: i + 1]))
        except ImportError:
            return False
    return True


def test_wiki_has_its_main_pages():
    """The wiki has a Home page (shown at the bare wiki URL), a sidebar and a footer."""
    assert "Home" in PAGES
    assert (WIKI / "_Sidebar.md").is_file()
    assert (WIKI / "_Footer.md").is_file()


@pytest.mark.parametrize("path", DOCUMENTS, ids=_ids(DOCUMENTS))
def test_links_to_the_wiki_exist(path):
    """Every link to the wiki points to an existing page and, if it has one, to an existing #section of it."""
    for page, anchor in WIKI_LINK.findall(_prose(path.read_text(encoding="utf-8"))):
        page = page or "Home"
        assert page in PAGES, f"{path.name}: link to the wiki page {page!r}, which does not exist"
        if anchor:
            anchors = _anchors(PAGES[page].read_text(encoding="utf-8"))
            assert anchor in anchors, f"{path.name}: link to {page}#{anchor}, but {page} has no such section"


@pytest.mark.parametrize("path", DOCUMENTS, ids=_ids(DOCUMENTS))
def test_links_to_readme_sections_exist(path):
    """Every link to a section of the README (https://github.com/Tag-E/h4lat#section) points to an existing one."""
    anchors = _anchors((ROOT / "README.md").read_text(encoding="utf-8"))
    for anchor in README_LINK.findall(_prose(path.read_text(encoding="utf-8"))):
        assert anchor in anchors, f"{path.name}: link to the README section #{anchor}, which does not exist"


@pytest.mark.parametrize("path", sorted(WIKI.glob("*.md")), ids=_ids(sorted(WIKI.glob("*.md"))))
def test_wiki_links_are_absolute(path):
    """Wiki pages link with full URLs: a relative link such as [text](Installation) is not resolved reliably."""
    for target in re.findall(r"\]\(([^)\s]+)\)", _prose(path.read_text(encoding="utf-8"))):
        assert target.startswith(("https://", "http://", "mailto:")), f"{path.name}: relative link {target!r}"


def test_sidebar_lists_every_page():
    """Every wiki page can be reached from the sidebar, shown next to every page."""
    sidebar = (WIKI / "_Sidebar.md").read_text(encoding="utf-8")
    linked = {page or "Home" for page, _ in WIKI_LINK.findall(sidebar)}
    assert set(PAGES) <= linked, f"pages missing from the sidebar: {sorted(set(PAGES) - linked)}"


@pytest.mark.parametrize("path", DOCUMENTS, ids=_ids(DOCUMENTS))
def test_h4lat_names_exist(path):
    """The names of h4lat in the text and in the Python examples exist, e.g. after a function has been renamed.

    Checked are the dotted names h4lat.… in inline code (`h4lat.cg_calculator`) and in Python code blocks, and the
    names imported by the "from h4lat… import …" lines of the Python code blocks.
    """
    markdown = path.read_text(encoding="utf-8")
    python = "\n".join(code for language, code in _code_blocks(markdown) if language == "python")
    inline_code = " ".join(re.findall(r"`([^`\n]+)`", _prose(markdown)))
    for suffix in H4LAT_NAME.findall(python + "\n" + inline_code):
        assert _resolves("h4lat" + suffix), f"{path.name}: h4lat{suffix} does not exist"
    for module, names in re.findall(r"^\s*from (h4lat(?:\.\w+)*) import ([^\n#]+)", python, re.MULTILINE):
        for name in re.split(r"[\s,()]+", names.strip()):
            if name:
                assert _resolves(f"{module}.{name}"), f"{path.name}: 'from {module} import {name}' fails"


def _formulas(markdown: str) -> list[tuple[int, str, bool]]:
    """The formulas of a page, as (line number, TeX, display) triples.

    Display formulas are the ```math blocks (the line number is that of the opening fence); in-line formulas are the
    $`…`$ spans outside the fenced code blocks.
    """
    formulas, fence, start, block = [], None, 0, []
    for number, line in enumerate(markdown.split("\n"), start=1):
        opening = re.match(r"^[ \t>]*```(\w*)[ \t]*$", line)
        if fence is None and opening:
            fence, start, block = opening.group(1), number, []
        elif fence is not None and re.match(r"^[ \t>]*```[ \t]*$", line):
            if fence == "math":
                formulas.append((start, "\n".join(block), True))
            fence = None
        elif fence is not None:
            block.append(line)
        else:
            formulas += [(number, tex, False) for tex in re.findall(r"\$`(.+?)`\$", line)]
    return formulas


def _is_one_group(tex: str) -> bool:
    """Whether *tex* is a single {…} group, from its first to its last character (\\{ and \\} are not braces)."""
    depth = 0
    for i, char in enumerate(tex):
        if char not in "{}" or (i > 0 and tex[i - 1] == "\\"):
            continue
        depth += 1 if char == "{" else -1
        if depth == 0 and i < len(tex) - 1:
            return False  # the first group closes before the end, e.g. {a} + {b}
    return tex.startswith("{") and depth == 0


# Macros that GitHub's math renderer refuses: it replaces the whole formula with "The following macros are not
# allowed: …". GitHub does not publish its list. \operatorname is confirmed (it broke a formula of this wiki: write
# \mathrm{Tr} instead of \operatorname{Tr}); the others are reported as refused or unsupported by other projects.
# \slashed is not a MathJax macro at all.
# (fmt: off keeps black from putting one macro per line)
# fmt: off
GITHUB_BLOCKED_MACROS = {
    "operatorname", "DeclareMathOperator", "newcommand", "renewcommand", "def", "href", "bm", "boldsymbol",
    "label", "ref", "eqref", "verb", "intertext", "mathds", "colorbox", "fcolorbox", "definecolor", "slashed",
    "require",
}
# fmt: on

WIKI_FILES = sorted(WIKI.glob("*.md"))


@pytest.mark.parametrize("path", WIKI_FILES, ids=_ids(WIKI_FILES))
def test_math_delimiters_are_balanced(path):
    """Every ```math block is closed, and every in-line formula $`…`$ opens and closes on the same source line.

    An unclosed delimiter does not raise any error on GitHub: the page just shows the raw LaTeX. A formula spread
    over two source lines may not be recognised as a formula at all.
    """
    markdown = path.read_text(encoding="utf-8")
    fences = re.findall(r"^[ \t>]*```", markdown, re.MULTILINE)
    assert len(fences) % 2 == 0, f"{path.name}: a fenced block (``` or ```math) is not closed"
    for line in _prose(markdown).split("\n"):
        assert line.count("$`") == line.count("`$"), f"{path.name}: $`…`$ split over two lines: {line[:60]!r}"


@pytest.mark.parametrize("path", WIKI_FILES, ids=_ids(WIKI_FILES))
def test_math_uses_no_macros_that_github_refuses(path):
    """No formula uses a macro that GitHub's math renderer refuses (see GITHUB_BLOCKED_MACROS)."""
    for number, tex, _ in _formulas(path.read_text(encoding="utf-8")):
        blocked = set(re.findall(r"\\([A-Za-z]+)", tex)) & GITHUB_BLOCKED_MACROS
        # (built outside the f-string: Python 3.11 allows no backslash inside the {…} of an f-string)
        macros = ", ".join("\\" + macro for macro in sorted(blocked))
        assert not blocked, f"{path.name}:{number}: GitHub refuses {macros} in {tex!r}"


@pytest.mark.parametrize("path", WIKI_FILES, ids=_ids(WIKI_FILES))
def test_inline_math_cannot_be_split(path):
    """Every in-line formula is one braced group, $`{…}`$, so that it is never split over two lines.

    GitHub renders math with MathJax 4, which lets the browser break an in-line formula at its top-level operators
    and spaces (=, +, ⊗, \\, …) when the line is full, e.g. in a narrow table cell. A braced group is a single atom
    for TeX, with no break point at the top level.
    """
    for number, tex, display in _formulas(path.read_text(encoding="utf-8")):
        if not display:
            assert _is_one_group(tex), f"{path.name}:{number}: write $`{{{tex}}}`$ instead of $`{tex}`$"


@pytest.mark.parametrize("path", WIKI_FILES, ids=_ids(WIKI_FILES))
def test_math_is_placed_where_github_renders_it(path):
    """Formulas are not in places where GitHub shows the raw LaTeX or breaks the layout.

    - A ```math block inside a list item (an indented fence) or a quote can be shown as a code block: display
      equations stand outside lists and quotes.
    - A formula inside bold text (**…$`x`$…**) may not be recognised.
    - A "|" inside a formula in a table row splits the cell before the formula is recognised.
    """
    markdown = path.read_text(encoding="utf-8")
    for number, line in enumerate(markdown.split("\n"), start=1):
        assert not re.match(r"^([ \t]+|[ \t]*>[ \t>]*)```math", line), f"{path.name}:{number}: ```math in a list/quote"
        for bold in re.findall(r"\*\*(.+?)\*\*", line):
            assert "$`" not in bold, f"{path.name}:{number}: formula inside bold text: {bold!r}"
        if line.lstrip().startswith("|"):
            for tex in re.findall(r"\$`(.+?)`\$", line):
                assert "|" not in tex, f"{path.name}:{number}: '|' inside a formula in a table: {tex!r}"
