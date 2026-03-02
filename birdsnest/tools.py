########################################################################################################
# Bird's Nest — Tool Registry & Execution Engine
# Server-side tool calling for RWKV models using the RWKV Runner pattern
#
# Pattern:  Model outputs   →  FunctionName\n```python\ntool_call("arg"="val")\n```
#           Server detects  →  regex match on buffered output
#           Server executes →  runs handler, returns result
#           Result injected →  via Observation: role back to model
########################################################################################################

import re
import json
import math
import time
import platform
import subprocess
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Callable, List
from pathlib import Path


# ── Tool Registry ─────────────────────────────────────────────────────────────

class Tool:
    """A registered tool that the model can call."""
    def __init__(self, name: str, description: str, parameters: Dict, handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.enabled = True

    def to_schema(self) -> Dict:
        """Return OpenAI-compatible function schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# Global registry
_tools: Dict[str, Tool] = {}


def register_tool(name: str, description: str, parameters: Dict = None):
    """Decorator to register a tool function."""
    if parameters is None:
        parameters = {"type": "object", "properties": {}, "required": []}

    def decorator(func):
        _tools[name] = Tool(name, description, parameters, func)
        return func
    return decorator


def get_tools() -> Dict[str, Tool]:
    return _tools


def get_enabled_tools() -> List[Tool]:
    return [t for t in _tools.values() if t.enabled]


def toggle_tool(name: str, enabled: bool) -> bool:
    if name in _tools:
        _tools[name].enabled = enabled
        return True
    return False


# ── Tool Detection (Dual Format) ─────────────────────────────────────────────

# Format 1: RWKV Runner pattern
#   FunctionName\n```python\ntool_call("arg"="val")\n```
TOOL_CALL_PATTERN = re.compile(r'([\w]+)\s*```[\w\s]*tool_call\((.*?)\)\s*```', re.DOTALL)
TOOL_CALL_HEAD = re.compile(r'([\w]+)\s*```[\w\s]*tool_call\(')

# Format 2: JSON pattern (what RWKV World models actually generate)
#   {"name": "function_name", "arguments": {...}}
JSON_TOOL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})',
    re.DOTALL
)

# Thresholds — tuned for fast flush on normal text
BUFFER_CHAR_LIMIT = 80          # max chars before forced flush
BUFFER_NEWLINE_LIMIT = 3        # newlines before forced flush
FUNC_NAME_CHAR_LIMIT = 50       # max chars before ruling out RWKV pattern

# Characters that can start a tool call
_TOOL_CALL_STARTERS = None  # lazy-init from registered tool names

def _get_tool_starters():
    """Get first chars of all registered tool names for fast prefix check."""
    global _TOOL_CALL_STARTERS
    if _TOOL_CALL_STARTERS is None:
        _TOOL_CALL_STARTERS = {name[0] for name in _tools.keys() if name}
    return _TOOL_CALL_STARTERS


def detect_tool_call(buffer: str):
    """
    Check buffered text for a tool call in either format.
    Returns (format_type, match) or (None, None).
    """
    # Try RWKV Runner format first
    match = TOOL_CALL_PATTERN.search(buffer)
    if match:
        return 'rwkv', match

    # Try JSON format
    match = JSON_TOOL_PATTERN.search(buffer)
    if match:
        func_name = match.group(1)
        if func_name in _tools:
            return 'json', match

    return None, None


def is_definitely_not_tool_call(buffer: str) -> bool:
    """
    Check if we can confidently say this buffer is NOT a tool call.
    
    KEY OPTIMIZATION: Check the first character aggressively.
    Tool calls start with either '{' (JSON) or a function name (RWKV).
    Normal prose starts with spaces, letters like "I", "The", etc.
    """
    stripped = buffer.lstrip()
    if not stripped:
        return False  # empty, keep waiting

    first_char = stripped[0]

    # ── Fast path: first char can't be a tool call ──
    # JSON tool calls start with '{'
    # RWKV tool calls start with a function name letter
    if first_char != '{':
        # Check if it could be the start of a function name
        starters = _get_tool_starters()
        if first_char not in starters:
            # Definitely not a tool call — flush after just a few chars
            if len(stripped) >= 3:
                return True
        else:
            # Could be a function name — check if any tool name matches
            could_match = any(name.startswith(stripped.split()[0]) for name in _tools if stripped.split())
            if not could_match and len(stripped) > 15:
                return True

    # ── JSON path: starts with { ──
    if first_char == '{':
        if '"name"' in buffer:
            return False  # looks like a JSON tool call, keep buffering
        if len(stripped) > 30 and '"name"' not in buffer:
            return True   # brace opened but no "name" key — not a tool call

    # ── General fallbacks ──
    if buffer.count('\n') >= BUFFER_NEWLINE_LIMIT:
        return True
    if len(buffer) > BUFFER_CHAR_LIMIT:
        return True

    # Check for RWKV Runner head
    if TOOL_CALL_HEAD.search(buffer):
        return False

    if len(buffer) > FUNC_NAME_CHAR_LIMIT and '```' not in buffer and '{' not in buffer:
        return True

    return False


def parse_tool_call(fmt: str, match) -> Tuple[str, Dict[str, str]]:
    """
    Parse a tool call match into (function_name, arguments_dict).
    Handles both RWKV Runner and JSON formats.
    """
    if fmt == 'json':
        func_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        return func_name, args

    # RWKV Runner format
    func_name = match.group(1)
    args_str = match.group(2).strip()

    args = {}
    if args_str:
        # Parse "key"="value" pairs (RWKV Runner pattern)
        arg_pattern = re.compile(r'''["']([^"']+)["']\s*=\s*["']([^"']+)["']''')
        for key, value in arg_pattern.findall(args_str):
            args[key] = value

        # Also try simple unquoted: key=value
        if not args:
            simple_pattern = re.compile(r'(\w+)\s*=\s*["\']?([^,"\']+)["\']?')
            for key, value in simple_pattern.findall(args_str):
                args[key] = value.strip()

        # Fallback: if single arg with no key, treat as "input"
        if not args and args_str:
            clean = args_str.strip().strip('"').strip("'")
            if clean:
                args["input"] = clean

    return func_name, args


def execute_tool(name: str, args: Dict[str, str]) -> str:
    """Execute a registered tool and return its result as a string."""
    tool = _tools.get(name)
    if not tool:
        return f"Error: Unknown tool '{name}'"
    if not tool.enabled:
        return f"Error: Tool '{name}' is disabled"
    try:
        result = tool.handler(args)
        return str(result)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


# ── System Prompt Builder ─────────────────────────────────────────────────────

def build_tool_system_prompt() -> str:
    """
    Minimal system prompt — just inform the model tools exist.
    Actual tool routing is handled server-side via detect_user_intent().
    """
    return ""  # No system prompt injection — tools are handled server-side now


def detect_user_intent(message: str) -> Optional[Tuple[str, Dict]]:
    """
    Server-side intent detection: pattern-match the user's message
    to determine if a tool should fire BEFORE the model generates.
    
    Returns (tool_name, args) or None if no tool match.
    This is the key for small models that can't reliably output JSON.
    """
    msg = message.lower().strip()

    # ── Time queries ──
    time_patterns = [
        'what time', 'current time', 'the time', 'time is it',
        'what\'s the time', 'tell me the time', 'time right now',
        'what date', 'today\'s date', 'current date', 'what day',
    ]
    if any(p in msg for p in time_patterns):
        if 'get_current_time' in _tools and _tools['get_current_time'].enabled:
            return ('get_current_time', {})

    # ── Math expressions ──
    math_patterns = [
        r'what is [\d\.\s\+\-\*\/\(\)\^x×\w]+',
        r'calculate [\d\.\s\+\-\*\/\(\)\^x×\w]+',
        r'(\d+)\s*(times|plus|minus|divided by|multiplied by|x|×|\*|\+|\-|\/)\s*(\d+)',
        r'how much is [\d\.\s\+\-\*\/\(\)\^x×\w]+',
    ]
    for pattern in math_patterns:
        m = re.search(pattern, msg)
        if m and 'calculate' in _tools and _tools['calculate'].enabled:
            # Extract the math expression
            expr = m.group(0)
            # Clean up natural language to math operators
            expr = re.sub(r'\bwhat is\b', '', expr)
            expr = re.sub(r'\bcalculate\b', '', expr)
            expr = re.sub(r'\bhow much is\b', '', expr)
            expr = re.sub(r'\btimes\b', '*', expr)
            expr = re.sub(r'\bmultiplied by\b', '*', expr)
            expr = re.sub(r'\bx\b', '*', expr)
            expr = re.sub(r'×', '*', expr)
            expr = re.sub(r'\bplus\b', '+', expr)
            expr = re.sub(r'\bminus\b', '-', expr)
            expr = re.sub(r'\bdivided by\b', '/', expr)
            expr = expr.strip()
            if expr:
                return ('calculate', {'expression': expr})

    # ── Search queries ──
    search_patterns = [
        (r'search (?:for |the web for |google for |the internet for )?(.+)', 1),
        (r'look up (.+)', 1),
        (r'find (?:information (?:about|on) )?(.+)', 1),
        (r'google (.+)', 1),
    ]
    for pattern, group in search_patterns:
        m = re.search(pattern, msg)
        if m and 'search_web' in _tools and _tools['search_web'].enabled:
            query = m.group(group).strip()
            if query and len(query) > 2:
                return ('search_web', {'query': query})

    # ── System info ──
    sys_patterns = ['system info', 'system status', 'cpu usage', 'ram usage', 'disk space', 'how much memory']
    if any(p in msg for p in sys_patterns):
        if 'get_system_info' in _tools and _tools['get_system_info'].enabled:
            return ('get_system_info', {})

    # ── Shell commands ──
    shell_patterns = [
        (r'^run (?:command |shell )?(.+)', 1),
        (r'^execute (.+)', 1),
        (r'^shell (.+)', 1),
        (r'^\$ (.+)', 1),
    ]
    for pattern, group in shell_patterns:
        m = re.search(pattern, msg)
        if m and 'run_shell' in _tools and _tools['run_shell'].enabled:
            return ('run_shell', {'command': m.group(group).strip()})

    # ── YouTube transcripts ──
    yt_patterns = [
        (r'transcript (?:of |for |from )?(?:this )?(?:video )?(.+)', 1),
        (r'subtitles (?:of |for |from )?(.+)', 1),
    ]
    for pattern, group in yt_patterns:
        m = re.search(pattern, msg)
        if m and 'youtube_transcript' in _tools and _tools['youtube_transcript'].enabled:
            url = m.group(group).strip()
            if 'youtu' in url or len(url) == 11:  # YouTube URL or video ID
                return ('youtube_transcript', {'url': url})
    # Direct YouTube URL detection
    yt_url = re.search(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+)', msg)
    if yt_url and 'transcript' in msg:
        if 'youtube_transcript' in _tools and _tools['youtube_transcript'].enabled:
            return ('youtube_transcript', {'url': yt_url.group(1)})

    # ── Memory ──
    if any(p in msg for p in ['remember this', 'save this', 'store this', 'remember that']):
        if 'memory' in _tools and _tools['memory'].enabled:
            # Try to extract key=value from message
            content = re.sub(r'(remember|save|store) (this|that):?\s*', '', msg).strip()
            if content:
                key = content.split()[0] if content.split() else 'note'
                return ('memory', {'action': 'save', 'key': key, 'value': content})
    if any(p in msg for p in ['what did i save', 'my memories', 'list memories', 'what do you remember']):
        if 'memory' in _tools and _tools['memory'].enabled:
            return ('memory', {'action': 'list'})
    recall_m = re.search(r'(?:recall|remember|what was) (.+)', msg)
    if recall_m and 'memory' in _tools and _tools['memory'].enabled:
        key = recall_m.group(1).strip()
        if key and not any(p in key for p in ['this', 'that', 'time', 'date']):
            return ('memory', {'action': 'recall', 'key': key})

    # ── Weather ──
    weather_patterns = [
        (r'weather (?:in |for |at )?(.+)', 1),
        (r'forecast (?:in |for |at )?(.+)', 1),
        (r"what's the weather (?:in |at |like in )?(.+)", 1),
        (r'temperature (?:in |at )?(.+)', 1),
    ]
    for pattern, group in weather_patterns:
        m = re.search(pattern, msg)
        if m and 'weather' in _tools and _tools['weather'].enabled:
            location = m.group(group).strip().rstrip('?')
            if location:
                return ('weather', {'location': location})

    # ── Clipboard ──
    clip_patterns = ['clipboard', 'paste', 'what did i copy', 'show clipboard', 'my clipboard',
                     "what's on my clipboard", 'read clipboard']
    if any(p in msg for p in clip_patterns):
        if 'clipboard' in _tools and _tools['clipboard'].enabled:
            return ('clipboard', {'action': 'read'})

    # ── Screenshot ──
    ss_patterns = ['take a screenshot', 'screenshot', 'screencap', 'capture screen', 'screen grab']
    if any(p in msg for p in ss_patterns):
        if 'screenshot' in _tools and _tools['screenshot'].enabled:
            return ('screenshot', {})

    # ── Todo ──
    todo_add = re.search(r'(?:add (?:a )?todo|new todo|add task|new task):?\s*(.+)', msg)
    if todo_add and 'todo' in _tools and _tools['todo'].enabled:
        return ('todo', {'action': 'add', 'task': todo_add.group(1).strip()})
    if any(p in msg for p in ['list todos', 'my todos', 'show todos', 'show tasks', 'list tasks', 'my tasks']):
        if 'todo' in _tools and _tools['todo'].enabled:
            return ('todo', {'action': 'list'})
    todo_done = re.search(r'(?:complete|finish|done|check) (?:todo |task )?#?(\d+)', msg)
    if todo_done and 'todo' in _tools and _tools['todo'].enabled:
        return ('todo', {'action': 'complete', 'task': todo_done.group(1)})
    todo_del = re.search(r'delete (?:todo |task )?#?(\d+)', msg)
    if todo_del and 'todo' in _tools and _tools['todo'].enabled:
        return ('todo', {'action': 'delete', 'task': todo_del.group(1)})

    # ── Translate ──
    trans_m = re.search(r'translate (.+?)(?:\s+(?:to|into|in)\s+)(\w+)', msg)
    if trans_m and 'translate' in _tools and _tools['translate'].enabled:
        return ('translate', {'text': trans_m.group(1).strip().strip('"\''), 'to': trans_m.group(2).strip()})

    # ── Image Generation ──
    img_patterns = [
        (r'(?:generate|create|make|draw|paint) (?:an? )?(?:image|picture|photo|art|illustration) (?:of |about |showing )?(.+)', 1),
        (r'(?:generate|create|make|draw|paint) (.+)', 1),
        (r'imagine (.+)', 1),
    ]
    for pattern, group in img_patterns:
        m = re.search(pattern, msg)
        if m and 'generate_image' in _tools and _tools['generate_image'].enabled:
            prompt = m.group(group).strip().rstrip('.')
            if prompt and len(prompt) > 3:  # Avoid triggering on "make it" etc
                return ('generate_image', {'prompt': prompt})

    # ── Database Query ──
    db_patterns = [
        (r'query (?:database |db )?(.+?):\s*(.+)', 1, 2),
        (r'sql (.+?):\s*(.+)', 1, 2),
    ]
    for pattern, path_group, query_group in db_patterns:
        m = re.search(pattern, msg)
        if m and 'query_database' in _tools and _tools['query_database'].enabled:
            return ('query_database', {'db_path': m.group(path_group).strip(), 'query': m.group(query_group).strip()})

    return None


# ── Built-in Tools ────────────────────────────────────────────────────────────

@register_tool(
    "get_current_time",
    "Get the current date and time",
)
def tool_get_current_time(args: Dict) -> str:
    fmt = args.get("format", "%A, %B %d, %Y at %I:%M:%S %p")
    return datetime.now().strftime(fmt)


@register_tool(
    "calculate",
    "Evaluate a mathematical expression safely",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        },
        "required": ["expression"],
    },
)
def tool_calculate(args: Dict) -> str:
    expr = args.get("expression") or args.get("input", "")
    # Safe math eval — only allow math functions
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "pow": pow, "int": int, "float": float,
        "pi": math.pi, "e": math.e, "sqrt": math.sqrt,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "log2": math.log2,
        "ceil": math.ceil, "floor": math.floor,
        "factorial": math.factorial, "gcd": math.gcd,
    }
    try:
        result = eval(expr, safe_globals, safe_locals)
        return str(result)
    except Exception as e:
        return f"Math error: {e}"


@register_tool(
    "get_system_info",
    "Get system information including CPU, RAM, GPU, and disk usage",
)
def tool_get_system_info(args: Dict) -> str:
    info = []
    info.append(f"OS: {platform.system()} {platform.release()}")
    info.append(f"Machine: {platform.machine()}")
    info.append(f"CPU: {platform.processor() or 'Unknown'}")

    # CPU usage
    try:
        cpu_pct = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count()
        info.append(f"CPU Usage: {cpu_pct}% ({cpu_count} cores)")
    except:
        pass

    # Memory
    try:
        mem = psutil.virtual_memory()
        info.append(f"RAM: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB ({mem.percent}%)")
    except:
        pass

    # Disk
    try:
        disk = psutil.disk_usage("/")
        info.append(f"Disk: {disk.used / (1024**3):.1f} GB / {disk.total / (1024**3):.1f} GB ({disk.percent}%)")
    except:
        pass

    # GPU (Mac MPS)
    try:
        import torch
        if torch.backends.mps.is_available():
            info.append("GPU: Apple MPS (Metal Performance Shaders) — Available")
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            info.append(f"GPU Memory Allocated: {allocated:.2f} GB")
    except:
        pass

    return "\n".join(info)


@register_tool(
    "search_web",
    "Search the web using DuckDuckGo and return results",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"],
    },
)
def tool_search_web(args: Dict) -> str:
    query = args.get("query") or args.get("input", "")
    if not query:
        return "Error: No search query provided"

    import urllib.request
    import urllib.parse
    from html.parser import HTMLParser

    class DDGParser(HTMLParser):
        """Parse DuckDuckGo HTML results — captures title, URL, and snippet."""
        def __init__(self):
            super().__init__()
            self.results = []  # list of {title, url, snippet}
            self._current = {}
            self._capture = None  # 'title' | 'snippet' | None
            self._text = ""

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            cls = attrs_dict.get("class", "")

            # Result title link — has both title text and href
            if tag == "a" and "result__a" in cls:
                href = attrs_dict.get("href", "")
                # DuckDuckGo wraps URLs in redirect — extract actual URL
                if "uddg=" in href:
                    import urllib.parse as up
                    parsed = up.parse_qs(up.urlparse(href).query)
                    href = parsed.get("uddg", [href])[0]
                self._current = {"title": "", "url": href, "snippet": ""}
                self._capture = "title"
                self._text = ""

            elif "result__snippet" in cls:
                self._capture = "snippet"
                self._text = ""

        def handle_endtag(self, tag):
            if self._capture == "title" and tag == "a":
                self._current["title"] = self._text.strip()
                self._capture = None
            elif self._capture == "snippet":
                self._current["snippet"] = self._text.strip()
                if self._current.get("title"):
                    self.results.append(self._current.copy())
                self._current = {}
                self._capture = None

        def handle_data(self, data):
            if self._capture:
                self._text += data

    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) BirdsNest/1.0"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        parser = DDGParser()
        parser.feed(html)

        if not parser.results:
            return f"No results found for: {query}"

        # Deduplicate by URL and limit to 5
        seen = set()
        unique = []
        for r in parser.results:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)
            if len(unique) >= 5:
                break

        output = f"Search results for: {query}\n\n"
        for i, r in enumerate(unique, 1):
            output += f"{i}. {r['title']}\n"
            output += f"   {r['url']}\n"
            if r['snippet']:
                output += f"   {r['snippet'][:120]}\n"
            output += "\n"
        return output

    except Exception as e:
        return f"Search error: {str(e)}"


@register_tool(
    "fetch_url",
    "Fetch and extract text content from a URL",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch content from"}
        },
        "required": ["url"],
    },
)
def tool_fetch_url(args: Dict) -> str:
    url = args.get("url") or args.get("input", "")
    if not url:
        return "Error: No URL provided"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    import urllib.request
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        """Extract visible text from HTML, skipping script/style."""
        def __init__(self):
            super().__init__()
            self.text_parts = []
            self._skip = False
            self._skip_tags = {"script", "style", "noscript", "svg", "head"}

        def handle_starttag(self, tag, attrs):
            if tag.lower() in self._skip_tags:
                self._skip = True
            if tag.lower() in ("br", "p", "div", "h1", "h2", "h3", "h4", "li"):
                self.text_parts.append("\n")

        def handle_endtag(self, tag):
            if tag.lower() in self._skip_tags:
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                text = data.strip()
                if text:
                    self.text_parts.append(text)

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) BirdsNest/1.0"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Cannot extract text from content type: {content_type}"
            html = resp.read(100_000).decode("utf-8", errors="replace")  # 100KB limit

        parser = TextExtractor()
        parser.feed(html)

        text = " ".join(parser.text_parts)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Truncate to ~2000 chars for model context
        if len(text) > 2000:
            text = text[:2000] + "\n\n[...truncated]"

        return f"Content from {url}:\n\n{text}" if text else f"No readable content found at {url}"

    except Exception as e:
        return f"Fetch error: {str(e)}"


# ── Tier 2 Tools ─────────────────────────────────────────────────────────────

# Sandbox directory for file operations
WORKSPACE_DIR = Path.home() / "birdsnest_workspace"

SAFE_READ_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".py", ".js", ".html", ".css",
    ".yaml", ".yml", ".toml", ".xml", ".log", ".sh", ".conf", ".cfg",
    ".ini", ".env", ".sql", ".r", ".rs", ".go", ".java", ".ts",
}


@register_tool(
    "read_file",
    "Read the contents of a local file",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["path"],
    },
)
def tool_read_file(args: Dict) -> str:
    file_path = args.get("path") or args.get("input", "")
    if not file_path:
        return "Error: No file path provided"

    p = Path(file_path).expanduser().resolve()

    if not p.exists():
        return f"Error: File not found: {p}"
    if not p.is_file():
        return f"Error: Not a file: {p}"
    if p.suffix.lower() not in SAFE_READ_EXTENSIONS:
        return f"Error: Cannot read files with extension '{p.suffix}'. Allowed: {', '.join(sorted(SAFE_READ_EXTENSIONS))}"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 5000:
            content = content[:5000] + f"\n\n[...truncated at 5000 chars, total: {len(content)}]"
        return f"File: {p.name}\n\n{content}"
    except Exception as e:
        return f"Read error: {str(e)}"


@register_tool(
    "write_file",
    "Write content to a file in the workspace directory (~/birdsnest_workspace/)",
    {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Filename to create (written to ~/birdsnest_workspace/)"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        "required": ["filename", "content"],
    },
)
def tool_write_file(args: Dict) -> str:
    filename = args.get("filename", "")
    content = args.get("content", "")
    if not filename:
        return "Error: No filename provided"
    if not content:
        return "Error: No content provided"

    # Security: strip path traversal, force into sandbox
    safe_name = Path(filename).name  # strips any directory components
    if not safe_name or safe_name.startswith("."):
        return "Error: Invalid filename"

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    target = WORKSPACE_DIR / safe_name

    try:
        target.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {target}"
    except Exception as e:
        return f"Write error: {str(e)}"


@register_tool(
    "run_python",
    "Execute Python code and return the output (10 second timeout)",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"],
    },
)
def tool_run_python(args: Dict) -> str:
    code = args.get("code") or args.get("input", "")
    if not code:
        return "Error: No Python code provided"

    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(WORKSPACE_DIR.mkdir(parents=True, exist_ok=True) or WORKSPACE_DIR),
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr

        if not output.strip():
            output = "(no output)"

        # Truncate
        if len(output) > 3000:
            output = output[:3000] + "\n\n[...truncated]"

        return output

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (10 second limit)"
    except Exception as e:
        return f"Execution error: {str(e)}"


@register_tool(
    "list_directory",
    "List files and directories at a given path",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list (defaults to workspace)"}
        },
        "required": [],
    },
)
def tool_list_directory(args: Dict) -> str:
    dir_path = args.get("path", "")
    if not dir_path:
        dir_path = str(WORKSPACE_DIR)

    p = Path(dir_path).expanduser().resolve()

    if not p.exists():
        return f"Error: Path not found: {p}"
    if not p.is_dir():
        return f"Error: Not a directory: {p}"

    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        if not entries:
            return f"Directory is empty: {p}"

        lines = [f"Contents of {p}:\n"]
        for entry in entries[:50]:  # limit to 50 entries
            if entry.is_dir():
                lines.append(f"  📁 {entry.name}/")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    sz = f"{size} B"
                elif size < 1024 * 1024:
                    sz = f"{size/1024:.1f} KB"
                else:
                    sz = f"{size/(1024*1024):.1f} MB"
                lines.append(f"  📄 {entry.name}  ({sz})")

        if len(list(p.iterdir())) > 50:
            lines.append(f"\n  ...and {len(list(p.iterdir())) - 50} more")

        return "\n".join(lines)

    except PermissionError:
        return f"Error: Permission denied: {p}"
    except Exception as e:
        return f"List error: {str(e)}"


# ── Tier 3: Advanced Tools ────────────────────────────────────────────────────

@register_tool(
    "run_shell",
    "Execute a shell command and return the output (15 second timeout)",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"}
        },
        "required": ["command"],
    },
)
def tool_run_shell(args: Dict) -> str:
    command = args.get("command") or args.get("input", "")
    if not command:
        return "Error: No command provided"

    # Block dangerous commands
    BLOCKED = ['rm -rf /', 'mkfs', 'dd if=', ':(){', 'shutdown', 'reboot',
               'halt', 'poweroff', 'init 0', 'init 6']
    cmd_lower = command.lower().strip()
    for b in BLOCKED:
        if b in cmd_lower:
            return f"Error: Blocked command (safety): {b}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(Path.home()),
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        if not output.strip():
            output = "(no output)"

        # Truncate
        if len(output) > 4000:
            output = output[:4000] + "\n\n[...truncated]"

        return output

    except subprocess.TimeoutExpired:
        return "Error: Command timed out (15 second limit)"
    except Exception as e:
        return f"Shell error: {str(e)}"


@register_tool(
    "youtube_transcript",
    "Fetch the transcript/subtitles from a YouTube video",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "YouTube video URL or ID"}
        },
        "required": ["url"],
    },
)
def tool_youtube_transcript(args: Dict) -> str:
    url = args.get("url") or args.get("input", "")
    if not url:
        return "Error: No YouTube URL provided"

    # Extract video ID
    video_id = url
    if "youtube.com" in url or "youtu.be" in url:
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        if "youtu.be" in parsed.netloc:
            video_id = parsed.path.lstrip("/")
        else:
            qs = urllib.parse.parse_qs(parsed.query)
            video_id = qs.get("v", [url])[0]

    # Try yt-dlp for subtitles (most reliable)
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--write-auto-sub", "--sub-lang", "en",
             "--convert-subs", "srt", "--print-to-file", "subtitle:%(id)s.srt",
             "-o", "%(id)s", f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=30,
            cwd="/tmp",
        )

        srt_path = Path(f"/tmp/{video_id}.en.srt")
        if not srt_path.exists():
            srt_path = Path(f"/tmp/{video_id}.srt")

        if srt_path.exists():
            raw = srt_path.read_text(encoding="utf-8", errors="replace")
            # Strip SRT formatting (timestamps, indices)
            lines = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    continue
                if "-->" in line:
                    continue
                # Remove HTML tags from auto-subs
                line = re.sub(r'<[^>]+>', '', line)
                if line and line not in lines[-1:]:  # dedupe consecutive
                    lines.append(line)
            srt_path.unlink(missing_ok=True)

            text = " ".join(lines)
            if len(text) > 3000:
                text = text[:3000] + "\n\n[...truncated]"

            return f"Transcript for {video_id}:\n\n{text}"
        else:
            return f"No English subtitles found for video: {video_id}"

    except FileNotFoundError:
        return "Error: yt-dlp not installed. Run: pip install yt-dlp"
    except subprocess.TimeoutExpired:
        return "Error: Transcript fetch timed out (30s limit)"
    except Exception as e:
        return f"Transcript error: {str(e)}"


# ── Memory Tool ──────────────────────────────────────────────────────────────

MEMORY_FILE = Path.home() / "birdsnest_workspace" / ".memory.json"

def _load_memory() -> Dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_memory(data: Dict):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(data, indent=2))


@register_tool(
    "memory",
    "Save, recall, list, or delete persistent memories across sessions",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action: save, recall, list, or delete"},
            "key": {"type": "string", "description": "Memory key/name"},
            "value": {"type": "string", "description": "Value to save (only for save action)"},
        },
        "required": ["action"],
    },
)
def tool_memory(args: Dict) -> str:
    action = (args.get("action") or "list").lower().strip()
    key = args.get("key", "").strip()
    value = args.get("value", "")

    mem = _load_memory()

    if action == "save":
        if not key:
            return "Error: 'key' required for save"
        if not value:
            return "Error: 'value' required for save"
        mem[key] = {"value": value, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        _save_memory(mem)
        return f"Saved memory: '{key}'"

    elif action in ("recall", "get", "load"):
        if not key:
            return "Error: 'key' required for recall"
        if key in mem:
            entry = mem[key]
            return f"Memory '{key}' (saved {entry.get('saved_at', '?')}):\n{entry['value']}"
        else:
            return f"No memory found for key: '{key}'"

    elif action == "list":
        if not mem:
            return "No memories saved yet."
        lines = ["Saved memories:\n"]
        for k, v in mem.items():
            preview = str(v.get("value", ""))[:60]
            lines.append(f"  • {k}: {preview}")
        return "\n".join(lines)

    elif action == "delete":
        if not key:
            return "Error: 'key' required for delete"
        if key in mem:
            del mem[key]
            _save_memory(mem)
            return f"Deleted memory: '{key}'"
        else:
            return f"No memory found for key: '{key}'"

    else:
        return f"Unknown action: '{action}'. Use: save, recall, list, or delete"


# ── Tier 4: Sidebar Tools ────────────────────────────────────────────────────

@register_tool(
    "weather",
    "Get current weather for a city or location",
    {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or location"}
        },
        "required": ["location"],
    },
)
def tool_weather(args: Dict) -> str:
    location = args.get("location") or args.get("input", "")
    if not location:
        return "Error: No location provided"

    # wttr.in — free, no API key needed
    try:
        import urllib.request
        url = f"https://wttr.in/{urllib.request.quote(location)}?format=%l:+%C+%t+%h+%w"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = resp.read().decode("utf-8").strip()
        
        # Also get a more detailed forecast
        url2 = f"https://wttr.in/{urllib.request.quote(location)}?format=%l\\n🌡️+%t+(feels+like+%f)\\n💧+Humidity:+%h\\n💨+Wind:+%w\\n🌅+Sunrise:+%S+|+Sunset:+%s\\n📊+UV+Index:+%u"
        req2 = urllib.request.Request(url2, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req2, timeout=10) as resp2:
            detail = resp2.read().decode("utf-8").strip()

        return f"Weather for {location}:\n\n{detail}"

    except Exception as e:
        return f"Weather error: {str(e)}"


@register_tool(
    "clipboard",
    "Read or write the system clipboard",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "read or write"},
            "text": {"type": "string", "description": "Text to copy (for write action)"},
        },
        "required": [],
    },
)
def tool_clipboard(args: Dict) -> str:
    action = (args.get("action") or "read").lower().strip()
    text = args.get("text", "")

    if action == "write" and text:
        try:
            proc = subprocess.run(
                ["pbcopy"], input=text, text=True, timeout=5
            )
            return f"Copied {len(text)} chars to clipboard"
        except Exception as e:
            return f"Clipboard write error: {str(e)}"
    else:
        # Read clipboard
        try:
            result = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, timeout=5
            )
            content = result.stdout
            if not content:
                return "Clipboard is empty"
            if len(content) > 2000:
                content = content[:2000] + "\n\n[...truncated]"
            return f"Clipboard contents ({len(result.stdout)} chars):\n\n{content}"
        except Exception as e:
            return f"Clipboard read error: {str(e)}"


@register_tool(
    "screenshot",
    "Take a screenshot of the screen",
    {
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "Optional: 'full' (default) or 'selection'"},
        },
        "required": [],
    },
)
def tool_screenshot(args: Dict) -> str:
    region = (args.get("region") or "full").lower().strip()
    
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = WORKSPACE_DIR / f"screenshot_{timestamp}.png"

    try:
        cmd = ["screencapture", "-x"]  # -x = no sound
        if region == "selection":
            cmd.append("-i")  # interactive selection
        cmd.append(str(filepath))

        result = subprocess.run(cmd, timeout=15, capture_output=True, text=True)

        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            return f"Screenshot saved: {filepath}\nSize: {size_kb:.1f} KB"
        else:
            return "Screenshot was cancelled or failed"

    except subprocess.TimeoutExpired:
        return "Screenshot timed out"
    except Exception as e:
        return f"Screenshot error: {str(e)}"


# ── Todo Tool ────────────────────────────────────────────────────────────────

TODOS_FILE = Path.home() / "birdsnest_workspace" / ".todos.json"

def _load_todos() -> list:
    if TODOS_FILE.exists():
        try:
            return json.loads(TODOS_FILE.read_text())
        except Exception:
            return []
    return []

def _save_todos(data: list):
    TODOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TODOS_FILE.write_text(json.dumps(data, indent=2))


@register_tool(
    "todo",
    "Manage a persistent todo/task list (add, list, complete, delete)",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "add, list, complete, or delete"},
            "task": {"type": "string", "description": "Task description (for add) or task number (for complete/delete)"},
        },
        "required": ["action"],
    },
)
def tool_todo(args: Dict) -> str:
    action = (args.get("action") or "list").lower().strip()
    task = args.get("task", "").strip()

    todos = _load_todos()

    if action == "add":
        if not task:
            return "Error: task description required"
        todos.append({
            "task": task,
            "done": False,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        })
        _save_todos(todos)
        return f"Added todo #{len(todos)}: {task}"

    elif action == "list":
        if not todos:
            return "No todos yet. Add one with: add todo [task]"
        lines = ["Your todos:\n"]
        for i, t in enumerate(todos, 1):
            status = "✅" if t.get("done") else "⬜"
            lines.append(f"  {status} {i}. {t['task']}")
        done_count = sum(1 for t in todos if t.get("done"))
        lines.append(f"\n{done_count}/{len(todos)} completed")
        return "\n".join(lines)

    elif action in ("complete", "done", "check"):
        try:
            idx = int(task) - 1
            if 0 <= idx < len(todos):
                todos[idx]["done"] = True
                _save_todos(todos)
                return f"Completed: {todos[idx]['task']}"
            else:
                return f"Invalid todo number: {task}. You have {len(todos)} todos."
        except ValueError:
            return "Error: provide the todo number to complete"

    elif action == "delete":
        try:
            idx = int(task) - 1
            if 0 <= idx < len(todos):
                removed = todos.pop(idx)
                _save_todos(todos)
                return f"Deleted: {removed['task']}"
            else:
                return f"Invalid todo number: {task}"
        except ValueError:
            return "Error: provide the todo number to delete"

    else:
        return f"Unknown action: '{action}'. Use: add, list, complete, or delete"


@register_tool(
    "translate",
    "Translate text between languages",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "to": {"type": "string", "description": "Target language (e.g. 'spanish', 'french', 'ja')"},
            "from_lang": {"type": "string", "description": "Source language (auto-detect if omitted)"},
        },
        "required": ["text", "to"],
    },
)
def tool_translate(args: Dict) -> str:
    text = args.get("text", "")
    to_lang = args.get("to", "")
    if not text or not to_lang:
        return "Error: 'text' and 'to' (target language) are required"

    # Map common names to ISO codes
    lang_map = {
        "english": "en", "spanish": "es", "french": "fr", "german": "de",
        "italian": "it", "portuguese": "pt", "russian": "ru", "chinese": "zh",
        "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi",
        "dutch": "nl", "turkish": "tr", "polish": "pl", "swedish": "sv",
    }
    to_code = lang_map.get(to_lang.lower(), to_lang.lower())
    from_lang = args.get("from_lang", "en")
    from_code = lang_map.get(from_lang.lower(), from_lang.lower())

    # ── Primary: CTranslate2 + Opus-MT (fully local) ──
    try:
        from transformers import MarianMTModel, MarianTokenizer

        hf_model = f"Helsinki-NLP/opus-mt-{from_code}-{to_code}"

        # Load tokenizer + model (cached by HuggingFace after first download ~300MB)
        tokenizer = MarianTokenizer.from_pretrained(hf_model)
        model = MarianMTModel.from_pretrained(hf_model)

        # Translate locally
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        return f"🔒 Translation ({from_code} → {to_code}, local):\n\n{result}"

    except Exception:
        pass

    # ── Fallback: translate-shell (uses Google — NOT local) ──
    try:
        to_code_cli = {
            "chinese": "zh-CN", "japanese": "ja", "korean": "ko",
        }.get(to_lang.lower(), to_code)

        result = subprocess.run(
            ["trans", "-b", f":{to_code_cli}", text],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            return f"⚠️ Translation (→ {to_lang}, via Google — not local):\n\n{result.stdout.strip()}\n\n(Local Opus-MT model not available for {from_code}→{to_code})"
        return f"Translation failed."

    except FileNotFoundError:
        return (
            f"No translation backend available.\n"
            f"Local model not found for {from_code}→{to_code}.\n"
            f"Install translate-shell as fallback: brew install translate-shell"
        )
    except Exception as e:
        return f"Translation error: {str(e)}"


# ── Final Tools: Image Gen + Database ─────────────────────────────────────────

IMAGES_DIR = Path.home() / "birdsnest_workspace" / "images"

@register_tool(
    "generate_image",
    "Generate an image from a text prompt using local AI (MLX Flux)",
    {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Text description of the image to generate"},
            "steps": {"type": "integer", "description": "Number of inference steps (default 4, more=better quality)"},
            "width": {"type": "integer", "description": "Image width (default 512)"},
            "height": {"type": "integer", "description": "Image height (default 512)"},
        },
        "required": ["prompt"],
    },
)
def tool_generate_image(args: Dict) -> str:
    prompt = args.get("prompt", "")
    if not prompt:
        return "Error: prompt is required"

    steps = args.get("steps", 4)
    width = args.get("width", 512)
    height = args.get("height", 512)

    # Ensure output directory
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Sanitize prompt for filename
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', prompt[:40]).strip('_').lower()
    filename = f"{safe_name}_{timestamp}.png"
    filepath = IMAGES_DIR / filename

    # Check if mflux is available
    try:
        check = subprocess.run(
            ["mflux-generate", "--help"],
            capture_output=True, text=True, timeout=5
        )
        if check.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return (
            "mflux not installed or not found.\n"
            "Install with: pip install mflux\n"
            "First run will download the Flux model (~12GB)."
        )

    # Build the mflux command
    cmd = [
        "mflux-generate",
        "--prompt", prompt,
        "--output", str(filepath),
        "--steps", str(steps),
        "--width", str(width),
        "--height", str(height),
        "--model", "schnell",  # Fast model (4 steps)
    ]

    try:
        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max
        )
        elapsed = time.time() - t0

        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            # Return the serve URL for the frontend
            serve_url = f"/workspace/images/{filename}"
            return (
                f"Image generated in {elapsed:.1f}s\n"
                f"Prompt: {prompt}\n"
                f"Size: {width}x{height}, {size_kb:.1f} KB\n"
                f"Steps: {steps}\n"
                f"File: {filepath}\n"
                f"URL: {serve_url}"
            )
        else:
            stderr = result.stderr[-500:] if result.stderr else "No error output"
            return f"Image generation failed.\nError: {stderr}"

    except subprocess.TimeoutExpired:
        return "Image generation timed out (5 min limit)"
    except Exception as e:
        return f"Image generation error: {str(e)}"


@register_tool(
    "query_database",
    "Query a local SQLite database with SQL",
    {
        "type": "object",
        "properties": {
            "db_path": {"type": "string", "description": "Path to the SQLite database file"},
            "query": {"type": "string", "description": "SQL query to execute"},
        },
        "required": ["db_path", "query"],
    },
)
def tool_query_database(args: Dict) -> str:
    import sqlite3

    db_path = args.get("db_path", "")
    query = args.get("query", "")

    if not db_path or not query:
        return "Error: db_path and query are required"

    # Expand home directory
    db_path = str(Path(db_path).expanduser())

    # Check file exists
    if not Path(db_path).exists():
        return f"Database not found: {db_path}"

    if not Path(db_path).is_file():
        return f"Not a file: {db_path}"

    # Safety: block destructive SQL
    dangerous = ['drop ', 'delete ', 'alter ', 'truncate ', 'update ', 'insert ', 'create ', 'attach ']
    query_lower = query.lower().strip()
    if any(query_lower.startswith(d) for d in dangerous):
        return f"Blocked: destructive SQL not allowed (query starts with '{query_lower.split()[0]}'). Read-only queries only."

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchmany(100)  # Max 100 rows

        if not rows:
            conn.close()
            return "Query returned no results."

        # Format as table
        columns = rows[0].keys()
        lines = [" | ".join(columns)]
        lines.append("-" * len(lines[0]))

        for row in rows:
            lines.append(" | ".join(str(row[c]) if row[c] is not None else "NULL" for c in columns))

        total = cursor.rowcount if cursor.rowcount >= 0 else len(rows)
        result = "\n".join(lines)
        if len(rows) == 100:
            result += "\n\n[...showing first 100 rows]"

        conn.close()
        return f"Query: {query}\nResults ({len(rows)} rows):\n\n{result}"

    except sqlite3.Error as e:
        return f"SQL error: {str(e)}"
    except Exception as e:
        return f"Database error: {str(e)}"


