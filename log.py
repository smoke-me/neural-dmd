# Tiny colored console logger. Zero deps.
# Tags are short words so log lines stay readable both for humans and LLMs:
#   info  - neutral status (cyan)
#   ok    - success / metric you care about (green)
#   warn  - non-fatal anomaly (yellow)
#   err   - failure (red)

# ANSI color codes mapped to each tag.
_C = {
    "info": "\033[36m",
    "ok":   "\033[32m",
    "warn": "\033[33m",
    "err":  "\033[31m",
}
_R = "\033[0m"  # reset color


def log(tag: str, msg: str) -> None:
    # Print "<colored TAG>  <message>". Tag is upper-cased and padded to 4
    # chars so columns line up across log lines.
    print(f"{_C[tag]}{tag.upper():<4}{_R}  {msg}")
