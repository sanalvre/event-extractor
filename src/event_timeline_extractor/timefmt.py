"""Time strings for yt-dlp --download-sections (start 0:00 to end timestamp)."""


def seconds_to_end_timestamp(seconds: float) -> str:
    """e.g. 20 -> '0:20', 125 -> '2:05'."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def download_section_first_seconds(max_seconds: float) -> str:
    """yt-dlp --download-sections value: first max_seconds from 0:00."""
    end = seconds_to_end_timestamp(max_seconds)
    return f"*0:00-{end}"
