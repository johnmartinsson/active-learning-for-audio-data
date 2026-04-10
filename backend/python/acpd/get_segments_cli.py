import json
import sys

from .pipeline import build_segments_response


def main():
    """Read JSON payload from stdin and print segmentation response as JSON.

    This entrypoint is designed for module execution via:
    ``python3 -m python.acpd.get_segments_cli``.
    """
    payload = json.loads(sys.stdin.read())
    response = build_segments_response(payload)
    print(json.dumps(response))


if __name__ == "__main__":
    main()
