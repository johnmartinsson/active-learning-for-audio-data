import json
import sys

from .strategies import sample_files


def main():
    """Read JSON payload from stdin and print sampled filenames as JSON."""
    payload = json.loads(sys.stdin.read())
    sampled_files = sample_files(payload)
    print(json.dumps({"sampled_files": sampled_files}))


if __name__ == "__main__":
    main()
