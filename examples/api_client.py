"""Read statistics from the optional local API using the standard library.

Terminal 1: python app.py --api
Terminal 2: python examples/api_client.py --token TOKEN_PRINTED_IN_TERMINAL_1
"""
import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/api/stats",
        headers={"Authorization": "Bearer " + args.token},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=10) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
