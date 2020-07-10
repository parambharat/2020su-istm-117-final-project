import requests
import json


def get_topic_tree(url):
    sess = requests.Session()
    response = sess.get(url, stream=True, params={"kind": "Video"})
    data = response.iter_content(chunk_size=None, decode_unicode=True)
    data = json.loads("".join(data))
    return data


def _nested_lookup(key, document, keys, wild=False, with_keys=False):
    """Lookup a key in a nested document, yield a value"""
    if isinstance(document, list):
        for d in document:
            for result in _nested_lookup(keys, d, wild=wild, with_keys=with_keys):
                yield result

    if isinstance(document, dict):
        for k, v in document.items():
            if key == k and document.get("kind",) == "Video":
                if with_keys:
                    yield {k: document.get(k) for k in keys}
                else:
                    yield v
            if isinstance(v, dict):
                for result in _nested_lookup(
                    key, v, keys, wild=wild, with_keys=with_keys
                ):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in _nested_lookup(
                        key, d, keys, wild=wild, with_keys=with_keys
                    ):
                        yield result


def main():
    URL = "http://api.khanacademy.org/api/v1/topictree"
    topics = get_topic_tree(URL)
    print(topics)


if __name__ == "__main__":
    main()
