import os
import requests
import json
import tqdm
import hashlib
import logging
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

CACHE_DIR = "data/cache"


def cache_response(hash_v, response, show_progress=True):
    """Cache the downloaded json from the khanAcademy API.
    :param: hash_v: the md5 hash of the url, used to create the file name.
    :param: response: the response object for the requests call.
    :param: show progress: bool, whether or not to show progress of the download.
    :return: stream: file stream of the downloaded data retrieved from cache.
    """
    f_name = os.path.join(CACHE_DIR, hash_v)
    if os.path.isfile(f_name):
        response = open(f_name).read()
        return response
    elif response is not None:
        with open(f_name, "w+") as out_f:
            if show_progress:
                response = tqdm.tqdm(response)
            out_f.write("".join(response))
        return open(f_name).read()


def get_data(url, params={"kind": "Video"}, show_progress=False):
    """Download the data from the KhanAcademy API with additional caching.
    :param url: the url of the download request
    :param params: parameters to pass to GET request of the API call.
    :param show_progress: bool, whether to show progress or not.
    :return: tuple, (data, was_cached): data, the downloaded data.
     was_cached: flag to show whether it was retrieved from cache.
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()
    response = cache_response(url_hash, None)
    was_cached = False
    if response:
        data = response
        was_cached = True
    else:
        sess = requests.Session()
        response = sess.get(url, stream=True, params=params)
        data = cache_response(
            url_hash, response.iter_content(chunk_size=1000, decode_unicode=True)
        )
    if show_progress:
        data = tqdm.tqdm(data)
    try:
        data = json.loads("".join(data))
        return data, was_cached
    except ValueError:
        return None, was_cached


def lookup(key, keys, document, with_keys=False):
    """Lookup a key in a nested document, yield a value.
    :param key: the key to look up in the nested json.
    :param keys: the set of keys to retrieve along with the lookup.
    :param document: the nested json to retrieve the data from.
    :param with_keys: bool, whether to retrieve data with extra keys.
    :return: document: the json record with the key looked up for retrieval.
    """
    if isinstance(document, list):
        for d in document:
            for result in lookup(keys, d, with_keys=with_keys):
                yield result

    if isinstance(document, dict):
        for k, v in document.items():
            if key == k and document.get("kind",) == "Video":
                if with_keys:
                    yield {k: document.get(k) for k in keys}
                else:
                    yield v
            if isinstance(v, dict):
                for result in lookup(key, keys, v, with_keys=with_keys):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in lookup(key, keys, d, with_keys=with_keys):
                        yield result


def get_transcripts(record):
    """Fetch the transcript for a video from the KhanAcademy API.
    :param record: the record to retrieve the transcript for.
    :return: tuple(record, was cached) the data in json format
    and whether it was retrieved from cache.
    """
    transcript_template = "http://api.khanacademy.org/api/internal/videos/{}/transcript"
    yt_id = record["youtube_id"]
    transcript_url = transcript_template.format(yt_id)

    transcript_data, was_cached = get_data(
        transcript_url, params={"locale": "en", "lang": "en"}, show_progress=False
    )
    record["transcript"] = transcript_data
    return record, was_cached


def main():
    # URL for the topic tree from KhanAcademy.
    TOPICS_URL = "http://api.khanacademy.org/api/v1/topictree"
    # The keys we will be retrieving data for.
    REQUIRED_KEYS = [
        "title",
        "translated_title",
        "translated_description_html",
        "id",
        "description_html",
        "kind",
        "render_type",
        "relative_url",
        "translated_description",
        "youtube_id",
        "keywords",
    ]
    # Location to store the downloaded data.
    OUTFILE = "data/ka_transcripts_data_with_topics.json"

    # fetch the topic tree from the API
    topics, was_cached = get_data(TOPICS_URL, show_progress=True)
    video_data = lookup("youtube_id", REQUIRED_KEYS, topics, with_keys=True)
    pool = Pool(cpu_count() - 1)
    video_data = pool.imap_unordered(get_transcripts, video_data)
    with open(OUTFILE, "a+") as ouf:
        for i, (item, was_cached) in tqdm.tqdm(enumerate(video_data), total=11000):
            # if not was_cached:
            logging.info(f"Downloading ... {i} and writing new data to file")
            item = json.dumps(item) + "\n"
            ouf.write(item)


if __name__ == "__main__":
    main()
