import json
from multiprocessing import Pool
import tqdm


def flatten_transcript(line):
    """Flatten the transcript text retrieved from the KhanAcademy API.

    :param line: A single record in dict format.
    :return: line, dict, the transcript in flattened format seperated by `\n`.
    """
    if line:
        transcript_text = " ".join(map(lambda x: x["text"], line["transcript"]))
        transcript_text = " ".join(transcript_text.split())
        line["transcript_text"] = transcript_text
        return line


def filter_keys(line):
    if line:
        keys = [
            "id",
            "transcript_text",
            "title",
            "description_html",
            "youtube_id",
        ]
        return dict(filter(lambda x: x[0] in keys, line.items()))


def enhance_doc(line):
    """Flatten a khanAcademy transcript and perform basic preprocessing.

    :param line: a single record in text format.
    :return: a single record in json format with the flattened transcript.
    """
    try:
        line = json.loads(line)
        line = flatten_transcript(line)
        line = filter_keys(line)
        return line
    except Exception:
        return None


def main():
    pool = Pool(10)
    data_f = open("data/ka_transcripts_data.json").read()
    lines = data_f.split("\n")
    lines = pool.imap_unordered(enhance_doc, lines, chunksize=100)

    outfile = "data/ka_transcripts_preprocessed.json"
    with open(outfile, "w+") as ouf:
        for line in tqdm.tqdm(lines, total=13000):
            if line:
                line = json.dumps(line) + "\n"
                ouf.write(line)


if __name__ == "__main__":
    main()
