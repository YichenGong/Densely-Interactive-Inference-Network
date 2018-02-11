#!/usr/bin/env python

"""
Datasets downloader, inspired by: https://github.com/lukecq1231/nli/blob/master/data/download.py
"""
import os
import zipfile
import urllib.parse
import requests


def unzip(file_name):
    with zipfile.ZipFile(file_name) as archive:
        if not all(os.path.exists(name) for name in archive.namelist()):
            print("Extracting: " + file_name)
            archive.extractall()
        else:
            print("Content exists, extraction skipped.")


def download(url):
    file_name = os.path.basename(urllib.parse.urlparse(url).path)
    if os.path.exists(file_name):
        print("Archive {} exists, skipping ...".format(file_name))
    else:
        os.system('wget --no-check-certificat ' + url)
    return file_name


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    if os.path.exists(destination):
        print("File {} exists, skipping ...".format(destination))
        return

    print("Downloading {} ...".format(destination))
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))

    snli_url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    mnli_url = 'https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
    vector_url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'

    unzip(download(snli_url))
    unzip(download(mnli_url))
    unzip(download(vector_url))

    proprocessed_shared_file_id = "0B6CTyAhSHoJTa3ZSSE5QQUJrb3M"
    download_file_from_google_drive(proprocessed_shared_file_id, "shared.jsonl") 

    # https://drive.google.com/file/d/1MddwhXZvsWSZZsvL_5oaOsJZFOshG6Dj/view?usp=sharing 
    # multinli_0.9_test_matched_unlabeled.jsonl
    # https://drive.google.com/file/d/1mzJFQ-BgkXJEr-fjv6L0t7SryeF546Cl/view?usp=sharing
    # multinli_0.9_test_mismatched_unlabeled.jsonl 
    multinli_matched_test = "1MddwhXZvsWSZZsvL_5oaOsJZFOshG6Dj" 
    download_file_from_google_drive(multinli_matched_test, "multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl") 
    multinli_mismatched_test = "1mzJFQ-BgkXJEr-fjv6L0t7SryeF546Cl"
    download_file_from_google_drive(multinli_mismatched_test, "multinli_0.9/multinli_0.9_test_mismatched_unlabeled.jsonl") 
    print("Done.")
