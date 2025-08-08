import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import os
import json
from typing import TypedDict, Optional
import base64

from PIL import Image


def download_image(img_url, save_dir, page_url):
    image_paths = []
    try:
        img_data = requests.get(img_url, timeout=10).content
        
        filename = os.path.join(save_dir, os.path.basename(img_url.split("?")[0]))
        image_paths.append(filename)
        with open(filename, "wb") as f:
            f.write(img_data)
        print(f"🖼️  Downloaded hero image from: {page_url}")
    except Exception as e:
        print(f"⚠️  Failed to download image: {img_url} — {e}")
    return image_paths


def scrape_pages(url_list, image_dir="hero_images"):
    corpus = ""
    

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    os.makedirs(image_dir, exist_ok=True)

    for url in url_list:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "lxml")

            # Remove junk
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()

            # ✅ Extract visible text
            text = soup.get_text(separator=' ', strip=True)
            corpus += text + "\n\n"

            # ✅ Extract hero image (first image as default)
            hero_img_tag = soup.find("img")
            if hero_img_tag:
                src = hero_img_tag.get("src")
                if src:
                    full_img_url = urljoin(url, src)
                    image_paths = download_image(full_img_url, image_dir, url)

            print(f"✅ Scraped: {url}")

        except Exception as e:
            print(f"❌ Failed: {url} — {e}")

    return corpus,image_paths


