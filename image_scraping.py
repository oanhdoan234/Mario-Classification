import requests
from bs4 import BeautifulSoup
import os


characters = ['Mario', 'Luigi', 'Peach', 'Daisy', 'Yoshi', 'Toad']

for ch in characters:
    # The URL of the gallery page
    url = "https://www.mariowiki.com/Gallery:Mario_Kart_8#Characters"

    # Make a GET request to the URL and store the response
    response = requests.get(url)

    # Parse the HTML content of the response using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all the image tags on the page
    image_tags = soup.find_all("img")

    # Create a directory to store the images
    dirr = ch + "_images"
    if not os.path.exists(dirr):
        os.makedirs(dirr)

    # Loop through the image tags and download the images that contain "Mario" in the filename
    for image_tag in image_tags:
        if ch in image_tag["src"]:
            image_url = image_tag["src"]
            image_name = image_url.split("/")[-1]
            image_path = f"{dirr}/{image_name}"
            image_data = requests.get(image_url).content
            with open(image_path, "wb") as f:
                f.write(image_data)
                print(f"Downloaded {image_name}")
