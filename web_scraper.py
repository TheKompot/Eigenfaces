import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

def scrape_images(url, delay):
    # Make a request to the starting webpage
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table with the class "bw-borderless"
    table = soup.find('table', class_='bw-borderless')
    if table is None:
        return []

    # Extract the URLs from the table cells
    urls = [a['href'] for a in table.find_all('a')]

    # Initialize an empty dict to store the images
    images = {}

    # Iterate through the URLs
    for url in urls:
        # Make a request to the URL
        response = requests.get(url)
        # Pause for the specified delay
        time.sleep(delay)
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find face_photo on the webpage
        face_photo = soup.find('aside', class_='span3')
        if face_photo is not None:
            face_photo = face_photo.find('img')
        # Add the face_photo to the list
        if face_photo is not None:
            images[url.split('/')[-1]] = face_photo['src']
        # if len(images) > 2:
        #     break


    # Combine the base URL of the starting webpage with the relative URLs of the images to create a list of fully-qualified URLs
    return {key:urljoin(url, image) for key, image in images.items()}

def scrape(url, delay, stop_url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pracoviska = soup.find('article', class_='span6')
    urls_to_scrape = [urljoin(url, a['href']) for a in pracoviska.find_all("a") if a.parent.name == 'li']

    # Take only teachers and remove duplicate links
    urls_to_scrape = set(urls_to_scrape[:urls_to_scrape.index(stop_url)])

    all_images = {}
    for url in urls_to_scrape:
        # print(url)
        all_images = {**all_images, **scrape_images(url, delay)}
        # all_images |= scrape_images(url, delay)
        # if len(all_images) > 2:
        #     break
    return all_images


# Example usage
# image_urls = scrape_images('https://fmph.uniba.sk/pracoviska/katedra-aplikovanej-informatiky/', 0.1)
# print(image_urls)
if __name__ == "__main__":
    image_urls = scrape("https://fmph.uniba.sk/pracoviska/", 0, 'https://fmph.uniba.sk/pracoviska/detasovane-pracovisko-turany/')
    # image_urls_en = scrape("https://fmph.uniba.sk/en/departments/", 0, 'https://fmph.uniba.sk/en/departments/library-and-publishing-services/')
    # print(image_urls==image_urls_en)