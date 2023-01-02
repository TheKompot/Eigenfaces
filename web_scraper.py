import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

def scrape_images(url, delay):
    '''
    Scrape images of people from given url, works for websites with similar HTML structure like
     'https://fmph.uniba.sk/pracoviska/katedra-aplikovanej-informatiky/' (function needs some adjustments
      for other types of websites)

    :param url: url of some department with list of people
    :param delay: set some delay in seconds (e.g. 1) if website locks you out
    :return: dict (keys are the last parts of URLs, values are URLs containing images)
    '''
    # Make a request to the starting webpage
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table with the class "bw-borderless" (list of teachers)
    table = soup.find('table', class_='bw-borderless')
    if table is None:
        return []

    # Extract the URLs from the table cells (URLs of teachers)
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

        # Just for testing purposes
        # if len(images) > 2:
        #     break


    # Combine the base URL of the starting webpage with the relative URLs of the images to create a list of fully-qualified URLs
    return {key:urljoin(url, image) for key, image in images.items()}

def scrape(url, delay, stop_url=None):
    '''
    Scrape images of people from given url, uses function "scrape_images(url, delay)", works for websites
     with similar HTML structure like 'https://fmph.uniba.sk/pracoviska/'
     (function needs some adjustments for other types of websites)

    :param url: url with list of departments
    :param delay: set some delay in seconds (e.g. 1) if website locks you out
    :param stop_url: url of the last department to scrape images from (if None, scrape all departments)
    :return: dict (keys are the last parts of URLs, values are URLs containing images)
    '''
    # Make a request to the starting webpage
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the urls of departments
    pracoviska = soup.find('article', class_='span6')
    urls_to_scrape = [urljoin(url, a['href']) for a in pracoviska.find_all("a") if a.parent.name == 'li']

    # Take only departments till stop_url and remove duplicate links
    if stop_url is not None:
        urls_to_scrape = set(urls_to_scrape[:urls_to_scrape.index(stop_url)+1])

    all_images = {}
    for url in urls_to_scrape:
        all_images = {**all_images, **scrape_images(url, delay)}

        # Just for testing purposes
        # if len(all_images) > 2:
        #     break

    return all_images


# Example usage of function image_urls
# image_urls = scrape_images('https://fmph.uniba.sk/pracoviska/katedra-aplikovanej-informatiky/', 0.1)
# print(image_urls)

if __name__ == "__main__":
    image_urls = scrape("https://fmph.uniba.sk/pracoviska/", 0, 'https://fmph.uniba.sk/pracoviska/katedra-telesnej-vychovy-a-sportu/')

    # As it turns out, english version has same images
    # image_urls_en = scrape("https://fmph.uniba.sk/en/departments/", 0, 'https://fmph.uniba.sk/en/departments/department-of-physical-education/')
    # print(image_urls==image_urls_en)  # Output: True

    with open('images.txt', 'w') as f:
        print(image_urls, file=f)
