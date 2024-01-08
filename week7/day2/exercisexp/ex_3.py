from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pprint  # To tidy up

options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run Chrome in headless mode
options.add_argument("--no-sandbox")  # Bypass OS security model
options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
driver = webdriver.Chrome(options=options)

url = "https://www.rottentomatoes.com/browse/movies_at_home/affiliates:netflix~critics:certified_fresh"
driver.get(url)

soup = BeautifulSoup(driver.page_source, 'html.parser')
movies = soup.find_all(attrs={"data-qa": "discovery-media-list-item"})

for movie in movies:
    title = [item.get_text() for item in soup.find_all(class_ = 'p--small')]
    score = [item.get_text() for item in soup.find_all(class_ = 'criticsscore')]
    release_date = [item.get_text() for item in soup.find_all(class_ = 'smaller')]

    print(f"Title: {title}")
    print(f"Score: {score}")
    print(f"Release Date: {release_date}")
    print("-------------")


# title = [item.get_text() for item in soup.find_all(class_ = 'p--small')]
# score = [item.get_text() for item in soup.find_all(class_ = 'criticsscore')]
# release_date = [item.get_text() for item in soup.find_all(class_ = 'smaller')]    
# print(f'{title},{score},{release_date}')