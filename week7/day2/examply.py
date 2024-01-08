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
url = "https://quotes.toscrape.com/"
driver.get(url)

# Wait for the page to load
wait = WebDriverWait(driver, 10)

# Extract quotes details
quotes_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.quote')))

# Extract quotes details
quotes_list = []
for quote_elem in quotes_elements:
    text = quote_elem.find_element(By.CLASS_NAME, 'text').text
    author = quote_elem.find_element(By.CLASS_NAME, 'author').text
    tags = quote_elem.find_elements(By.CLASS_NAME, 'tag')
    tag_list = [tag.text for tag in tags]

    quotes_list.append({
        'author': author,
        'text': text,
        'tags': tag_list,
    })

pprint.pprint(quotes_list)
driver.quit()
