from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def extract_text_from_website(url, wait_time=5):
    
    # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--log-level=3")  # Suppress logs

    driver = webdriver.Chrome(options=options)
    
    try:
        print(f"Loading {url} ...")
        driver.get(url)
        time.sleep(wait_time)  # Wait for JS to load
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        title_tag = soup.find("title")
        # Remove script and style elements
        for element in soup(["script", "style", "noscript", "iframe"]):
            element.decompose()

        # Get only visible text
        visible_text = soup.get_text(separator="\n", strip=True)

        # Optional: filter out short or garbage lines
        lines = [line for line in visible_text.splitlines()]

        title = title_tag.get_text(strip=True) if title_tag else "No title found"
        return title, "\n".join(lines)

    finally:
        driver.quit()




