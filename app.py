# app.py
from flask import Flask, render_template
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time

app = Flask(__name__)

def load_text_file(filename):
    """Load text content from a file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def load_fonts(filename):
    """Load font names from a file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

@app.route('/')
def home():
    """Landing page with links to all fonts"""
    fonts = load_fonts('fonts.txt')
    return render_template('index.html', fonts=fonts)

@app.route('/font/<font_name>')
def show_font(font_name):
    """Page for individual font"""
    lorem_text = load_text_file('lorem_ipsum.txt')
    font_data = {
        'name': font_name,
        'url': f'https://fonts.googleapis.com/css2?family={font_name.replace(" ", "+")}&display=swap'
    }
    return render_template('templates/single_font.html', font=font_data, text=lorem_text)

def setup_webdriver():
    """Configure Chrome WebDriver with appropriate options"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-gpu')
    return webdriver.Chrome(options=chrome_options)

def capture_screenshots(output_dir='font-images'):
    """Capture screenshots for each font"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fonts = load_fonts('fonts.txt')
    driver = setup_webdriver()
    base_url = 'http://localhost:5000'

    try:
        for font in fonts:
            print(f"Processing font: {font}")
            
            # Navigate to the font's page
            font_url = f"{base_url}/font/{font.replace(' ', '%20')}"
            driver.get(font_url)
            
            # Wait for font to load
            time.sleep(2)  # Give time for font to load
            
            # Find and screenshot the text block
            element = driver.find_element(By.CLASS_NAME, 'text-block')
            
            # Clean filename
            filename = f"{font.replace(' ', '_').lower()}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Take screenshot
            element.screenshot(filepath)
            print(f"Saved screenshot to {filepath}")
            
    finally:
        driver.quit()

if __name__ == '__main__':
    # Run Flask app in a separate thread
    from threading import Thread
    server = Thread(target=lambda: app.run(debug=False))
    server.daemon = True
    server.start()
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Start capturing screenshots
    capture_screenshots()