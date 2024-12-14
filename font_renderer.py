from pathlib import Path
from typing import List
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import os
import shutil

from flask import Flask, render_template
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FontConfig:
    """Configuration for font rendering"""
    name: str
    google_fonts_url: str
    output_path: Path
    image_width: int = 512
    image_height: int = 512
    font_size: int = 24

class FontRenderer:
    def __init__(self, 
                 fonts_file: str = 'fonts.txt',
                 text_file: str = 'lorem_ipsum.txt',
                 output_dir: str = 'font-images',
                 template_dir: str = 'templates',
                 port: int = 5000,
                 max_workers: int = 4,
                 scroll_height: int = 512):
        self.fonts = self._load_fonts(fonts_file)
        self.text = self._load_text(text_file)
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)
        self.port = port
        self.max_workers = max_workers
        self.scroll_height = scroll_height
        self.app = self._create_flask_app()
        
        # Ensure output directory exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_fonts(self, filename: str) -> List[str]:
        """Load and validate font names from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                fonts = [line.strip() for line in f if line.strip()]
            if not fonts:
                raise ValueError("No fonts found in fonts file")
            logger.info(f"Loaded {len(fonts)} fonts from {filename}")
            return fonts
        except FileNotFoundError:
            raise FileNotFoundError(f"Fonts file not found: {filename}")

    def _load_text(self, filename: str) -> str:
        """Load and validate text content"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                raise ValueError("Text file is empty")
            logger.info(f"Loaded {len(text)} characters of text from {filename}")
            return text * 10  # Repeat text to ensure enough content for scrolling
        except FileNotFoundError:
            raise FileNotFoundError(f"Text file not found: {filename}")

    def _setup_webdriver(self) -> webdriver.Chrome:
        """Configure Chrome WebDriver with optimized settings"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        return webdriver.Chrome(options=chrome_options)

    def _create_flask_app(self) -> Flask:
        """Create and configure Flask application"""
        app = Flask(__name__, template_folder=str(self.template_dir.absolute()))
        
        @app.route('/font/<font_name>')
        def render_font(font_name):
            font_config = FontConfig(
                name=font_name,
                google_fonts_url=f'https://fonts.googleapis.com/css2?family={font_name.replace(" ", "+")}&display=swap',
                output_path=self.output_dir / font_name.lower().replace(' ', '_')
            )
            return render_template(
                'single_font.html',
                font=font_config,
                text=self.text
            )
        
        return app

    def _capture_font_screenshots(self, font: str, num_samples: int = 1000) -> None:
        """Capture screenshots for a single font with scrolling"""
        driver = self._setup_webdriver()
        font_dir = self.output_dir / font.lower().replace(' ', '_')
        font_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load the page just once
            url = f"http://localhost:{self.port}/font/{font.replace(' ', '%20')}"
            driver.get(url)
            
            # Wait for font to load
            wait = WebDriverWait(driver, 10)
            text_block = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, 'text-block'))
            )
            
            # Take screenshots with scrolling
            scroll_position = 0
            for i in range(num_samples):
                try:
                    # Scroll to new position
                    driver.execute_script(f"document.querySelector('.text-block').scrollTop = {scroll_position};")
                    time.sleep(0.1)  # Small delay to let scroll complete
                    
                    # Take screenshot
                    filename = font_dir / f"sample_{i:04d}.png"
                    text_block.screenshot(str(filename))
                    
                    # Update scroll position
                    scroll_position += self.scroll_height
                    
                    # Log progress
                    if i % 100 == 0:
                        logger.info(f"Generated {i} samples for font {font}")
                        
                except Exception as e:
                    logger.error(f"Error capturing screenshot {i} for font {font}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing font {font}: {e}")
        finally:
            driver.quit()

    def generate_dataset(self) -> None:
        """Generate the complete font dataset using multiple threads"""
        logger.info(f"Starting dataset generation for {len(self.fonts)} fonts")
        
        # Start Flask server in a separate thread
        from threading import Thread
        server = Thread(target=lambda: self.app.run(port=self.port, debug=False))
        server.daemon = True
        server.start()
        time.sleep(2)  # Give server time to start
        
        # Process fonts in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self._capture_font_screenshots, self.fonts)
        
        logger.info("Dataset generation complete")

def main():
    # Initialize and run the font renderer
    renderer = FontRenderer(
        fonts_file='fonts.txt',
        text_file='lorem_ipsum.txt',
        output_dir='font-images',
        template_dir='templates',
        max_workers=4,
        scroll_height=512
    )
    
    try:
        renderer.generate_dataset()
    except KeyboardInterrupt:
        logger.info("Dataset generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")

if __name__ == "__main__":
    main()