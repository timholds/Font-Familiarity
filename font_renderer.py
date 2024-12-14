from pathlib import Path
from typing import List
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import os
import signal
import sys

from flask import Flask, render_template
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('font_renderer.log')
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    global running
    logger.info("Received shutdown signal. Completing current font before exiting...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

class FontRenderer:
    def __init__(self, 
                 fonts_file: str = 'fonts.txt',
                 text_file: str = 'lorem_ipsum.txt',
                 output_dir: str = 'font-images',
                 template_dir: str = 'templates',
                 port: int = 5000,
                 resume_from: str = None):
        # Set core parameters first
        self.scroll_height = 400
        self.port = port
        self.resume_from = resume_from
        
        # Then load files and create directories
        self.fonts = self._load_fonts(fonts_file)
        self.text = self._load_text(text_file)
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)
        self.flask_app = None
        self.server_thread = None
        
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)


    def _load_text(self, filename: str) -> str:
        """Load and validate text content"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                base_text = f.read().strip()
            if not base_text:
                raise ValueError("Text file is empty")
                
            # Calculate needed repetitions
            words_per_line = 10  # approximate
            pixels_per_line = 36  # 24px font * 1.5 line height
            total_scroll_pixels = self.scroll_height * 1000  # 1000 samples
            needed_lines = total_scroll_pixels / pixels_per_line
            needed_words = needed_lines * words_per_line
            words_in_base = len(base_text.split())
            repetitions = int((needed_words / words_in_base) * 1.5)  # Add 50% buffer
            
            text = base_text * repetitions
            logger.info(f"Text stats: {len(text.split())} words, {len(text)} chars")
            return text
        except FileNotFoundError:
            raise FileNotFoundError(f"Text file not found: {filename}")

    def _load_fonts(self, filename: str) -> List[str]:
        """Load and validate font names from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                # Strip whitespace and filter out empty lines
                fonts = [line.strip() for line in f if line.strip()]
            if not fonts:
                raise ValueError("No fonts found in fonts file")
            logger.info(f"Loaded {len(fonts)} fonts from {filename}")
            return fonts
        except FileNotFoundError:
            raise FileNotFoundError(f"Fonts file not found: {filename}")


    def _setup_webdriver(self) -> webdriver.Chrome:
        """Configure Chrome WebDriver with optimized settings"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        service = Service()
        return webdriver.Chrome(service=service, options=chrome_options)

    def start_flask(self):
        """Start Flask server in a controlled way"""
        if self.flask_app is None:
            self.flask_app = Flask(__name__, template_folder=str(self.template_dir.absolute()))
            
            @self.flask_app.route('/font/<font_name>')
            def render_font(font_name):
                return render_template(
                    'single_font.html',
                    font={'name': font_name},
                    text=self.text
                )
            
            from threading import Thread
            self.server_thread = Thread(target=lambda: self.flask_app.run(port=self.port, debug=False))
            self.server_thread.daemon = True
            self.server_thread.start()
            time.sleep(2)  # Wait for server to start
            logger.info("Flask server started")

    def get_completed_fonts(self) -> set:
        """Get list of fonts that have already been processed"""
        completed = set()
        for font_dir in self.output_dir.iterdir():
            if font_dir.is_dir():
                # Convert directory name back to font name format
                font_name = font_dir.name.replace('_', ' ').lower()
                # Check if directory has 1000 images
                if len(list(font_dir.glob('*.png'))) >= 1000:
                    completed.add(font_name)
        return completed

    def generate_dataset(self) -> None:
        logger.info(f"Starting dataset generation for {len(self.fonts)} fonts")
        self.start_flask()
        
        # Get completed fonts
        completed_fonts = self.get_completed_fonts()
        logger.info(f"Found {len(completed_fonts)} already completed fonts")
        
        # Determine starting point
        start_idx = 0
        if self.resume_from:
            logger.info(f"Attempting to resume from font: '{self.resume_from}'")
            
            # Normalize the resume font name and current fonts list
            resume_font_norm = self.resume_from.strip().lower()
            font_lookup = {font.strip().lower(): idx for idx, font in enumerate(self.fonts)}
            
            if resume_font_norm in font_lookup:
                start_idx = font_lookup[resume_font_norm]
                logger.info(f"Found resume point. Starting from index {start_idx} ({self.fonts[start_idx]})")
            else:
                logger.error(f"Could not find font '{self.resume_from}' in fonts list")
                logger.info("First few available fonts: " + ", ".join(self.fonts[:5]))
                logger.info(f"Resume font (normalized): '{resume_font_norm}'")
                logger.info(f"Available normalized fonts (first 5): " + 
                          ", ".join(f"'{f.strip().lower()}'" for f in self.fonts[:5]))
                return
        
        fonts_to_process = self.fonts[start_idx:]
        total = len(fonts_to_process)
        logger.info(f"Will process {total} fonts starting with: {fonts_to_process[0]}")
        
        for idx, font in enumerate(fonts_to_process, 1):
            if not running:
                logger.info("Received shutdown signal, stopping gracefully...")
                break
                
            font_norm = font.strip().lower()
            if font_norm in completed_fonts:
                logger.info(f"Skipping already completed font {font}")
                continue
                
            try:
                logger.info(f"Processing font {idx}/{total}: {font} (overall {start_idx + idx}/{len(self.fonts)})")
                self._capture_font_screenshots(font)
                logger.info(f"Completed font {font}")
                
                # Save last completed font
                with open('last_completed_font.txt', 'w') as f:
                    f.write(font)
                    
            except Exception as e:
                logger.error(f"Failed to process font {font}: {e}", exc_info=True)
                with open('failed_fonts.txt', 'a') as f:
                    f.write(f"{font}\n")
                continue
        
        logger.info("Dataset generation complete")

    def _capture_font_screenshots(self, font: str, num_samples: int = 1000) -> None:
        """Capture screenshots for a single font with scrolling"""
        driver = None
        try:
            driver = self._setup_webdriver()
            font_dir = self.output_dir / font.lower().replace(' ', '_')
            font_dir.mkdir(parents=True, exist_ok=True)
            
            url = f"http://localhost:{self.port}/font/{font.replace(' ', '%20')}"
            logger.info(f"Loading URL: {url}")
            driver.get(url)
            
            wait = WebDriverWait(driver, 10)
            text_block = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, 'text-block'))
            )
            
            scroll_position = 0
            for i in range(num_samples):
                if not running:
                    break
                    
                try:
                    driver.execute_script(
                        f"document.querySelector('.text-block').scrollTop = {scroll_position};"
                    )
                    time.sleep(0.01)
                    
                    filename = font_dir / f"sample_{i:04d}.png"
                    text_block.screenshot(str(filename))
                    scroll_position += self.scroll_height
                    
                    if i % 100 == 0:
                        logger.info(f"Generated {i}/{num_samples} samples for font {font}")
                        
                except Exception as e:
                    logger.error(f"Error capturing screenshot {i} for font {font}: {e}")
                    raise
                    
        finally:
            if driver:
                driver.quit()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', help='Resume from specific font')
    args = parser.parse_args()

    renderer = FontRenderer(
        fonts_file='full_fonts_list.txt',
        text_file='lorem_ipsum.txt',
        output_dir='font-images',
        template_dir='templates',
        resume_from=args.resume_from
    )
    
    try:
        renderer.generate_dataset()
    except KeyboardInterrupt:
        logger.info("Dataset generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}", exc_info=True)
    finally:
        logger.info("Process complete")

if __name__ == "__main__":
    main()