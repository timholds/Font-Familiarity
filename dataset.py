from typing import List, Tuple

class FontDataset:
    def __init__(self, text_samples: List[str], fonts_file='fonts.txt'):
        self.fonts = self.load_fonts(fonts_file)
        self.text_samples = text_samples

    def load_fonts(self, fonts_file):
        with open(fonts_file, 'r') as file:
            fonts = [line.strip() for line in file.readlines()]
        return fonts

    def get_fonts(self):
        return self.fonts
    
    def __len__(self):
        return len(self.fonts)
    
    def __getitem__(self, idx) -> Tuple[str, str, int]:
        """
        Returns:
            font_name: Name of the font
            text: Text sample to render
            font_idx: Index of the font in the dataset
        """
        font_idx = idx // len(self.text_samples)
        text_idx = idx % len(self.text_samples)
        
        return self.font_names[font_idx], self.text_samples[text_idx], font_idx
    




# class FontDataset(Dataset):
#     def __init__(self, font_paths: List[str], text_samples: List[str]):
#         self.font_paths = font_paths
#         self.text_samples = text_samples
#         self.transforms = transforms.Compose([
#             transforms.Resize((64, 512)),
#             transforms.ToTensor(),
#         ])
        
#         # Create font_to_idx mapping
#         self.font_to_idx = {path: idx for idx, path in enumerate(font_paths)}
        
#     def __len__(self):
#         return len(self.font_paths) * len(self.text_samples)
    
    # def __getitem__(self, idx):
    #     font_idx = idx // len(self.text_samples)
    #     text_idx = idx % len(self.text_samples)
        
    #     font_path = self.font_paths[font_idx]
    #     text = self.text_samples[text_idx]
        
    #     # Create image with rendered text
    #     img = Image.new('L', (512, 64), color='white')
    #     try:
    #         font = ImageFont.truetype(font_path, 48)
    #         draw = ImageDraw.Draw(img)
    #         draw.text((10, 10), text, font=font, fill='black')
    #     except Exception as e:
    #         print(f"Error loading font {font_path}: {e}")
    #         # Return a blank image if font loading fails
    #         pass
        
    #     return self.transforms(img), font_idx
    