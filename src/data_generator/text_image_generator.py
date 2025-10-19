"""
Text Image Generator Module
Automatically generates text images with bounding box annotations
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm


class TextImageGenerator:
    """Generate synthetic text images with ground truth annotations"""

    def __init__(
        self,
        fonts_dir: str = None,
        text_sources: List[str] = None,
        image_size: Tuple[int, int] = (800, 1000),
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        """
        Initialize TextImageGenerator

        Args:
            fonts_dir: Directory containing font files
            text_sources: List of text strings to render
            image_size: Output image dimensions (width, height)
            bg_color: Background color RGB
            text_color: Text color RGB
        """
        self.image_size = image_size
        self.bg_color = bg_color
        self.text_color = text_color

        # Load fonts
        self.fonts = self._load_fonts(fonts_dir)

        # Load or generate text sources
        self.text_sources = text_sources if text_sources else self._default_texts()

    def _load_fonts(self, fonts_dir: Optional[str]) -> List[ImageFont.FreeTypeFont]:
        """Load available fonts"""
        fonts = []
        font_sizes = [16, 18, 20, 22, 24]

        # Try to load system fonts
        system_fonts = [
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',  # macOS
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # macOS
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # Linux
            'C:\\Windows\\Fonts\\malgun.ttf',  # Windows
        ]

        for font_path in system_fonts:
            if os.path.exists(font_path):
                for size in font_sizes:
                    try:
                        fonts.append(ImageFont.truetype(font_path, size))
                    except:
                        pass

        # Fallback to default font
        if not fonts:
            for size in font_sizes:
                fonts.append(ImageFont.load_default())

        return fonts

    def _default_texts(self) -> List[str]:
        """Generate default Korean text samples"""
        return [
            "컴퓨터 비전은 이미지와 비디오를 분석하여 의미 있는 정보를 추출하는 인공지능 분야입니다.",
            "딥러닝 기술의 발전으로 객체 인식, 이미지 분할, 얼굴 인식 등이 가능해졌습니다.",
            "OCR 기술은 문서 이미지에서 텍스트를 자동으로 추출하는 중요한 응용 분야입니다.",
            "형광펜으로 표시된 중요한 내용을 디지털화하면 학습 효율이 크게 향상됩니다.",
            "OpenCV는 실시간 컴퓨터 비전을 위한 오픈소스 라이브러리입니다.",
            "이미지 전처리는 노이즈 제거, 이진화, 대비 향상 등을 포함합니다.",
            "색공간 변환은 RGB에서 HSV, Lab 등으로 변환하여 색상 분석을 용이하게 합니다.",
            "모폴로지 연산은 이미지의 형태학적 특징을 분석하는 기법입니다.",
            "바운딩 박스는 객체의 위치를 사각형 영역으로 표현하는 방법입니다.",
            "Intersection over Union은 객체 검출 성능을 평가하는 지표입니다.",
            "데이터 증강은 학습 데이터의 다양성을 높여 모델 성능을 향상시킵니다.",
            "Character Error Rate는 OCR 성능을 평가하는 핵심 지표입니다.",
            "합성 데이터 생성은 라벨링 비용을 절감하고 대규모 데이터셋을 구축합니다.",
            "텍스트 감지와 텍스트 인식은 OCR 파이프라인의 두 가지 주요 단계입니다.",
            "전이 학습을 통해 사전 학습된 모델을 새로운 작업에 적용할 수 있습니다."
        ]

    def generate_text_image(
        self,
        text: str = None,
        font: ImageFont.FreeTypeFont = None,
        line_spacing: int = 10,
        margin: int = 50
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        Generate a single text image with bounding boxes

        Args:
            text: Text to render (random if None)
            font: Font to use (random if None)
            line_spacing: Spacing between lines
            margin: Image margin

        Returns:
            Tuple of (image, annotations)
            annotations: List of {text, bbox: [x, y, w, h]}
        """
        # Select random text and font if not provided
        if text is None:
            text = random.choice(self.text_sources)
        if font is None:
            font = random.choice(self.fonts)

        # Create blank image
        img = Image.new('RGB', self.image_size, self.bg_color)
        draw = ImageDraw.Draw(img)

        # Split text into words
        words = text.split()

        # Render text with word wrapping
        annotations = []
        x, y = margin, margin
        max_width = self.image_size[0] - 2 * margin

        current_line = []
        current_line_words = []

        for word in words:
            # Test if adding this word exceeds max width
            test_line = ' '.join(current_line_words + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]

            if text_width <= max_width:
                current_line_words.append(word)
            else:
                # Render current line
                if current_line_words:
                    line_text = ' '.join(current_line_words)
                    self._render_line(draw, line_text, x, y, font, annotations)

                    # Move to next line
                    line_height = bbox[3] - bbox[1]
                    y += line_height + line_spacing

                # Start new line with current word
                current_line_words = [word]

        # Render last line
        if current_line_words:
            line_text = ' '.join(current_line_words)
            self._render_line(draw, line_text, x, y, font, annotations)

        return img, annotations

    def _render_line(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        font: ImageFont.FreeTypeFont,
        annotations: List[Dict]
    ):
        """Render a line of text and record word-level bounding boxes"""
        words = text.split()
        current_x = x

        for word in words:
            # Get word bounding box
            bbox = draw.textbbox((current_x, y), word, font=font)
            word_width = bbox[2] - bbox[0]
            word_height = bbox[3] - bbox[1]

            # Draw text
            draw.text((current_x, y), word, font=font, fill=self.text_color)

            # Record annotation
            annotations.append({
                'text': word,
                'bbox': [current_x, y, word_width, word_height]
            })

            # Move to next word position
            space_width = draw.textbbox((0, 0), ' ', font=font)[2]
            current_x += word_width + space_width

    def generate_batch(
        self,
        output_dir: str,
        num_images: int = 100,
        save_annotations: bool = True
    ) -> List[str]:
        """
        Generate batch of text images

        Args:
            output_dir: Directory to save images
            num_images: Number of images to generate
            save_annotations: Save annotations as JSON

        Returns:
            List of generated image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        all_annotations = []

        for i in tqdm(range(num_images), desc="Generating images"):
            # Generate image
            img, annotations = self.generate_text_image()

            # Save image
            image_name = f"text_{i:04d}.png"
            image_path = os.path.join(output_dir, image_name)
            img.save(image_path)
            image_paths.append(image_path)

            # Store annotations
            all_annotations.append({
                'image_id': i,
                'image_name': image_name,
                'image_path': image_path,
                'annotations': annotations
            })

        # Save annotations
        if save_annotations:
            annotations_path = os.path.join(output_dir, 'annotations.json')
            with open(annotations_path, 'w', encoding='utf-8') as f:
                json.dump(all_annotations, f, ensure_ascii=False, indent=2)
            print(f"Saved annotations to {annotations_path}")

        return image_paths


if __name__ == "__main__":
    # Test generation
    generator = TextImageGenerator()

    # Generate single image
    img, annotations = generator.generate_text_image()
    img.save("test_output.png")
    print(f"Generated test image with {len(annotations)} word annotations")

    # Generate batch
    generator.generate_batch("data/synthetic", num_images=10)
