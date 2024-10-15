import argparse
from src.PipeLine import PipeLine
from src.utils import load_image

def main(img_path: str):
    image = load_image(img_path)
    pipe = PipeLine("config.yaml")
    pipe.predict(image, save=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image.")
    parser.add_argument("img_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    main(args.img_path)