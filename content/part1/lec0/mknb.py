# this script was developed with the help of OpenAI GPT-4.1 mini
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell

def create_notebook_with_images_as_rise_slides(image_dir, output_notebook=None):
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
    images = [f for f in os.listdir(image_dir)
              if os.path.splitext(f)[1].lower() in image_extensions]

    images.sort()

    # If notebook output filename is not provided, generate from directory name
    if output_notebook is None:
        notebook_name = os.path.basename(os.path.normpath(image_dir)) + '.ipynb'
        output_notebook = os.path.join('.', notebook_name)

    nb = new_notebook()

    cells = []
    for img in images:
        img_path = os.path.join(image_dir, img)
        relative_path = os.path.relpath(img_path, os.path.dirname(output_notebook))
        md = f"![{img}]({relative_path})"
        
        cell = new_markdown_cell(md)
        cell.metadata['slideshow'] = {'slide_type': 'slide'}
        
        cells.append(cell)

    nb['cells'] = cells

    with open(output_notebook, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"RISE slide notebook created: {output_notebook}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a RISE slideshow notebook displaying images from a directory.")
    parser.add_argument("image_dir", help="Directory containing image files")
    parser.add_argument("-o", "--output", help="Output notebook filename (optional)")
    args = parser.parse_args()

    create_notebook_with_images_as_rise_slides(args.image_dir, args.output)
