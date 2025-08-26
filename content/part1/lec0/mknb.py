# this script was developed with the help of OpenAI GPT-4.1 mini
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell
import json

def create_notebook_with_images_as_rise_slides(image_dir, output_notebook=None, notebook_meta=None):
    """
    Create a Jupyter notebook with RISE slideshow markdown cells displaying images in a directory.

    Parameters:
    - image_dir: Directory containing image files
    - output_notebook: Path to output notebook file (optional)
    - notebook_meta: Dictionary of notebook metadata (optional)
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
    images = [f for f in os.listdir(image_dir)
              if os.path.splitext(f)[1].lower() in image_extensions]

    images.sort()

    # If notebook output filename is not provided, generate from directory name
    if output_notebook is None:
        notebook_name = os.path.basename(os.path.normpath(image_dir)) + '.ipynb'
        output_notebook = os.path.join('.', notebook_name)

    nb = new_notebook()

    # Apply notebook metadata if provided
    if notebook_meta is not None:
        nb.metadata.update(notebook_meta)

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
    parser.add_argument(
        "-m", "--metadata",
        help=(
            "JSON string or path to JSON file providing notebook metadata "
            "(optional). For example: '{\"kernelspec\": {\"name\": \"python3\"}}'"
        )
    )

    args = parser.parse_args()

    notebook_metadata = None

    # Priority: command line metadata > nb.json file > None
    if args.metadata:
        # Try to load metadata from file, fallback to JSON string
        if os.path.isfile(args.metadata):
            with open(args.metadata, 'r', encoding='utf-8') as f:
                notebook_metadata = json.load(f)
        else:
            try:
                notebook_metadata = json.loads(args.metadata)
            except json.JSONDecodeError as e:
                print(f"Error parsing notebook metadata JSON: {e}")
                exit(1)
    else:
        # If no metadata argument provided, try to load nb.json from current directory
        nb_json_path = os.path.join(os.getcwd(), "nb.json")
        if os.path.isfile(nb_json_path):
            try:
                with open(nb_json_path, 'r', encoding='utf-8') as f:
                    notebook_metadata = json.load(f)
                print("Loaded notebook metadata from nb.json")
            except Exception as e:
                print(f"Error loading nb.json metadata file: {e}")
                exit(1)

    create_notebook_with_images_as_rise_slides(args.image_dir, args.output, notebook_metadata)
