from flashkeras.utils.otherimports import *
from flashkeras.utils.typehints import *
from flashkeras.utils.typehints import BatchIterator

def show_images_from_batch(batches: BatchIterator, num_images: int = 1, fig_size: tuple[int, int] = (15,5)):
    """Display one or more images sampled from a batch iterator.

    `Flash Explanation:` *`Use this to visually inspect images produced by a data pipeline.`*
    """
    all_images = []
    images, labels = (None, None)
    while True:
        tuple_or_images = next(batches)
        try:
            images, labels = tuple_or_images
        except:
            images = tuple_or_images

        for img in images:
            all_images.append(img)
        if batches.batch_index == 0: break
    
    num_images = min(num_images, len(all_images))
    
    plt.figure(figsize=fig_size)
    
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(all_images[i])
        plt.axis('off')
    
    plt.show()

def show_images_nparray(images: np.ndarray, num_images: int = 1, fig_size: tuple[int, int] = (15,5)):
    """Display several images from a NumPy image batch.

    `Flash Explanation:` *`Use this for a quick visual check of an image array.`*
    """
    num_images = min(num_images, len(images))

    plt.figure(figsize=fig_size)
    
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()

def show_image_nparray(image: np.ndarray, fig_size: tuple[int, int] = (7,7)):
    """Display a single image stored as a NumPy array.

    `Flash Explanation:` *`Use this to inspect one array-backed image.`*
    """
    plt.figure(figsize=fig_size)
    
    plt.imshow(image)
    plt.axis('off')

    plt.show()

def show_images_from_directory(dir_path: str, num_images=1):
    """Display images selected from a directory.

    `Flash Explanation:` *`Use this to preview image files before loading them into a dataset.`*
    """

    files = os.listdir(dir_path)
    
    valid_extentions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = [f for f in files if os.path.splitext(f)[1].lower() in valid_extentions]
    
    num_images = min(num_images, len(images))
    
    plt.figure(figsize=(15, 5))
    
    for i in range(num_images):
        img_path = os.path.join(dir_path, images[i])
        img = mpimg.imread(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
    
    plt.show()

def show_image(image: Union[np.ndarray, Image.Image, str], fig_size: tuple[int, int] = (7,7)) -> Tuple[int, int]:
    """Display an image supplied as an array, PIL image, or file path.

    `Flash Explanation:` *`Use this for a format-agnostic single-image preview.`*
    """
    '''Provide the image or path to the image and shows it.
    '''

    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        show_image_nparray(image, fig_size)
    
    raise ValueError("The image must be one of the following types: ``np.ndarray``, ``Image.Image`` or a ``str`` representing the path.")
