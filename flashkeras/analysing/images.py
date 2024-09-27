from flashkeras.utils.otherimports import *
from flashkeras.utils.typehints import *
from flashkeras.utils.typehints import BatchIterator

def show_images_from_batch(batches: BatchIterator, num_images: int = 1, fig_size: tuple[int, int] = (15,5)):
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
    num_images = min(num_images, len(images))

    plt.figure(figsize=fig_size)
    
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()

def show_image_nparray(image: np.ndarray, fig_size: tuple[int, int] = (7,7)):
    plt.figure(figsize=fig_size)
    
    plt.imshow(image)
    plt.axis('off')

    plt.show()

def show_images_from_directory(dir_path: str, num_images=1):

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
    '''Provide the image or path to the image and shows it.
    '''

    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        show_image_nparray(image, fig_size)
    
    raise ValueError("The image must be one of the following types: ``np.ndarray``, ``Image.Image`` or a ``str`` representing the path.")
