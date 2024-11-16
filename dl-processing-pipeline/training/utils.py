from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import numpy as np
import zlib
import torch.utils.data
import os
from torch.utils.data import Dataset, DataLoader
import time
import json
import logging
from logging.config import dictConfig
from typing import List, Tuple, Callable, Optional
from torch.utils.data import get_worker_info
import psutil
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown


LOGGER = logging.getLogger()

import cProfile
import pstats
import io

import hashlib

class DecodeJPEG:
    """
    Decodes raw JPEG byte data into a PIL image.

    Methods:
        __call__(raw_bytes): Accepts raw JPEG bytes and returns a PIL Image object.
    """
    def __call__(self, raw_bytes):
        return Image.open(BytesIO(raw_bytes)).convert('RGB')

class ConditionalNormalize:
    """
    Conditionally normalizes tensors representing images, adjusting single-channel images
    to three-channel (RGB) by repeating channels if necessary.

    Attributes:
        mean (list): Mean values for normalization across RGB channels.
        std (list): Standard deviation values for normalization across RGB channels.

    Methods:
        __call__(tensor): Applies normalization if the tensor is three-channel.
    """
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, tensor):
        # Only apply normalization if the tensor has 3 channels
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)  # Repeat the single channel across the 3 RGB channels

        # Apply normalization to 3-channel (RGB) images
        return self.normalize(tensor)

class RemoteDataset(torch.utils.data.IterableDataset):
    """
    Streams image data from a remote gRPC server, applying specified transformations and
    optional decompression for each sample. Manages communication with the server and
    yields image batches for training.

    Attributes:
        host (str): Host address of the gRPC server.
        port (int): Port number of the gRPC server.
        batch_size (int): Number of images per batch.

    Methods:
        __iter__(): Connects to the gRPC server and iteratively requests image batches,
                    yielding decompressed and preprocessed samples.
        preprocess_sample(sample, transformations_applied): Applies a series of transformations
                    based on specified settings, preparing the image for model input.
    """
    def __init__(self, host, port, batch_size=256):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        LOGGER.info(f"Initialized RemoteDataset with host={self.host}, port={self.port}, batch_size={self.batch_size}")

    def __iter__(self):
        LOGGER.info("Starting RemoteDataset __iter__")


        try:
            connect_start = time.time()
            channel = grpc.insecure_channel(
                f'{self.host}:{self.port}',
                options=[
                    ('grpc.max_send_message_length', -1),  # 64 MB
                    ('grpc.max_receive_message_length', -1),  # 64 MB
                    ('grpc.http2.max_pings_without_data', 0),  # No limit
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 10000),
                    ('grpc.http2.max_frame_size', 16777216),  # 16 MB, adjust as needed
                ]
            )

            stub = data_feed_pb2_grpc.DataFeedStub(channel)
            config_request = data_feed_pb2.Config(batch_size=self.batch_size)
            sample_stream = stub.get_samples(config_request)
            batch_start = time.time()
            batch_time = 0


            # LOGGER.debug("Requesting data with batch size: %s", self.batch_size)

            batch_images = []
            batch_labels = []

            for sample_batch in sample_stream:
                for sample in sample_batch.samples:
                    # Deserialize image data
                    img_data = sample.image
                    if sample.is_compressed:
                        img_data = zlib.decompress(img_data)
                    # LOGGER.debug(f"Decompression time: {decompress_end - decompress_start:.4f} seconds")
                    # Convert img_data to tensor and add to batch
                    img_tensor = self.preprocess_sample(img_data, sample.transformations_applied)

                    # LOGGER.debug(f"Transformation time: {transform_end - transform_start:.4f} seconds")

                    batch_images.append(img_tensor)
                    batch_labels.append(torch.tensor(sample.label))

                    # Yield a batch when it reaches the desired batch size
                    if len(batch_images) == self.batch_size:
                        yield torch.stack(batch_images), torch.stack(batch_labels)
                        # print(f"Batch images dimensions: {torch.stack(batch_images).shape}")
                        # print(f"First entry values: {torch.stack(batch_images)[0]}")
                        # print(f"Batch labels dimensions: {torch.stack(batch_labels).shape}")
                        # print(f"First label value: {torch.stack(batch_labels)[0]}")
                        batch_end = time.time()
                        batch_time = batch_end - batch_start
                        batch_start = time.time()
                        # LOGGER.debug(f"Yielded a batch of size: {self.batch_size} in {batch_time:.4f} seconds")
                        batch_images = []
                        batch_labels = []
                        # LOGGER.debug(f"Yielded a batch of size: {self.batch_size}")

        except Exception:
            LOGGER.error("Unexpected error in RemoteDataset __iter__", exc_info=True)


    def preprocess_sample(self, sample, transformations_applied):
        """
        Applies a sequence of transformations to an image sample based on the number of transformations
        requested. Handles decoding and optional normalization, as well as resizing and flipping.

        Arguments:
            sample (bytes): Image data to be transformed.
            transformations_applied (int): Number of transformations to apply to the sample.

        Returns:
            Transformed sample as a tensor, or None if an error occurs.
        """
        try:
            if 0 < transformations_applied <= 3:
                    sample = Image.open(BytesIO(sample)).convert('RGB')
            elif transformations_applied > 3:
                img_array = np.frombuffer(sample, dtype=np.float32).copy()
                # Reshape based on expected image shape, e.g., (3, 224, 224) for RGB images
                sample = torch.from_numpy(img_array.reshape(3, 224, 224))
            # List of transformations to apply individually
            decode_jpeg = DecodeJPEG()


            transformations = [
                decode_jpeg,  # Decode raw JPEG bytes to a PIL image
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converts PIL images to tensors
                ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Conditional normalization
            ]

            processed_sample = sample
            for i in range(transformations_applied, len(transformations)):
                if transformations[i] is not None:
                    processed_sample = transformations[i](processed_sample)
        except Exception:
            LOGGER.error("Error in preprocess_sample", exc_info=True)
            return None
        return processed_sample


class ImagePathDataset(Dataset):
    """
    Custom Dataset class that reads image file paths from a directory structure and assigns
    labels based on directory names. Useful for loading data without immediately reading images.

    Attributes:
        root_dir (str): Root directory of the dataset with subdirectories for each class.
        image_paths (list): List of image file paths.
        targets (list): List of labels corresponding to each image path.
        classes (list): List of class names.
        class_to_idx (dict): Dictionary mapping class names to indices.
        samples (list): List of (image_path, target) tuples.

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image bytes and corresponding label for the specified index.
    """

    def __init__(self, root_dir: str, loader: Optional[Callable[[str], bytes]] = None):
        self.root_dir = root_dir
        self.image_paths = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        # self.loader = loader if loader is not None else self.default_loader

        # Traverse the directory structure and collect image paths and targets
        class_names = sorted(os.listdir(root_dir))
        self.classes = class_names
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        # print(f"Class names: {class_names}")

        for class_name in class_names:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in sorted(os.listdir(class_dir)):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.targets.append(class_idx)
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        # worker_info = get_worker_info()
        # if worker_info is not None:
        #     # Split dataset among workers
        #     num_workers = worker_info.num_workers
        #     worker_id = worker_info.id
        #     # Determine the subset of indices for this worker
        #     per_worker = len(self.image_paths) // num_workers
        #     worker_start = worker_id * per_worker
        #     worker_end = worker_start + per_worker
        #     idx = (idx % per_worker) + worker_start

        img_path = self.image_paths[idx]
        target = self.targets[idx]
        return img_path, target
        
    
    
def encode_p(img: Image, fmt: str = "JPEG", quality: int = 80) -> bytes:
    """
    Compresses an image in-memory using Pillow and returns the compressed data as bytes.
    
    Parameters:
    - img (Image): The PIL image to compress.
    - fmt (str): Compression format (e.g., 'JPEG', 'JPEG2000', 'WebP').
    - quality (int): Quality level for the compression, default is 80.
    
    Returns:
    - bytes: The compressed image as bytes.
    """
    # Prepare a BytesIO stream to hold the in-memory compressed image
    output = BytesIO()
    
    # Apply compression based on the format
    if fmt == "JPEG":
        img.save(output, format="JPEG", quality=quality)
    elif fmt == "JPEG2000":
        img.save(output, format="JPEG2000", quality_mode="rates")
    elif fmt == "WebP":
        img.save(output, format="WebP", quality=quality)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    
    # Get the byte data from the BytesIO stream
    compressed_data = output.getvalue()
    output.close()
    
    return compressed_data

def custom_collate_fn(batch):
    raw_images = []
    targets = []
    for img_path, target in batch:
        with open(img_path, 'rb') as f:
            raw_img_data = f.read()  # Read the raw JPEG image in binary
        raw_images.append(raw_img_data)
        targets.append(target)
    
    return raw_images, targets  # Return two lists: images and targets

if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    # Example usage of RemoteDataset
    dataset = RemoteDataset(host='localhost', port=50051, batch_size=32)
    train_loader = DataLoader(
        dataset, batch_size=None, num_workers=0, pin_memory=False
    )

    total_images = 0
    target_images = 1000
    start_time = time.time()
    batch_time_start = time.time()
    for i, (images, target) in enumerate(train_loader):

        batch_size = images.shape[0]  # Current batch size (could vary depending on availability)
        total_images += batch_size

        # Flatten the nested batches into a single batch dimension
        # LOGGER.debug(f"Batch {i}: Loaded {batch_size} images. Total loaded so far: {total_images}")
        images = images.view(-1, 3, 224, 224)  # Flatten: (2, 2, 3, 224, 224) -> (4, 3, 224, 224)
        target = target.view(-1)  # Adjust target as well
        batch_time_end = time.time()
        # LOGGER.debug(f"Batch {i}: Loaded {batch_size} images in {batch_time_end - batch_time_start:.4f} seconds")
        batch_time_start = time.time()

        # Stop if we've loaded 1000 images
        if total_images >= target_images:
            end_time = time.time()
            elapsed_time = end_time - start_time
            throughput = total_images / elapsed_time  # Images per second

            LOGGER.debug(f"Loaded {total_images} images in {elapsed_time:.2f} seconds. Throughput: {throughput:.2f} images/second")
            # profiler.disable()
            # stream = io.StringIO()
            # stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
            # stats.print_stats(20)  # Display top 20 time-consuming functions
            # LOGGER.debug(stream.getvalue())
            break


    # Measure end time and calculate throughput

def load_logging_config():
    """
    Loads the logging configuration from a JSON file and applies it.

    This function reads a JSON configuration file for logging, modifies
    file handler paths if not in production, and applies the configuration
    using dictConfig from the logging module.

    If the 'PROD' environment variable is not set,
    the file paths for handlers are overridden to point to local log files.

    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    log_config = os.path.join(base_path, "logging.json")

    with open(log_config, 'r') as read_file:
        config = json.load(read_file)

    if os.environ.get("PROD") is None:
        config["handlers"]["file"]["filename"] = os.path.join(base_path, 'logs/debug_logs.log')
        config["handlers"]["data_collection_handler"]["filename"] = os.path.join(base_path, 'logs/data_log.log')

    dictConfig(config)

def custom_collate_fn(batch):
    """
    Custom collate function for the DataLoader, reading raw images (instead of auto encoding to PIL Image object) and targets from disk.
    
    Arguments:
        batch (list): List of tuples with image paths and labels.
    Returns:
        tuple: Three lists - raw images as binary data and their corresponding targets and id's.
    """
    raw_images = []
    targets = []
    sample_ids = []
    for img_path, target in batch:
        with open(img_path, 'rb') as f:
            raw_img_data = f.read()  # Read the raw JPEG image in binary
        raw_images.append(raw_img_data)
        targets.append(target)
        sample_ids.append(generate_id(img_path))
        # print(f"Loaded image {img_path} with target {target}")
    
    return raw_images, targets, sample_ids  # Return two lists: images and targets

@staticmethod
def generate_id(filename):
    """Generate a unique integer ID based on the file name."""
    # Hash the file name and convert to an integer for a stable ID
    hash_object = hashlib.md5(filename.encode())
    return int(hash_object.hexdigest(), 16) % (10 ** 8)

def monitor_system(logger):
    logger.info("=== CPU ===")
    logger.info(f"CPU Utilization: {psutil.cpu_percent(interval=1)}%, CPU Core Utilization: {psutil.cpu_percent(interval=1, percpu=True)}, RAM Usage: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB / {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

    if torch.cuda.is_available():
        logger.info("\n=== GPU ===")
        nvmlInit()
        for i in range(torch.cuda.device_count()):
            handle = nvmlDeviceGetHandleByIndex(i)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            memory = nvmlDeviceGetMemoryInfo(handle)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, GPU Utilization: {utilization.gpu}%, GPU Memory Utilization: {utilization.memory}%, GPU Memory Used: {memory.used / (1024 ** 2):.2f} MB, GPU Memory Total: {memory.total / (1024 ** 2):.2f} MB")

        nvmlShutdown()
    else:
        logger.info("\nNo GPU available.")