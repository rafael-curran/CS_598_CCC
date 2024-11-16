import multiprocessing as mp
from concurrent import futures
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import torch
from torchvision import datasets, transforms
import numpy as np
import os
import zlib
import time
from io import BytesIO
import argparse
from utils import DecodeJPEG, ConditionalNormalize, ImagePathDataset, load_logging_config, custom_collate_fn
import asyncio
import hashlib
import logging
import numpy as np


from PIL import Image

kill = mp.Event()  # Global event to signal termination
num_cores = mp.cpu_count()

LOGGER = logging.getLogger()
DATA_LOGGER = logging.getLogger("data_collection")

if os.environ.get("PROD") is None:
    IMAGENET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet")
    LOGGER.info('Running in PROD: imagenet path is %s', IMAGENET_PATH)
else:
    IMAGENET_PATH = "/data/imagenet"
    LOGGER.info('Running in LOCAL: imagenet path is %s', IMAGENET_PATH)

def parse_args():
    """
    Parses command-line arguments to configure the data feed server.
    Arguments include:
    - `--offloading`: Sets the level of offloading (0 for no offloading, 1 for full offloading, 2 for dynamic).
    - `--compression`: 0- no compression, 1- compress all samples before sending, 2- compress based on selective offloading plan.
    - `--batch_size`: Determines the batch size for loading images.
    - `--compression-method`: Specifies the type of compression to use (e.g., 'zlib', 'pillow').
    - `--compression-level`: compression level for image compression (0-9) for zlib
    - '--compression-quality': Quality level for image compression (1-95) for pillow
    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Start the data feed server with an offloading plan.")
    parser.add_argument('--offloading', type=int, default=0, help='Set to 0 for no offloading, 1 for full offloading, or 2 for dynamic offloading including compression.')
    parser.add_argument('--compression', type=int, default=0, help='Set to 1 to enable compression before sending the sample.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for loading images.')
    parser.add_argument(
        '--compression-method', type=str, choices=['zlib', 'pillow'], default='zlib',
        help="Specifies the type of compression to use ('zlib' or 'pillow')."
    )
    parser.add_argument(
        '--compression-level', type=int, default=6, choices=range(0, 10),
        help='Compression level for zlib (0-9). Default is 6.'
    )
    parser.add_argument(
        '--compression-quality', type=int, default=75, choices=range(1, 96),
        help='Compression quality for pillow (1-95). Default is 75.'
    )
    return parser.parse_args()


def handle_termination(signum, frame):
    """
    Handles system termination signals by setting a global event `kill`,
    which signals workers and processes to stop gracefully.
    Arguments:
    - `signum`: Signal number.
    - `frame`: Current stack frame (not used directly).
    """
    LOGGER.info("Termination signal received. Stopping workers...")
    kill.set()  # Set the event to stop the fill_queue process


class DataFeedService(data_feed_pb2_grpc.DataFeedServicer):
    """
    Implements the gRPC service for streaming batched image samples to a client.
    Manages interactions with a shared queue and applies offloading plans as requested.

    Attributes:
        q (multiprocessing.Queue): Queue from which samples are retrieved.
        offloading_plan (dict): Cache storing the offloading plan for each sample ID.
    """
    def __init__(self, q, offloading_plan):
        self.q = q
        self.offloading_plan = offloading_plan  # Store offloading plan for each sample
    
    async def update_offloading_plan(self, request, context):
        """
        Updates the offloading plan for a specific sample based on client request.

        Arguments:
            request (OffloadingRequest): Contains sample_id, transformations, and compress flag.
            context: gRPC context for the server.
        Returns:
            Response: Status message confirming update.
        """
        # Update the offloading plan with the request details
        sample_id = request.sample_id
        self.offloading_plan[sample_id] = {
            'transformations': request.transformations,
            'compress': request.compress
        }
        print(f"Updated offloading plan for sample {sample_id}: "
              f"transformations={request.transformations}, compress={request.compress}")

        # Return a confirmation response
        return data_feed_pb2.Response(status=f"Offloading plan updated for sample {sample_id}")


    async def get_samples(self, request, context):
        """
        Asynchronous gRPC method to yield batched samples from a shared queue.
        Applies any requested transformations and compression to each sample.

        Arguments:
            request: gRPC request object.
            context: gRPC context for the server.
        Yields:
            SampleBatch: Batch of samples formatted for gRPC transmission.
        """

        LOGGER.debug("Server: Received request for samples")
        while not kill.is_set():
            try:
                # Attempt to retrieve the next sample batch
                sample_batch = self.q.get(timeout=5)  # Get individual samples from the queue 1699848 /16 *100000
                sample_batch_proto = [
                    data_feed_pb2.Sample(
                        id=sample[0],
                        image=sample[1],
                        label=sample[2],
                        transformations_applied=sample[3],
                        is_compressed=sample[4]
                    )
                    for sample in sample_batch
                ]

                # Log the data types before yielding
                LOGGER.info("Debug - Types in `get_samples` before yielding: id: %s, image: %s, label: %s, transformations_applied: %s, is_compressed: %s",
                            type(sample_batch[0][0]), type(sample_batch[0][1]), type(sample_batch[0][2]), type(sample_batch[0][3]), type(sample_batch[0][4]))
                # Calculate and print the data size of sample_batch_proto
                data_size = sum(len(sample.image) for sample in sample_batch_proto)
                LOGGER.info(f"Data size of sample_batch_proto: {data_size} bytes")

                # Yield the data in the expected gRPC format
                yield data_feed_pb2.SampleBatch(samples=sample_batch_proto)

            except Exception:
                LOGGER.error("Server: Error while yielding samples", exc_info=True)
                break  # Exit on unrecoverable errorsprint(f"Worker {worker_id} - Batch {batch_idx}: Loaded {len(data)} images.")

def fill_queue(q, kill, args, dataset_path, offloading_plan, worker_id):
    """
    Loads image batches from the dataset, applies transformations based on offloading settings,
    and enqueues them for streaming to clients. Handles optional compression.

    Arguments:
        q (multiprocessing.Queue): Queue for sample batches.
        kill (mp.Event): Global event to signal worker termination.
        args (Namespace): Parsed command-line arguments.
        dataset_path (str): Path to the dataset.
        offloading_plan (dict): Maps sample IDs to transformations. Will be used for selective offloading.
        worker_id (int): Unique ID for the worker instance (used in logging).
    """
    # Custom decode transformation
    decode_jpeg = DecodeJPEG()

    transformations = [
        decode_jpeg,  # Decode raw JPEG bytes to a PIL image
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts PIL images to tensors
        ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]

    # Ensure that ImageFolder uses the transform to convert images to tensors
    dataset = ImagePathDataset(os.path.join(dataset_path, 'train'))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, collate_fn=custom_collate_fn)
    while not kill.is_set():
        for batch_idx, (data, target, sample_ids) in enumerate(loader):
            LOGGER.info(f"Worker {worker_id} - Batch {batch_idx}: Loaded {len(data)} images.")
            sample_batch =[]
            for i, img in enumerate(data): # Loop over individual samples
                sample_id = sample_ids[i]
                is_compressed = False
                if args.offloading == 0:
                    num_transformations = 0
                elif args.offloading == 1:
                    num_transformations = 5
                else:
                    num_transformations, is_compressed = offloading_plan.get(sample_id, (0, False))
                    if num_transformations == 0 and (is_compressed or args.compression == 1):
                        num_transformations = 1

                transformed_data = img  
                for j in range(min(num_transformations, 5)):  
                    transformed_data = transformations[j](transformed_data)
                # serialize everything to bytes, except for the PIL Image if compression method is 'pillow'
                if not (args.compression_method == 'pillow' and isinstance(transformed_data, Image.Image)):
                    if isinstance(transformed_data, Image.Image):
                        # Serialize PIL Image to bytes (JPEG format by default)
                        img_byte_arr = BytesIO()
                        transformed_data.save(img_byte_arr, format='JPEG')
                        transformed_data = img_byte_arr.getvalue()
                    elif isinstance(transformed_data, torch.Tensor):
                        # Serialize Torch Tensor to bytes (NumPy array)
                        transformed_data = transformed_data.numpy().tobytes()
                    elif isinstance(transformed_data, np.ndarray):
                        # Serialize NumPy array to bytes
                        transformed_data = transformed_data.tobytes()
                    else:
                        # Fallback: Use bytes() for other types
                        transformed_data = bytes(transformed_data)
                    

                # Step 2: Compression (if enabled)
                if args.compression == 1 or is_compressed:
                    if args.compression_method == 'pillow' and isinstance(transformed_data, Image.Image):
                        img_byte_arr = BytesIO()
                        transformed_data.save(
                            img_byte_arr, format='JPEG', quality=args.compression_quality
                        )
                        transformed_data = img_byte_arr.getvalue()
                        # Kind of confusing, but I'm setting it to false because it doesn't need to be decompressed, but zlib does
                        is_compressed = False
                    else:
                        # compress everything else with zlib
                        transformed_data = zlib.compress(transformed_data, level=args.compression_level)
                        is_compressed = True

                label = target[i]
                
                sample = (sample_id, transformed_data, label, num_transformations, is_compressed)
                sample_batch.append(sample)
            added = False
            while not added and not kill.is_set():
                try:
                    q.put(sample_batch, timeout=1)
                    # LOGGER.info(f"Worker {worker_id}: Successfully added sample {sample_id} to queue.")
                    added = True
                except:
                    continue

async def serve(args):
    """
    Initializes and runs the gRPC server to serve data to clients.
    Spawns multiple worker processes to fill the data queue and manages shutdown signals.

    Arguments:
        offloading_value (int): Sets the offloading configuration.
        compression_value (int): Determines whether to compress samples before sending.
        batch_size (int): Number of images in each batch.
    """
    q = mp.Queue(5000)

    # Cache for storing the offloading plan (sample_id -> number of transformations)
    offloading_plan = {}

    # Start the fill_queue process
    workers = []
    for worker_id in range(num_cores):
        p = mp.Process(target=fill_queue, args=(q, kill, args, 'imagenet', offloading_plan, worker_id))
        workers.append(p)
        p.start()
    
    # Start the gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=8),
        # Client and Server: Increase message sizes and control flow limits
        options = [
                ('grpc.max_send_message_length', -1),  # 1 GB
                ('grpc.max_receive_message_length', -1),  # 1 GB
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 10000),
                ('grpc.http2.max_frame_size', 16777216),  # 16 MB, adjust as needed

            ]

    )
    data_feed_pb2_grpc.add_DataFeedServicer_to_server(DataFeedService(q, offloading_plan), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()
    
    kill.set()
    for p in workers:
        p.join()

if __name__ == '__main__':
    """
    Main entry point of the script. Parses command-line arguments and starts
    the asynchronous data feed server with the specified configuration.
    """
    load_logging_config()
    args = parse_args()
    
    asyncio.run(serve(args))
    
