import time
import torch
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
from torchvision import datasets, transforms
import numpy as np
import zlib
import sys
from io import BytesIO
from PIL import Image
import os
import heapq
from decision_engine import DecisionEngine
from utils import DecodeJPEG, ConditionalNormalize, RemoteDataset, ImagePathDataset, encode_p
import torchvision.models as models
import torch.nn as nn
from utils import load_logging_config
import logging


LOGGER = logging.getLogger()


class Profiler:
    def __init__(self, batch_size, dataset_path, grpc_host, grpc_port):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        # Automatically choose MPS (for macOS GPU) if available, otherwise CPU
        self.lr = 0.1  # Learning rate
        self.momentum = 0.9  # Momentum
        self.weight_decay = 1e-4  # Weight decay
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    def stage_one_profiling(self):
        # 1. Measure GPU throughput with synthetic data (remains the same)
        num_samples_gpu = 100
        start = time.time()
        train_dataset = datasets.FakeData(100, (3, 224, 224), 10, transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True
        )
        # define loss function (criterion), optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss().to(self.device)
        model = models.__dict__["alexnet"]()
        model = model.to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        model.train()
        start_time = time.time()

        for i, (images, target) in enumerate(train_loader):
            # Flatten the nested batches into a single batch dimension
            if i >= num_samples_gpu:
                break
            images = images.view(
                -1, 3, 224, 224
            )  # Flatten: (2, 2, 3, 224, 224) -> (4, 3, 224, 224)
            target = target.view(-1)  # Adjust target as well

            images, target = images.to(self.device), target.to(self.device)

            output = model(images)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gpu_time = time.time() - start_time
        gpu_throughput = num_samples_gpu / gpu_time

        # 2. Measure I/O throughput over gRPC (between training and storage node)
        num_samples_io = 0
        io_samples = []  # Store I/O samples for later CPU processing
        start = time.time()
        channel = grpc.insecure_channel(
            f"{self.grpc_host}:{self.grpc_port}",
            options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
            ]
        )
        stub = data_feed_pb2_grpc.DataFeedStub(channel)

        # Initiate the streaming request
        config_request = data_feed_pb2.Config(batch_size=self.batch_size)
        sample_stream = stub.get_samples(config_request)
        for sample_batch in sample_stream:
            for i, sample in enumerate(sample_batch.samples):
                if i >= 1000:  # Limit to 100 samples for profiling
                    break
                io_samples.append(sample)  # Collect individual samples
                num_samples_io += 1
            if num_samples_io >= 1000:
                break

        io_time = time.time() - start
        io_throughput = num_samples_io / io_time

        # 3. Measure CPU throughput (reuse samples from I/O section)
        num_samples_cpu = 0
        start = time.time()

        # Reuse samples fetched during I/O for CPU processing
        sample_metrics = []
        for i, s in enumerate(io_samples):
            if s.is_compressed:
                decompressed_image = zlib.decompress(s.image)
            else:
                decompressed_image = s.image

            if s.transformations_applied < 5:
                processed_image, _, _ ,_,_= self.preprocess_sample(decompressed_image, s.transformations_applied)
            else:
                img_np = np.frombuffer(decompressed_image, dtype=np.float32)
                img_np = img_np.reshape((3, 224, 224))
                processed_image = torch.tensor(img_np)

            label = torch.tensor(s.label)
            num_samples_cpu += 1

        cpu_time = time.time() - start
        LOGGER.info(f"CPU Time: {cpu_time}")
        LOGGER.info(f"Num Samples CPU: {num_samples_cpu}")
        LOGGER.info(f"Sample Metrics: {sample_metrics}")
        cpu_throughput = num_samples_cpu / cpu_time

        return gpu_throughput, io_throughput, cpu_throughput

    def stage_two_profiling(self):
        cpu_device = torch.device("cpu")
        channel = grpc.insecure_channel(
            f"{self.grpc_host}:{self.grpc_port}",
            options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
            ]
        )
        stub = data_feed_pb2_grpc.DataFeedStub(channel)

        # Use the same Config request with an appropriate batch size
        config_request = data_feed_pb2.Config(batch_size=self.batch_size)
        sample_stream = stub.get_samples(config_request)

        sample_metrics = []
        for sample_batch in sample_stream:
            for i, sample in enumerate(sample_batch.samples):
                original_size = len(sample.image) if isinstance(sample.image, bytes) else sample.image.nelement() * sample.image.element_size()
                img_data = sample.image
                if sample.is_compressed:
                    img_data = zlib.decompress(img_data)
                # Process the sample and record metrics
                if isinstance(sample.image, bytes):
                    transformed_data, times_per_transformation, transformed_sizes_per_transformation, compressed_sizes, compression_times = self.preprocess_sample(img_data, sample.transformations_applied)
                elif isinstance(sample.image, torch.Tensor):
                    transformed_data, times_per_transformation, transformed_sizes_per_transformation, compressed_sizes, compression_times = self.preprocess_sample(img_data, sample.transformations_applied)
                
                # Append the metrics for this sample
                sample_metrics.append({
                    'sample_id': sample.id,
                    'original_size': original_size,
                    'transformed_sizes': transformed_sizes_per_transformation,
                    'preprocessing_times': times_per_transformation,
                    'compressed_sizes': compressed_sizes,
                    'compression_times': compression_times
                })
                
                # Limit the number of samples to 100 for testing
                if len(sample_metrics) >= 10000:
                    return sample_metrics

    def preprocess_sample(self, sample, transformations_applied):
        # List of transformations to apply individually
        # LOGGER.debug(f"Debug - preprocess_sample: Data type of sample: {type(sample)}, Transformations applied: {transformations_applied}")

        try:
            # Decode the image based on transformations applied
            if 0 < transformations_applied <= 3:
                sample = Image.open(BytesIO(sample)).convert('RGB')
            elif transformations_applied > 3:
                img_array = np.frombuffer(sample, dtype=np.float32).copy()
                sample = torch.from_numpy(img_array.reshape(3, 224, 224))

            decode_jpeg = DecodeJPEG()  # Assuming this is a method of the class

            transformations = [
                decode_jpeg,  # Decode raw JPEG bytes to a PIL image
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converts PIL images to tensors
                ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Conditional normalization
            ]

            processed_sample = sample
            transformed_sizes = []
            preprocessing_times = []
            compressed_sizes = []
            compression_times = []

            # Apply transformations starting from the index `transformations_applied`
            for i in range(transformations_applied, len(transformations)):
                transform = transformations[i]
                
                # Measure preprocessing time
                start_time = time.time()
                processed_sample = transform(processed_sample)
                elapsed_time = time.time() - start_time
                preprocessing_times.append(elapsed_time)

                # Measure transformed size
                if isinstance(processed_sample, torch.Tensor):
                    data_size = processed_sample.nelement() * processed_sample.element_size()
                elif isinstance(processed_sample, np.ndarray):
                    data_size = processed_sample.nbytes
                elif isinstance(processed_sample, Image.Image):
                    data_size = len(processed_sample.tobytes())
                else:
                    data_size = sys.getsizeof(processed_sample)
                transformed_sizes.append(data_size)

                # Measure compressed size and compression time
                start_time = time.time()
                if isinstance(processed_sample, torch.Tensor):
                    data_bytes = processed_sample.numpy().tobytes()
                elif isinstance(processed_sample, np.ndarray):
                    data_bytes = processed_sample.tobytes()
                elif isinstance(processed_sample, Image.Image):
                    img_byte_arr = BytesIO()
                    processed_sample.save(img_byte_arr, format='JPEG')
                    data_bytes = img_byte_arr.getvalue()
                else:
                    data_bytes = bytes(processed_sample)
                    
                compressed_data = zlib.compress(data_bytes)
                compression_time = time.time() - start_time
                compression_times.append(compression_time)
                compressed_sizes.append(len(compressed_data))

        except Exception as e:
            print("Error in preprocess_sample:", e)
            print("Stack trace:", sys.exc_info())

        return processed_sample, preprocessing_times, transformed_sizes, compressed_sizes, compression_times

    def run_profiling(self):
        # Stage 1: Basic throughput analysis
        gpu_throughput, io_throughput, cpu_preprocessing_throughput = (
            self.stage_one_profiling()
        )
        LOGGER.info(f"GPU Throughput: {gpu_throughput}")
        LOGGER.info(f"I/O Throughput: {io_throughput}")
        LOGGER.info(f"CPU Preprocessing Throughput: {cpu_preprocessing_throughput}")
        # if io_throughput < cpu_preprocessing_throughput:
        #     # Stage 2: Detailed sample-specific profiling
        #     return gpu_throughput, io_throughput, cpu_preprocessing_throughput, self.stage_two_profiling()
        # return gpu_throughput, io_throughput, cpu_preprocessing_throughput, None
        return gpu_throughput, io_throughput, cpu_preprocessing_throughput, self.stage_two_profiling()

    def compress_with_zlib(self, tensor: torch.Tensor) -> bytes:
        """
        Compresses a tensor using zlib after converting it to bytes.
        
        Parameters:
        - tensor (torch.Tensor): The tensor to compress.
        
        Returns:
        - bytes: Compressed byte representation of the tensor.
        """
        # Convert tensor to a byte array
        byte_data = tensor.numpy().tobytes()
        
        # Compress with zlib
        compressed_data = zlib.compress(byte_data)
        return compressed_data


    def decompress_with_zlib(self, compressed_data: bytes) -> torch.Tensor:
        """
        Decompresses zlib-compressed byte data and converts it back to a tensor.
        
        Parameters:
        - compressed_data (bytes): The zlib-compressed byte data.
        
        Returns:
        - torch.Tensor: Decompressed tensor.
        """
        # Decompress the byte data
        decompressed_data = zlib.decompress(compressed_data)
        
        # Convert back to a numpy array, then to a tensor
        img_array = np.frombuffer(decompressed_data, dtype=np.float32).reshape(3, 224, 224)
        return torch.tensor(img_array)


if __name__ == '__main__':
    load_logging_config()
    profiler = Profiler(batch_size=200, dataset_path='imagenet', grpc_host='localhost', grpc_port=50051)
    gpu_throughput, io_throughput, cpu_preprocessing_throughput, sample_metrics = profiler.run_profiling()
    #estimate io_bandwidth by sum of original sizes and io_throughput
    io_bandwidth = 82100000
    LOGGER.info(f"IO Bandwidth: {io_bandwidth}")
    # print("Sample Metrics:", sample_metrics)
    if sample_metrics:  # If the profiler identifies an I/O bottleneck
        decision_engine = DecisionEngine(sample_metrics,  io_bandwidth= io_bandwidth,  cpu_cores_compute=1, cpu_cores_storage=8, grpc_host='localhost', grpc_port=50051)
        offloading_plan = decision_engine.send_offloading_requests()
        print(offloading_plan)

