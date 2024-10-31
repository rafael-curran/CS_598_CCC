import time
import torch
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
from torchvision import datasets, transforms
import numpy as np
import zlib
import time
from decision_engine import DecisionEngine
import sys
from io import BytesIO
from PIL import Image
import os
from utils import DecodeJPEG, ConditionalNormalize, RemoteDataset, ImagePathDataset
import torchvision.models as models
import torch.nn as nn

class Profiler:
    def __init__(self, batch_size, dataset_path, grpc_host, grpc_port):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        # Automatically choose MPS (for macOS GPU) if available, otherwise CPU
        self.lr = 0.1               # Learning rate
        self.momentum = 0.9         # Momentum
        self.weight_decay = 1e-4    # Weight decay
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


    def stage_one_profiling(self):
        # 1. Measure GPU throughput with synthetic data (remains the same)
        num_samples_gpu =  100
        start = time.time()
        train_dataset = datasets.FakeData(100, (3, 224, 224), 10, transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)


        # define loss function (criterion), optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss().to(self.device)
        model = models.__dict__['alexnet']()
        model = model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)

        model.train()
        start_time = time.time()

        for i, (images, target) in enumerate(train_loader):
            # Flatten the nested batches into a single batch dimension
            if i >= num_samples_gpu:
                break
            images = images.view(-1, 3, 224, 224)  # Flatten: (2, 2, 3, 224, 224) -> (4, 3, 224, 224)
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
            f'{self.grpc_host}:{self.grpc_port}',
            options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
            ]
        )
        stub = data_feed_pb2_grpc.DataFeedStub(channel)

        samples = stub.StreamSamples(iter([]))

        for i, sample_batch in enumerate(samples):
            if i >= 100:  # Limit to 100 batches for profiling
                break
            for s in sample_batch.samples:
                io_samples.append(s)

                 # Store the sample for later CPU processing
                num_samples_io += 1


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
                processed_image, _, _ = self.preprocess_sample(decompressed_image, s.transformations_applied)
            else:
                img_np = np.frombuffer(decompressed_image, dtype=np.float32)
                img_np = img_np.reshape((3, 224, 224))
                processed_image = torch.tensor(img_np)

            label = torch.tensor(s.label)
            num_samples_cpu += 1

        cpu_time = time.time() - start
        print("CPU Time:", cpu_time)
        print("Num Samples CPU:", num_samples_cpu)
        print("Sample Metrics:", sample_metrics)
        cpu_throughput = num_samples_cpu / cpu_time

        return gpu_throughput, io_throughput, cpu_throughput



    def stage_two_profiling(self):
        cpu_device = torch.device("cpu")
        channel = grpc.insecure_channel(
            f'{self.grpc_host}:{self.grpc_port}',
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

                # Process the sample and record metrics
                if isinstance(sample.image, bytes):
                    transformed_data, times_per_transformation, transformed_sizes_per_transformation = self.preprocess_sample(sample.image, sample.transformations_applied)
                elif isinstance(sample.image, torch.Tensor):
                    transformed_data, times_per_transformation, transformed_sizes_per_transformation = self.preprocess_sample(sample.image, sample.transformations_applied)

                sample_metrics.append({
                    'original_size': original_size,
                    'transformed_sizes': transformed_sizes_per_transformation,
                    'preprocessing_times': times_per_transformation
                })
                if len( sample_metrics) >= 10:
                    return sample_metrics


    def preprocess_sample(self, sample, transformations_applied):
        # List of transformations to apply individually
        decode_jpeg = DecodeJPEG()  # Assuming this is a method of the class

        transformations = [
            decode_jpeg,  # Decode raw JPEG bytes to a PIL image
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converts PIL images to tensors
            ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Conditional normalization
        ]

        processed_sample = sample
        sizes = []
        times = []

        # Apply transformations starting from the index `transformations_applied`
        for i in range(transformations_applied, len(transformations)):
            transform = transformations[i]

            if transform is not None:
                start_time = time.time()
                processed_sample = transform(processed_sample)  # Apply transformation
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)  # Record the time for each transformation
            else:
                times.append(0)  # No-op for None transformations

            processed_sample = sample
            sizes = []
            times = []

            # Apply transformations starting from the index `transformations_applied`
            for i in range(transformations_applied, len(transformations)):
                transform = transformations[i]

                if transform is not None:
                    start_time = time.time()
                    processed_sample = transform(processed_sample)  # Apply transformation
                    elapsed_time = time.time() - start_time
                    times.append(elapsed_time)  # Record the time for each transformation
                else:
                    times.append(0)  # No-op for None transformations

                # Calculate size after the transformation
                if isinstance(processed_sample, torch.Tensor):
                    # For PyTorch tensors
                    data_size = processed_sample.nelement() * processed_sample.element_size()
                elif isinstance(processed_sample, np.ndarray):
                    # For NumPy arrays
                    data_size = processed_sample.nbytes
                elif isinstance(processed_sample, Image.Image):
                    # For PIL images
                    data_size = len(processed_sample.tobytes())  # Convert to bytes and measure length
                else:
                    # Fallback to sys.getsizeof for other types
                    data_size = sys.getsizeof(processed_sample)

                sizes.append(data_size)
        except Exception as e:
            print ("Error in preprocess_sample:", e)
            print ("Stack trace:", sys.exc_info())
        # print("Transformation Times:", times)
        return processed_sample, times, sizes




    def run_profiling(self):
        # Stage 1: Basic throughput analysis
        gpu_throughput, io_throughput, cpu_preprocessing_throughput = self.stage_one_profiling()
        print("GPU Throughput:", gpu_throughput)
        print("I/O Throughput:", io_throughput)
        print("CPU Preprocessing Throughput:", cpu_preprocessing_throughput)
        # if io_throughput < cpu_preprocessing_throughput:
        #     # Stage 2: Detailed sample-specific profiling
        #     return gpu_throughput, io_throughput, cpu_preprocessing_throughput, self.stage_two_profiling()
        return gpu_throughput, io_throughput, cpu_preprocessing_throughput, None




if __name__ == '__main__':
    profiler = Profiler(batch_size=1, dataset_path='imagenet', grpc_host='localhost', grpc_port=50051)
    gpu_throughput, io_throughput, cpu_preprocessing_throughput, sample_metrics = profiler.run_profiling()
    print("Sample Metrics:", sample_metrics)
    # if sample_metrics:  # If the profiler identifies an I/O bottleneck
    #     decision_engine = DecisionEngine(sample_metrics, gpu_t = gpu_throughput, cpu_t =cpu_preprocessing_throughput, io_t = io_throughput,  cpu_cores_compute=1, cpu_cores_storage=8, grpc_host='localhost', grpc_port=50051)
    #     offloading_plan = decision_engine.iterative_offloading()
    #     print(offloading_plan)

