import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import logging
from utils import load_logging_config

LOGGER = logging.getLogger()

import heapq
class DecisionEngine:
    def __init__(self, sample_metrics, io_bandwidth,  cpu_cores_compute, cpu_cores_storage, grpc_host, grpc_port):
        """
        :param sample_metrics: List of tuples (original_size, transformed_size, preprocessing_time)
        :param tg: GPU time for one epoch (TG)
        :param tcc: CPU time on compute node (TCC)
        :param tcs: CPU time on storage node (TCS)
        :param tnet: Network transfer time (TNet)
        :param cpu_cores_compute: Number of CPU cores on compute node
        :param cpu_cores_storage: Number of CPU cores on storage node
        :param grpc_host: Hostname for the gRPC server
        :param grpc_port: Port for the gRPC server
        """
        self.sample_metrics = sample_metrics
        self.num_samples = len(sample_metrics)
        self.cpu_cores_compute = cpu_cores_compute
        self.cpu_cores_storage = cpu_cores_storage
        self.io_bandwidth = io_bandwidth
        
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.offloading_plan = {}
        
    def decide_offloading(self):
        offloading_heap = []
        LOGGER.info(f"Sample_metrics data type = {type(self.sample_metrics)}")

        # Iterate through each sample in sample_metrics
        for _, sample in enumerate(self.sample_metrics):
            sample_id = sample['sample_id']
            original_size = sample['original_size']
            transformed_sizes = sample['transformed_sizes']
            preprocessing_times = sample['preprocessing_times']
            compressed_sizes = sample['compressed_sizes']
            compression_times = sample['compression_times']

            # Track cumulative preprocessing time
            cumulative_time = 0
            best_efficiency = 0
            best_plan = None
            LOGGER.debug(
                f"Processing sample {sample_id}: original_size={original_size}"
            )

            # Check offloading plan without compression
            for i, transformed_size in enumerate(transformed_sizes):
                cumulative_time += preprocessing_times[i]
                size_reduction = original_size - transformed_size if transformed_size else 0
                if cumulative_time > 0 and size_reduction > 0:
                    efficiency = size_reduction / cumulative_time
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_plan = (sample_id, size_reduction, cumulative_time, i + 1, False)

            # Check offloading plan with compression
            cumulative_time = 0
            for i, (compressed_size, compression_time) in enumerate(zip(compressed_sizes, compression_times)):
                cumulative_time += preprocessing_times[i]
                cumulative_time_with_compression = cumulative_time + compression_time
                size_reduction = original_size - compressed_size if compressed_size else 0
                if cumulative_time_with_compression > 0 and size_reduction > 0:
                    efficiency = size_reduction / cumulative_time_with_compression
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_plan = (sample_id, size_reduction, cumulative_time_with_compression, i + 1, True)

            # Insert the best offloading plan into the heap if it has positive efficiency
            if best_efficiency > 0:
                heapq.heappush(offloading_heap, (-best_efficiency, best_plan))
                LOGGER.info(f"Sample {sample_id} added to offloading plan with efficiency {best_efficiency:.4f}, "
                      f"stage {best_plan[3]}, compression: {best_plan[4]}")

        return offloading_heap

    def iterative_offloading(self):
        total_preprocessing_time_compute = sum([sum(sample['preprocessing_times']) for sample in self.sample_metrics])
        total_preprocessing_time_storage = 0
        current_data_traffic = sum([sample['original_size'] for sample in self.sample_metrics])

        current_tcc = total_preprocessing_time_compute / self.cpu_cores_compute  
        current_tcs = 0
        current_tnet = current_data_traffic / self.io_bandwidth
        current_offloading_plan = {}  

        offloading_plan = {}
        decisions = self.decide_offloading()

        for _, (sample_id,sample_size_reduction, cumulative_time, stage, compression_used) in decisions:
            current_data_traffic -= sample_size_reduction
            total_preprocessing_time_compute -= cumulative_time
            total_preprocessing_time_storage += cumulative_time

            current_tcc = total_preprocessing_time_compute / self.cpu_cores_compute  
            current_tcs = total_preprocessing_time_storage / self.cpu_cores_storage
            current_tnet = current_data_traffic / self.io_bandwidth

            LOGGER.info(f"Sample {sample_id} selected for offloading: stage={stage}, "
                  f"compression_used={compression_used}, current_tnet={current_tnet}, current_tcs={current_tcs}, current_tcc={current_tcc}")

            current_offloading_plan[sample_id] = stage
            offloading_plan[sample_id] = (stage, compression_used)

            # Check if offloading should stop based on TNet and TCS comparison
            if current_tnet < current_tcs and current_tnet < current_tcc:
                LOGGER.info(f"Stopping offloading: current_tnet={current_tnet} is less than current_tcs={current_tcs} and current_tcc={current_tcc}")
                break

        return offloading_plan

    def send_offloading_requests(self):
        # Create a gRPC channel to communicate with the storage server
        channel = grpc.insecure_channel(f"{self.grpc_host}:{self.grpc_port}")
        stub = data_feed_pb2_grpc.DataFeedStub(channel)

        # Create a stream for offloading requests
        for request in self._generate_offloading_requests():
            try:
                # LOGGER.info(f"Sending Offloading Request: Sample {request.sample_id}, Transformations: {request.transformations}, Compress: {request.compress}")
                response = stub.update_offloading_plan(request, timeout=5.0)
                # LOGGER.info(f"Server Response: {response.status}")
            except Exception as e:
                LOGGER.info(f"Error while sending offloading request: {e}")

    def _generate_offloading_requests(self):
        # Create offloading plan and generate gRPC messages for the server
        offloading_plan = self.iterative_offloading()

        # Iterate through the offloading plan and yield OffloadingRequest messages
        for sample_id, (transformations, is_compressed) in offloading_plan.items():
            yield data_feed_pb2.OffloadingRequest(
                sample_id=sample_id,
                transformations=transformations,
                compress=is_compressed
            )

def test_decision_engine():
    # Sample test data for 3 samples
    sample_metrics = [
        {
            "original_size": 10000,
            "transformed_sizes": [8000, 6000, 4000, 3000, 2500],  # Sizes after each stage without compression
            "preprocessing_times": [0.5, 0.6, 0.4, 0.3, 0.2],      # Time taken for each stage's transformation
            "compressed_sizes": [7500, 5500, 3500, 2800, 2300],     # Sizes after each stage with compression
            "compression_times": [0.1, 0.1, 0.1, 0.1, 0.1]          # Compression time for each stage
        },
        {
            "original_size": 12000,
            "transformed_sizes": [10000, 8000, 6000, 4500, 4000],
            "preprocessing_times": [0.7, 0.5, 0.6, 0.4, 0.3],
            "compressed_sizes": [9000, 7000, 5000, 4200, 3800],
            "compression_times": [0.1, 0.1, 0.1, 0.1, 0.1]
        },
        {
            "original_size": 15000,
            "transformed_sizes": [12000, 10000, 8000, 6000, 5000],
            "preprocessing_times": [0.8, 0.7, 0.5, 0.4, 0.3],
            "compressed_sizes": [11000, 9000, 7000, 5500, 4800],
            "compression_times": [0.1, 0.1, 0.1, 0.1, 0.1]
        }
    ]

    # Example values for other parameters
    io_bandwidth = 10000000  # IO bandwidth in bytes/s
    cpu_cores_compute = 4  # Number of CPU cores on compute node
    cpu_cores_storage = 2  # Number of CPU cores on storage node

    # Initialize the DecisionEngine with test data and gRPC details
    engine = DecisionEngine(
        sample_metrics=sample_metrics,
        io_bandwidth=io_bandwidth,
        cpu_cores_compute=cpu_cores_compute,
        cpu_cores_storage=cpu_cores_storage,
        grpc_host='localhost',
        grpc_port=50051
    )

    # Generate the offloading plan using iterative_offloading and print the result
    # offloading_plan = engine.iterative_offloading()
    # LOGGER.info("Generated Offloading Plan:")
    # for sample_id, (stage, compression_used) in offloading_plan.items():
    #     LOGGER.info(f"Sample {sample_id}: Stage {stage}, Compression Used: {compression_used}")
    
    # Send offloading requests to the storage server
    engine.send_offloading_requests()


# Run the test function
if __name__ == '__main__':
    test_decision_engine()

    
