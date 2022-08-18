from multiprocessing import Array, Process, Queue, Value
import numpy as np
import os
import queue
import tensorflow as tf

class MultiprocessDataGenerator:
    def __init__(self, batch_size=32, buffer_size=5, num_workers=1):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers

        self.__workers = []
        self.__is_running = Value('b', False)
        self.__ready_batches = Queue(buffer_size)
        self.__stale_batches = Queue(buffer_size)
        self.__current_batch = 0
        self.__is_initialized = False

        self.config = {}
        self.data = None # Loaded during `start`
        self.buffered_batches = []
        
    # To Override ----------------------------------------------------------------------------------

    def data_signature(self):
        """
        return (output_shapes, output_types)
        """
        raise NotImplementedError("Must implement batch buffers")


    def load_data(self):
        return None
    
    
    @staticmethod
    def generate_batch(data, batch_size, config, global_config):
        raise NotImplementedError("Must implement batch generation")
    
    
    @staticmethod
    def worker_init(config, global_config):
        pass
    
    
    @staticmethod
    def worker_clean(config, global_config):
        print("Nothing to clean.")

    # Internal Processing --------------------------------------------------------------------------
    
    def __add_batch_array(self, shape, dtype):
        if dtype == np.float32:
            arr_type = 'f'
        elif dtype == np.int32:
            arr_type = 'i'
        else:
            raise Exception("Batch array data type not supported.")
            
        arr_shape = np.concatenate(((self.buffer_size,), shape)).astype(np.int32)
        arr_size = int(np.product(arr_shape))
        arr = np.frombuffer(Array(arr_type, arr_size, lock=False), dtype=dtype).reshape(arr_shape)
        self.buffered_batches.append(arr)
        
    
    def __load_data_signature(self):
        output_shapes, output_types = self.data_signature()
        if len(output_shapes) != len(output_types):
            raise Exception("Data signature shape/type lengths do not match!")
        output_shapes = tuple([self.batch_size,] + list(shape) for shape in output_shapes)
        return output_shapes, output_types
    
    
    def test_sig(self):
        output_shapes, output_types = self.__load_data_signature()
        output_shapes = tuple(tf.TensorShape(shape) for shape in output_shapes)
        return output_shapes, output_types


    def init(self):
        # Load the data (if applicable)
        self.data = self.load_data()
        
        # Configure the batch information
        output_shapes, output_types = self.__load_data_signature()
        for (shape, dtype) in zip(output_shapes, output_types):
            self.__add_batch_array(shape, dtype)
        for i in range(1, self.buffer_size):
            self.__stale_batches.put(i)
        self.__is_initialized = True
        
    
    def __iter__(self):
        return self

        
    def __next__(self):
        # Mark the current batch as stale
        self.__stale_batches.put(self.__current_batch)
        self.__current_batch = self.__ready_batches.get()
        return tuple(buffer[self.__current_batch] for buffer in self.buffered_batches)
    
    
    def __del__(self):
        if self.__is_running.value:
            self.stop()
            
            
    @staticmethod
    def worker(worker_id, is_running, buffered_batches, stale_batches, ready_batches, batch_size, data, global_config, init, clean, generate_batch):
        config = {"id": worker_id}
        init(config, global_config)
        try:
            while is_running.value and os.getppid() != 1:
                try:
                    batch_id = stale_batches.get(timeout=1.0)
                    for i, subbatch in enumerate(generate_batch(data, batch_size, config, global_config)):
                        buffered_batches[i][batch_id] = subbatch
                    ready_batches.put(batch_id)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    continue
        except Exception as e:
            clean(config, global_config)
            raise e
        clean(config, global_config)
        
        
    # Interface ------------------------------------------------------------------------------------
    
    def is_running(self):
        return self.__is_running.value
    

    def start(self):
        assert not self.__is_running.value, "Workers are already running"
        if not self.__is_initialized:
            self.init()
        args = (
            self.__is_running,
            self.buffered_batches,
            self.__stale_batches,
            self.__ready_batches,
            self.batch_size,
            self.data,
            self.config,
            self.__class__.worker_init,
            self.__class__.worker_clean,
            self.__class__.generate_batch)
        self.__is_running.value = True
        for i in range(self.num_workers):
            worker = Process(target=MultiprocessDataGenerator.worker, args=(i,) + args)
            worker.start()
            self.__workers.append(worker)


    def stop(self):
        assert self.__is_running.value, "Workers are already stopped"
        self.__is_running.value = False
        for worker in self.__workers:
            worker.join()
        self.__workers = []
            
            
    def terminate(self):
        self.__is_running.value = False
        for worker in self.__workers:
            worker.terminate()
        self.__workers = []
        
    # Wrappers -------------------------------------------------------------------------------------
        
    def as_generator(self):
        while True:
            yield next(self)
        
            
    def as_dataset(self):
        output_shapes, output_types = self.__load_data_signature()
        output_shapes = tuple(tf.TensorShape(shape) for shape in output_shapes)
        return tf.data.Dataset.from_generator(
            self.as_generator,
            output_shapes=output_shapes,
            output_types=output_types)
    
    
    def as_dist_dataset(self, strategy):
        dataset = self.as_dataset()
        return strategy.experimental_distribute_dataset(dataset)