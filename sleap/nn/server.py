# import multiprocessing as mp
import threading
import zmq
import jsonpickle
import time
import socket

def run_data_gen(labels_file, results):
#     local_data = threading.local()
#     local_data.results = dict()
#     results = dict()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://*:9001")
    pub.send_string(jsonpickle.encode(dict(event="starting data_gen",)))
#     pub.send(umsgpack.packb(dict(event="starting data_gen")))

    from time import time, sleep
    results["ready"] = False
    results["times"] = dict()
    results["times"]["start_time"] = time()

    results["times"]["start_io"] = time()
    from sleap.io.dataset import Labels
    labels = Labels.load_json(labels_file)
    results["times"]["end_io"] = time()

    # TESTING: just use a few frames
    labels.labeled_frames = labels.labeled_frames[0:7]    

    from sleap.nn.datagen import generate_images, generate_confidence_maps

    results["times"]["start_imgs"] = time()
    results["imgs"] = generate_images(labels)
    results["times"]["end_imgs"] = time()
    
    results["times"]["start_conf"] = time()
    results["confmaps"] = generate_confidence_maps(labels, sigma=5)
    results["times"]["end_conf"] = time()
    
    results["times"]["end_time"] = time()
    
    results["times"]["total"] = results["times"]["end_time"] - results["times"]["start_time"]
    results["times"]["io"] = results["times"]["end_io"] - results["times"]["start_io"]
    results["times"]["imgs"] = results["times"]["end_imgs"] - results["times"]["start_imgs"]
    results["times"]["conf"] = results["times"]["end_conf"] - results["times"]["start_conf"]
    
    results["ready"] = True

    pub.send_string(jsonpickle.encode(dict(event="data_gen done",results=results)))
#     pub.send(umsgpack.packb(dict(event="data_gen done",results=results)))
    pub.close()
    print("done with data_gen")

if __name__ == "__main__":

    training_data = dict()

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    print(f"server starting at {ip_address}")
    ctx = zmq.Context()

    # Controller
    ctrl = ctx.socket(zmq.SUB)
    ctrl.subscribe("")
    ctrl.bind("tcp://*:9000")

    def poll(timeout=10):
        if ctrl.poll(timeout, zmq.POLLIN):
#             return umsgpack.unpackb(sub.recv())
            return jsonpickle.decode(ctrl.recv_string())
        return None

    is_running = True
    while is_running:
        msg = poll()
        if msg is not None:
            print(msg)
            if msg["command"] == "stop":
                is_running = False
                break
            elif msg["command"] == "data_gen":
                #data_gen(msg["labels_file"], training_data)
                threading.Thread(target=run_data_gen, args=(msg["labels_file"], training_data)).start()

        time.sleep(1)
    print("server finished")