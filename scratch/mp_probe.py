import multiprocessing as mp
import time

def worker(shared_t, i):
    while True:
        with shared_t.get_lock():
            t = shared_t.value
            shared_t.value += 1
        if t >= 10:
            break
        print(f"Worker {i} got {t}")
        time.sleep(0.1)

if __name__ == '__main__':
    shared_t = mp.Value('i', 0)
    procs = []
    for i in range(2):
        p = mp.Process(target=worker, args=(shared_t, i))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
