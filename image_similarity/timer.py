import time
from image_similarity.stats_writer import write_stats_to_file_and_console

times = 10

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        for x in range(0, times):
            result = method(*args, **kw)
        te = time.time()
        write_stats_to_file_and_console(f'Images: {args[1]}, {args[2]}\nTime: { format((te - ts) / times * 1000, ".2f")} ms\n')
        return result
    return timed


