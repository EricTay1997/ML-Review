# Concurrency in Python

- Credit to [Shiv](https://github.com/shivgodhia) for some parts of this. 
- OS Basics ([OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/))
  - The operating system (OS) is system software that manages computer hardware and software resources
    - Virtual machine: The OS takes physical resources, such as a CPU, memory, or disk, and virtualizes them.
    - Resource manager: The OS handles tough and tricky issues related to concurrency
    - File system: The OS stores files persistently, thus making them safe over the long-term.
  - Program: Code, data, etc.
  - Process: The process is a running program. 
  - Virtualization
    - Each process accesses its own private virtual address space (sometimes just called its address space), which the OS somehow maps onto the physical memory of the machine.
    - The address space of a process contains all of the memory state of the running program
      - Code
      - Stack: keep track of where it is in the function call chain as well as to allocate local variables and pass parameters and return values to and from routines. Grows negatively. 
      - Heap: dynamically-allocated, user-managed memory. It grows positively
    - Segmentation
      - Place each segment (code, stack heap) independently in physical memory, thus avoiding filling physical memory with unused virtual address space.
      - We chop up physical memory into variable-sized pieces.
      - External fragmentation: physical memory quickly becomes full of little holes of free space, making it difficult to allocate new segments, or to grow existing ones
      - Internal fragmentation: An allocator hands out chunks of memory bigger than that requested. Any unasked for (and thus unused) space in such a chunk is wasted.
    - Paging
      - We chop up physical memory into fixed-sized pieces.
      - Page table: Per-process data structure that stores address translations for each of the virtual pages of the address space
      - Speeding up address translations with translation-lookaside buffers (TLBs)
        - Part of the chip’s memory-management unit (MMU), functions as a cache
    - Virtual memory > physical memory
      - Enabled with a swap space: we swap pages out of memory to it and swap pages into memory from it
      - Page fault: A page is not present and has been swapped to disk
        - OS looks in the PTE to find the address, and issues the request to disk to fetch the page into memory
        - Thrashing is when there are many page faults
  - Concurrency
    - Thread: A thread is a part of a process that shares the same memory (address space) and resources (data) with other threads in the same process
    - Semaphore: A semaphore is an object with an integer value that we can manipulate with two routines
  - Threads vs processes
  - Thread / Process Communication
    - Process/Process
      - IPC mechanisms include Pipes, Named Pipes (FIFOs), Message Queues, Shared Memory, Semaphores, and Sockets
      - Processes need to use the kernel to communicate since memory is isolated, adding overhead.
    - Thread/Thread
      - Communication is often through shared memory within the same process, because threads share the same memory space
      - Requires Mutexes/Condition Variables/Semaphores for synchronization to prevent data corruption
    - Process/Thread
      - Threads communicate with their parent process by sharing the process's memory space. This includes the heap, code, and static data segments.
      - **Shared Variables:** The most common method; threads can directly read and write to shared variables. Requires synchronization to avoid race conditions.
      - **Signals:** The process can send signals to threads, but the interpretation is context-dependent. Commonly used to signal events such as thread cancellation.
      - **Callbacks/Function Pointers:** While technically a shared memory mechanism, the process can set function pointers that threads execute. It provides a structured way of communicating work or requesting tasks.
      - **Thread Creation and Joining:** The parent process can create and join threads, which can indirectly be considered communication by the creation and synchronisation of the threads.
      - **Message Passing (within the same process):** Although not as clearly a process-to-thread mechanism, message-passing concepts (like queues or channels) can exist within the memory space shared by the process and its threads. 
- Multithreading vs Multiprocessing
  - Multithreading (`threading`) is the ability of a processor to execute multiple threads concurrently.
    - Useful in I/O with a lot of latency (rather than performing computations)
      - The GIL ensures that only one thread executes Python bytecode at a time, but it doesn't prevent threads from being created or from performing I/O operations. 
      - When one thread is waiting for I/O, the GIL is released, allowing other threads to run and do useful work.
    - Precautions have to be taken in case threads write to the same memory at the same time, since threads run in the same memory space
    - The Global Interpreter Lock (GIL) synchronizes the execution of threads
    - In CPython, this means that only one thread can execute at a time
  - Multiprocessing (`multiprocessing`) is the ability of a system to run multiple processors in parallel
    - Processes have separate memory
    - Harder to share objects between processes
- Global Interpreter Lock (GIL)
  - Mutex (thread lock) ensuring that only one thread controls the interpreter at a time
  - In place to prevent race conditions with memory and reference allocation
  - Particularly important when Python interacts with (multithreaded) C-extensions.
    - Because Python utilizes reference counting in memory management, running Python from multiple threads could lead to memory leaks or worse, crashing the program.
- Computers and Latency
  - CPU Instruction: 0.01ns
  - Memory reference: 100 ns
  - Read 1MB from memory: 3 microseconds
  - Read 1MB from disk: 825 microseconds
  - Disk seek: 2 milliseconds
  - Ping USA to Europe: 150 milliseconds
  - Access RAM, Access Disk, Access Network
  - I/O Bound - the program spends more time waiting than running instructions
- Python libraries
  - I/O-Bound processes
    - `threading`
    - `asyncio`
  - CPU-Bound processes
    - `multiprocessing` (This can also be used for I/O-bound processes but is a lot more heavy weight)
  - Comparisons:
    - `asyncio` vs `threading`
      - `asyncio` is single-threaded
      - `asyncio` uses cooperative multitasking. `threading` uses preemptive. 
        - `asyncio`: The tasks decide when to give up control.
        - `threading`: The operating system decides when to switch tasks external to Python.
      - `asyncio` takes fewer resources and less time than creating threads. Single-threaded nature may also make it less buggy.
        - The operating system maintains data structures for each thread, known as thread control blocks, which store information about the thread's state, priority, and other attributes. These structures consume system memory.
        - Threads share instruction, global and heap regions but has its own individual stack and registers. The operating system allocates a portion of memory for each thread's execution context, which includes things like the call stack, local variables, and function execution state.
      - `asyncio` needs special asynchronous versions of libraries to gain the full advantage of `asyncio`. For example, `requests` isn't designed to notify the event loop that it’s blocked.
    - `threading` vs `multiprocessing`

|**Aspect**|**Process**|**Thread**|
|---|---|---|
|**Independence**|Independent execution unit.|Dependent on the parent process.|
|**Memory**|Each process has its own memory space.|Threads share the same memory space.|
|**Communication**|Inter-process communication (IPC) required.|Easier communication via shared memory.|
|**Overhead**|High overhead (context switching is costly).|Low overhead (faster context switching).|
|**Creation Time**|Slower to create a process.|Faster to create a thread.|
|**Crash Impact**|A process crash does not affect other processes.|A thread crash can crash the entire process.|
|**Resources**|Allocated separate resources (file handles, etc.).|Shares resources (file handles, sockets, etc.).|

## Examples

### Threading

```python
import concurrent.futures
import threading
import queue

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    a = ex.map(f, arr) # non-blocking
    b = ex.submit(f, *args) #non-blocking
    for val in a:
      print(a) # blocking
    b.result() # blocking

lock = threading.Lock()
with lock:
    pass

event = threading.Event()
event.set()
event.clear()
event.is_set()

s = threading.Semaphore(value=10)
s.acquire() # decrements
print(s._value)
s.release()

q = queue.Queue(maxsize = 10)
q.put(message)
q.get()
q.task_done()
q.join()
q.empty()
```

### Asyncio

```python
import asyncio
import aiohttp

async def fetch_data(url):
    await asyncio.sleep(2)  # Simulate a network request that takes 2 seconds
    return f"Data from {url}"
async def main():
    data = await fetch_data("https://example.com")  # Pause until fetch_data is complete
    print(f"Received: {data}")
    
async def main():
    async with aiohttp.ClientSession() as session:
        async with session.get('http://httpbin.org/get') as resp:
            await resp.json()

async def download_site(session, url):
    async with session.get(url) as resp:
        await resp.json()

async def main(sites):
    async with aiohttp.ClientSession() as session:
        tasks = [download_site(session, url) for url in sites]
        await asyncio.gather(*tasks, return_exceptions=True)

async with aiofiles.open('filename', mode='r') as f:
    contents = await f.read()
async with aiofiles.open('filename') as f:
    async for line in f:
        ...
async with aiofiles.open('filename', 'w') as f:
    await f.write(write_data)

asyncio.run(main())
loop = asyncio.get_event_loop()
await loop.run_in_executor(None, blocking_f()) # Multithreading
with concurrent.futures.ProcessPoolExecutor() as pool:
  result = await loop.run_in_executor(pool, cpu_bound) # Multiprocessing
producers = [asyncio.create_task(produce(n, q)) for n in range(nprod)]
await asyncio.gather(*producers)
q = asyncio.Queue(maxsize = 10)
await q.put(message)
await q.get()
q.task_done()
await q.join()
```

### Multiprocessing
```python
import multiprocessing

def init_worker(shared_queue):
  global queue
  queue = shared_queue
def f(message):
  global queue
  queue.put(message)

shared_queue = multiprocessing.Queue(maxsize = 10)
with multiprocessing.Pool(initializer = init_worker, initargs = (shared_queue,)) as pool:
    a = pool.map(f, arr) # blocking
    b = pool.apply_async(f, *args) # non-blocking
    c = pool.map_async(f, arr) # non-block
    b.get() # blocking
    c.get() # blocking

# Same syntax as threading: multiprocessing.Lock(), multiprocessing.Event(), multiprocessing.Semaphore(), q.get(), q.empty()
```
See [SuperFastPython](https://superfastpython.com/multiprocessing-pool-shared-global-variables/)

## Asynchronous programming

### What is asynchronous programming, and how does it differ from synchronous programming?
- **Asynchronous Programming:** A programming paradigm that allows a unit of work to run separately from the main application thread. When the work is complete, it notifies the main thread whether the operation was successful or not.
    - **Key Idea:** Instead of waiting for a long-running operation (e.g., network request, file I/O) to complete, the program can continue executing other tasks. When the operation is finished, a callback function is executed or a result is made available.
    - **Non-blocking I/O:** Asynchronous programming relies on non-blocking I/O operations, which means that a function call returns immediately, even if the operation is not complete.
- **Synchronous Programming:** The traditional programming model where code is executed sequentially, one line after another. When a long-running operation is encountered, the program waits for it to complete before moving to the next line.
    - **Blocking I/O:** Synchronous programs typically use blocking I/O operations, where a function call doesn't return until the operation is complete.
- **Differences:**
    - **Execution Flow:** Asynchronous programs can execute multiple tasks concurrently (but not necessarily in parallel), while synchronous programs execute tasks one at a time.
    - **Responsiveness:** Asynchronous programs can remain responsive even when performing long-running operations, as they don't block the main thread. Synchronous programs can become unresponsive while waiting for an operation to complete.
    - **Complexity:** Asynchronous programs can be more complex to write and debug due to the use of callbacks, futures, or async/await.

### `asyncio`
- **`asyncio`:** Python's built-in library for writing asynchronous code using the `async` and `await` syntax.
    - **Event Loop:** The core of an `asyncio` program is the event loop, which is responsible for scheduling and running asynchronous tasks.
      - `asyncio.run()` is responsible for getting the event loop, running tasks until they are marked as complete, and then closing the event loop.
      - Alternatively, we do
      - ```python
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()```
    - **Coroutines:** Functions defined with `async def` are called coroutines. They can be paused and resumed at `await` points.
      - Coroutines are repurposed generators that take advantage of the peculiarities of generator methods
    - **`async`:**
      - Used to define a coroutine function.
      - A coroutine function can be paused at `await` points and resumed later.
      - When called, a coroutine function returns a coroutine object. It doesn't execute the function body immediately.
      - `async for` iterates over an asynchronous iterator
    - **`await`:**
      - Used inside a coroutine to pause execution until the awaited coroutine, task, or future is complete.
      - Can only be used inside `async def` functions.
      - `await` gives control back to the event loop, allowing other tasks to run.
      - When the awaited object is complete, the coroutine resumes execution from where it left off, and the value of the `await` expression is the result of the awaited object.
      - Replaces `yield from`
    - **Tasks:** Represent the execution of a coroutine. Created using `asyncio.create_task()`.
      - We can do `a = await asyncio.gather(t, t2)` to put a collection of coroutines (futures) into a single future. `await` then means that we're waiting for all tasks to be completed.
    - **Futures:** Low-level objects that represent the result of an asynchronous operation that may not be complete yet.

#### What are some common use cases for asynchronous programming?
- **I/O-bound tasks:** Network programming (e.g., web servers, API clients), file I/O, database operations, where the program spends a lot of time waiting for external resources.
- **Concurrency:** Handling multiple connections or requests concurrently without using threads or processes.
- **GUI programming:** Keeping the user interface responsive while performing background tasks.
- **Web scraping:** Fetching and processing multiple web pages concurrently.
- **Real-time applications:** Handling streams of data or events in a non-blocking way.