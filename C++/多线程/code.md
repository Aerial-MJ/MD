## 多线程
```c++
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex mtx;   // 互斥锁，用于保护临界区
int counter = 0;   // 临界资源
std::condition_variable cv;
bool stop= false;

void increment(int i) {
    // 临界区：修改 counter 的代码
    std::unique_lock<std::mutex> lock(mtx);
    std::cout<<i<<"+++++++"<<std::endl;
    cv.wait(lock,[]{return stop;}); // 释放锁并等待通知
    ++counter;
    std::cout<<i<<"-------"<<std::endl;
    std::cout<<i<<"-------"<<std::endl;
    std::cout<<i<<"-------"<<std::endl;
    std::cout<<i<<"-------"<<std::endl;
    std::cout << counter << std::endl;
}

int main() {
    // 创建多个线程，模拟并发访问
    std::thread t1(increment, 1);
    std::thread t2(increment, 2);
    std::thread t3(increment, 3);


    // 通过 notify_all 唤醒所有等待的线程
    //std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 确保所有线程都开始等待

    stop= true;
//    cv.notify_all();  // 唤醒所有线程

    t1.join();
    t2.join();
    t3.join();

    std::cout << "Final counter value: " << counter << std::endl;

    return 0;
}

```

在这段代码中，`std::condition_variable` 被用来使得线程在修改共享资源 `counter` 时能通过条件变量等待一个通知 (`cv.wait(lock, []{return stop;})`)。但即使没有显式调用 `notify_one` 或 `notify_all`，程序仍然能够“完美运行”，原因如下：

### 1. cv.wait 的阻塞与唤醒

`cv.wait(lock, []{return stop;})` 会使线程进入等待状态，并且释放 `mtx` 锁，直到条件满足（即 `stop == true`）。而且，当条件变量被触发时，线程会自动重新获得 `mtx` 锁并继续执行。

### 2. stop 变量的变化

在 `main` 函数中，你将 `stop` 变量设置为 `true`（即 `stop = true;`）。这个修改是在所有线程启动后（即线程进入等待状态之前）进行的。

在这段代码中，由于 `stop` 在 `main` 线程中已经被设置为 `true`，所有的线程在 `cv.wait` 中都能看到这个条件已经满足。因此，线程在等待时会立即唤醒并继续执行。

### 3. 没有显式通知的原因

实际上，`std::condition_variable` 的作用是让线程在条件不满足时进行等待，直到条件发生变化。通常，你需要显式调用 `notify_one` 或 `notify_all` 来通知等待的线程条件已经满足，进而唤醒它们。

但是，`cv.wait` 也支持直接检查传入的条件（在这里就是 `stop == true`）。如果条件已经是 `true`，线程会立即继续执行，而不需要等待通知。因此，虽然没有显式调用 `notify_one` 或 `notify_all`，但 `stop` 变量已经被设置为 `true`，这使得等待的线程会立刻重新获得锁并继续执行。
