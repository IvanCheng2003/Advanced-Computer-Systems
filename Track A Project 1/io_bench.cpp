#include <bits/stdc++.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

using namespace std;

double now_s() {
    using clock = chrono::steady_clock;
    return chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Mode 1: traditional read() into a user buffer
int run_read(const string &path, size_t block_size) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        close(fd);
        return 1;
    }
    off_t file_size = st.st_size;
    cout << "File size: " << file_size << " bytes\n";

    vector<char> buf(block_size);
    volatile unsigned long long checksum = 0;

    double t0 = now_s();
    while (true) {
        ssize_t n = read(fd, buf.data(), block_size);
        if (n < 0) {
            perror("read");
            close(fd);
            return 1;
        }
        if (n == 0) break; // EOF

        for (ssize_t i = 0; i < n; ++i) {
            checksum += (unsigned char)buf[i];
        }
    }
    double t1 = now_s();

    double dt = t1 - t0;
    double bytes = (double)file_size;
    double bw = bytes / dt / 1e9; // GB/s

    cout << "Mode=read\n";
    cout << "Time_s=" << dt << "\n";
    cout << "Bandwidth_GBps=" << bw << "\n";
    cout << "Checksum=" << checksum << "\n";

    close(fd);
    return 0;
}

// Mode 2: mmap() the file and scan over the mapping
int run_mmap(const string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        close(fd);
        return 1;
    }
    off_t file_size = st.st_size;
    cout << "File size: " << file_size << " bytes\n";

    void *addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    volatile unsigned long long checksum = 0;
    unsigned char *p = (unsigned char *)addr;

    double t0 = now_s();
    for (off_t i = 0; i < file_size; ++i) {
        checksum += p[i];
    }
    double t1 = now_s();

    double dt = t1 - t0;
    double bytes = (double)file_size;
    double bw = bytes / dt / 1e9; // GB/s

    cout << "Mode=mmap\n";
    cout << "Time_s=" << dt << "\n";
    cout << "Bandwidth_GBps=" << bw << "\n";
    cout << "Checksum=" << checksum << "\n";

    munmap(addr, file_size);
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " read|mmap <file> [block_size_bytes]\n";
        cerr << "Example (read): " << argv[0] << " read testfile.bin 65536\n";
        cerr << "Example (mmap): " << argv[0] << " mmap testfile.bin\n";
        return 1;
    }

    string mode = argv[1];
    string path = argv[2];

    if (mode == "read") {
        size_t block_size = 64 * 1024; // 64 KiB default
        if (argc >= 4) {
            block_size = stoull(argv[3]);
        }
        return run_read(path, block_size);
    } else if (mode == "mmap") {
        return run_mmap(path);
    } else {
        cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }
}
